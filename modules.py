from torch import nn
import torch
import torch.nn.functional as F
import numbers
from einops import rearrange


def normalize(x, power=1):
    power_emp = torch.mean(x ** 2)
    x = (power / power_emp) ** 0.5 * x
    return power_emp, x


class DepthToSpace(torch.nn.Module):
    def __init__(self, block_size):
        super().__init__()
        self.bs = block_size

    def forward(self, x):
        N, C, H, W = x.size()
        x = x.view(N, self.bs, self.bs, C // (self.bs ** 2), H, W)
        x = x.permute(0, 3, 4, 1, 5, 2).contiguous()
        x = x.view(N, C // (self.bs ** 2), H * self.bs, W * self.bs)
        return x


def awgn(snr, x, device):
    # snr(db)
    n = 1 / (10 ** (snr / 10))
    sqrt_n = n ** 0.5
    noise = torch.randn_like(x) * sqrt_n
    noise = noise.to(device)
    x_hat = x + noise
    return x_hat


# ================ MSA 相关组件 ================

def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')

def to_4d(x, h, w):
    return rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)

class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(BiasFree_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma+1e-5) * self.weight

class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma+1e-5) * self.weight + self.bias

class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type='WithBias'):
        super(LayerNorm, self).__init__()
        if LayerNorm_type =='BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)

class FeedForward(nn.Module):
    def __init__(self, dim, ffn_expansion_factor=2.66, bias=False):
        super(FeedForward, self).__init__()
        hidden_features = int(dim*ffn_expansion_factor)
        self.project_in = nn.Conv2d(dim, hidden_features*2, kernel_size=1, bias=bias)
        self.dwconv = nn.Conv2d(hidden_features*2, hidden_features*2, kernel_size=3, stride=1, padding=1, groups=hidden_features*2, bias=bias)
        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x = F.gelu(x1) * x2
        x = self.project_out(x)
        return x

class SemanticAttention(nn.Module):
    """语义注意力模块 - 基于MSA的三分支注意力"""
    def __init__(self, dim, num_heads=8, bias=False):
        super(SemanticAttention, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        # 三个分支的QKV
        self.qkv_f = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)  # foreground
        self.qkv_b = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)  # background  
        self.qkv_g = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)  # global
        
        # 深度卷积
        self.dwconv_f = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim, bias=bias)
        self.dwconv_b = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim, bias=bias)
        self.dwconv_g = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim, bias=bias)
        
        self.project_out = nn.Conv2d(dim*3, dim, kernel_size=1, bias=bias)
        
        # 语义mask生成 - 自适应生成前景/背景mask
        self.mask_gen = nn.Sequential(
            nn.Conv2d(dim, dim//4, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim//4, 1, kernel_size=1),
            nn.Sigmoid()
        )
    
    def forward(self, x, external_mask=None):
        b, c, h, w = x.shape
        
        # 生成或使用外部语义mask
        if external_mask is not None:
            semantic_mask = F.interpolate(external_mask, size=(h, w), mode='bilinear', align_corners=False)
        else:
            semantic_mask = self.mask_gen(x)
        
        fg_mask = semantic_mask
        bg_mask = 1 - semantic_mask
        
        # 前景注意力分支 - 专注于重要语义区域
        q_f = self.dwconv_f(self.qkv_f(x))
        k_f = self.dwconv_f(self.qkv_f(x))
        v_f = self.dwconv_f(self.qkv_f(x))
        # 应用前景mask
        q_f = q_f * fg_mask
        k_f = k_f * fg_mask
        q_f = rearrange(q_f, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k_f = rearrange(k_f, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v_f = rearrange(v_f, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        q_f = F.normalize(q_f, dim=-1)
        k_f = F.normalize(k_f, dim=-1)
        attn_f = (q_f @ k_f.transpose(-2, -1)) * self.temperature
        attn_f = attn_f.softmax(dim=-1)
        out_f = (attn_f @ v_f)
        out_f = rearrange(out_f, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)
        # 背景注意力分支 - 处理背景区域
        q_b = self.dwconv_b(self.qkv_b(x))
        k_b = self.dwconv_b(self.qkv_b(x))
        v_b = self.dwconv_b(self.qkv_b(x))
        # 应用背景mask
        q_b = q_b * bg_mask
        k_b = k_b * bg_mask
        q_b = rearrange(q_b, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k_b = rearrange(k_b, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v_b = rearrange(v_b, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        
        q_b = F.normalize(q_b, dim=-1)
        k_b = F.normalize(k_b, dim=-1)
        attn_b = (q_b @ k_b.transpose(-2, -1)) * self.temperature
        attn_b = attn_b.softmax(dim=-1)
        out_b = (attn_b @ v_b)
        out_b = rearrange(out_b, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)
        
        # 全局注意力分支 - 捕获全局语义关系
        q_g = self.dwconv_g(self.qkv_g(x))
        k_g = self.dwconv_g(self.qkv_g(x))
        v_g = self.dwconv_g(self.qkv_g(x))
        
        q_g = rearrange(q_g, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k_g = rearrange(k_g, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v_g = rearrange(v_g, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        
        q_g = F.normalize(q_g, dim=-1)
        k_g = F.normalize(k_g, dim=-1)
        attn_g = (q_g @ k_g.transpose(-2, -1)) * self.temperature
        attn_g = attn_g.softmax(dim=-1)
        out_g = (attn_g @ v_g)
        out_g = rearrange(out_g, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)
        
        # 融合三个分支 - element-wise产品+求和策略
        out = torch.cat([out_f, out_b, out_g], dim=1)
        out = self.project_out(out)
        
        return out, semantic_mask

class MSA_Block(nn.Module):
    """MSA增强块 - 集成语义注意力和前馈网络"""
    def __init__(self, dim, num_heads=8, ffn_expansion_factor=2.66, bias=False, LayerNorm_type='WithBias'):
        super(MSA_Block, self).__init__()
        
        self.norm1 = LayerNorm(dim, LayerNorm_type)
        self.attn = SemanticAttention(dim, num_heads, bias)
        self.norm2 = LayerNorm(dim, LayerNorm_type)
        self.ffn = FeedForward(dim, ffn_expansion_factor, bias)

    def forward(self, x, external_mask=None):
        # 注意力增强
        attn_out, semantic_mask = self.attn(self.norm1(x), external_mask)
        x = x + attn_out
        
        # 前馈网络
        x = x + self.ffn(self.norm2(x))
        
        return x, semantic_mask


# ================ 增强的ResNet组件 ================

class EnhancedResidualBlock(nn.Module):
    """增强的残差块，集成MSA语义注意力"""
    def __init__(self, inchannel, outchannel, stride=1, use_msa=False, num_heads=4):
        super(EnhancedResidualBlock, self).__init__()
        self.use_msa = use_msa
        
        self.left = nn.Sequential(
            nn.Conv2d(inchannel, outchannel, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(outchannel),
            nn.PReLU(),
            nn.Conv2d(outchannel, outchannel, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(outchannel)
        )
        
        self.shortcut = nn.Sequential()
        if stride != 1 or inchannel != outchannel:
            self.shortcut = nn.Sequential(
                nn.Conv2d(inchannel, outchannel, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(outchannel)
            )
        # MSA语义注意力增强
        if use_msa and outchannel >= 64:  # 只在通道数足够大时使用MSA
            self.msa_block = MSA_Block(outchannel, num_heads=min(num_heads, outchannel//16))
        
        self.prelu = nn.PReLU()

    def forward(self, x, external_mask=None):
        out = self.left(x)
        out += self.shortcut(x)
        out = self.prelu(out)
        semantic_mask = None
        if self.use_msa and hasattr(self, 'msa_block'):
            out, semantic_mask = self.msa_block(out, external_mask)
        return out, semantic_mask


class Enhanced_Encoder(nn.Module):
    """增强的编码器，集成MSA语义注意力机制"""
    def __init__(self, config):
        super(Enhanced_Encoder, self).__init__()
        self.config = config
        self.inchannel = 64
        
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.PReLU(),
        )
        
        # 逐层增加MSA，在更深层使用更强的语义理解
        self.layer1 = self.make_layer(EnhancedResidualBlock, 64, 1, stride=1, use_msa=False)
        self.layer2 = self.make_layer(EnhancedResidualBlock, 128, 1, stride=2, use_msa=True, num_heads=2)
        self.layer3 = self.make_layer(EnhancedResidualBlock, 256, 2, stride=2, use_msa=True, num_heads=4)
        
        if config.mod_method == 'bpsk':
            self.layer4 = self.make_layer(EnhancedResidualBlock, config.channel_use, 2, stride=2, use_msa=True, num_heads=8)
        else:
            self.layer4 = self.make_layer(EnhancedResidualBlock, config.channel_use * 2, 2, stride=2, use_msa=True, num_heads=8)

    def make_layer(self, block, channels, num_blocks, stride, use_msa=False, num_heads=4):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for i, stride in enumerate(strides):
            # 只在最后一个块使用MSA以避免计算开销过大
            use_msa_block = use_msa and (i == num_blocks - 1)
            layers.append(block(self.inchannel, channels, stride, use_msa_block, num_heads))
            self.inchannel = channels
        return nn.ModuleList(layers)

    def forward(self, x):
        z0 = self.conv1(x)
        
        semantic_mask = None
        # Layer 1
        for layer in self.layer1:
            z0, mask = layer(z0, semantic_mask)
            if mask is not None:
                semantic_mask = mask
        
        # Layer 2 
        z1 = z0
        for layer in self.layer2:
            z1, mask = layer(z1, semantic_mask)
            if mask is not None:
                semantic_mask = mask
        
        # Layer 3
        z2 = z1
        for layer in self.layer3:
            z2, mask = layer(z2, semantic_mask)
            if mask is not None:
                semantic_mask = mask
        
        # Layer 4
        z3 = z2
        for layer in self.layer4:
            z3, mask = layer(z3, semantic_mask)
            if mask is not None:
                semantic_mask = mask
        
        return z3, semantic_mask


class Enhanced_Decoder_Recon(nn.Module):
    """增强的重建解码器，集成渐进式细化和MSA引导"""
    def __init__(self, config):
        super(Enhanced_Decoder_Recon, self).__init__()
        self.config = config

        if config.mod_method == 'bpsk':
            input_channel = int(config.channel_use / (4 * 4))
        else:
            input_channel = int(config.channel_use * 2 / (4 * 4))

        self.conv1 = nn.Sequential(
            nn.Conv2d(input_channel, 256, 1, 1, 0),
            nn.PReLU())

        self.inchannel = 256

        # 使用MSA增强的解码层
        self.layer1 = nn.Sequential(
            self.make_layer(EnhancedResidualBlock, 256, 2, stride=1, use_msa=True),
            nn.PReLU())

        self.layer2 = nn.Sequential(
            self.make_layer(EnhancedResidualBlock, 256, 2, stride=1, use_msa=True),
            nn.PReLU())

        self.DepthToSpace1 = DepthToSpace(4)

        self.conv2 = nn.Sequential(
            nn.Conv2d(16, 128, 1, 1, 0),
            nn.PReLU())

        self.inchannel = 128

        self.layer3 = nn.Sequential(
            self.make_layer(EnhancedResidualBlock, 128, 2, stride=1, use_msa=True),
            nn.PReLU())

        self.DepthToSpace2 = DepthToSpace(2)

        self.conv3 = nn.Sequential(
            nn.Conv2d(32, 3, 1, 1, 0))
        
        # 渐进式监督预测头
        self.progressive_heads = nn.ModuleList([
            nn.Conv2d(256, 3, kernel_size=3, padding=1),  # 4x4 -> 预测
            nn.Conv2d(128, 3, kernel_size=3, padding=1),  # 16x16 -> 预测
            nn.Conv2d(32, 3, kernel_size=3, padding=1)    # 32x32 -> 预测
        ])

    def make_layer(self, block, channels, num_blocks, stride, use_msa=False):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for i, stride in enumerate(strides):
            use_msa_block = use_msa and (i == num_blocks - 1)
            layers.append(block(self.inchannel, channels, stride, use_msa_block))
            self.inchannel = channels
        return nn.ModuleList(layers)

    def forward(self, z, semantic_mask=None):
        z0 = self.conv1(z.reshape(z.shape[0], -1, 4, 4))
        
        # 渐进式细化解码
        progressive_preds = []
        
        # Stage 1: 4x4 特征细化
        z1 = z0
        for layer in self.layer1[:-1]:  # 除了最后的PReLU
            if hasattr(layer, '__iter__'):
                for sublayer in layer:
                    z1, _ = sublayer(z1, semantic_mask)
            else:
                z1, _ = layer(z1, semantic_mask)
        z1 = self.layer1[-1](z1)  # PReLU
        
        pred1 = self.progressive_heads[0](z1)  # 4x4预测
        pred1_up = F.interpolate(pred1, size=(32, 32), mode='bilinear', align_corners=False)
        progressive_preds.append(pred1_up)
        
        # Stage 2: 上采样到16x16
        z2 = z1
        for layer in self.layer2[:-1]:
            if hasattr(layer, '__iter__'):
                for sublayer in layer:
                    z2, _ = sublayer(z2, semantic_mask)
            else:
                z2, _ = layer(z2, semantic_mask)
        z2 = self.layer2[-1](z2)  # PReLU
        
        z3 = self.DepthToSpace1(z2)
        z4 = self.conv2(z3)
        
        pred2 = self.progressive_heads[1](z4)  # 16x16预测
        pred2_up = F.interpolate(pred2, size=(32, 32), mode='bilinear', align_corners=False)
        progressive_preds.append(pred2_up)
        
        # Stage 3: 最终32x32重建
        z5 = z4
        for layer in self.layer3[:-1]:
            if hasattr(layer, '__iter__'):
                for sublayer in layer:
                    z5, _ = sublayer(z5, semantic_mask)
            else:
                z5, _ = layer(z5, semantic_mask)
        z5 = self.layer3[-1](z5)  # PReLU
        
        z5 = self.DepthToSpace2(z5)
        
        pred3 = self.progressive_heads[2](z5)  # 32x32预测
        progressive_preds.append(pred3)
        
        final_output = self.conv3(z5)
        progressive_preds.append(final_output)
        
        return final_output, progressive_preds


# ================ 保持原有的分类解码器 ================

class Decoder_Class(nn.Module):
    def __init__(self, half_width, layer_width):
        super(Decoder_Class, self).__init__()
        self.layer_width = layer_width
        self.Half_width = half_width
        
        # 增加语义增强预处理
        self.semantic_enhance = nn.Sequential(
            nn.Linear(half_width * 2, half_width * 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(half_width * 2, half_width * 2),
            nn.ReLU()
        )
        
        self.fc_spinal_layer1 = nn.Sequential(
            nn.Dropout(p=0.5), nn.Linear(self.Half_width, self.layer_width),
            nn.PReLU(),
        )
        self.fc_spinal_layer2 = nn.Sequential(
            nn.Dropout(p=0.5), nn.Linear(self.Half_width + self.layer_width, self.layer_width),
            nn.PReLU(),
        )
        self.fc_spinal_layer3 = nn.Sequential(
            nn.Dropout(p=0.5), nn.Linear(self.Half_width + self.layer_width, self.layer_width),
            nn.PReLU(),
        )
        self.fc_spinal_layer4 = nn.Sequential(
            nn.Dropout(p=0.5), nn.Linear(self.Half_width + self.layer_width, self.layer_width),
            nn.PReLU(),
        )
        
        # 增强的最终分类层
        self.last_fc = nn.Sequential(
            nn.Linear(self.layer_width * 4, self.layer_width * 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(self.layer_width * 2, 10)
        )

    def forward(self, z):
        # 语义增强预处理
        z_enhanced = self.semantic_enhance(z)
        
        x1 = self.fc_spinal_layer1(z_enhanced[:, 0:self.Half_width])
        x2 = self.fc_spinal_layer2(torch.cat([z_enhanced[:, self.Half_width:2 * self.Half_width], x1], dim=1))
        x3 = self.fc_spinal_layer3(torch.cat([z_enhanced[:, 0:self.Half_width], x2], dim=1))
        x4 = self.fc_spinal_layer4(torch.cat([z_enhanced[:, self.Half_width:2 * self.Half_width], x3], dim=1))
        
        x = torch.cat([x1, x2, x3, x4], dim=1)
        y_class = self.last_fc(x)
        return y_class


# ================ 向后兼容的原始组件 ================

# 保持原有组件以确保兼容性
ResidualBlock = EnhancedResidualBlock  # 别名以保持兼容
Encoder = Enhanced_Encoder
Decoder_Recon = Enhanced_Decoder_Recon

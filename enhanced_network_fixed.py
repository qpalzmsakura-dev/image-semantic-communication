from torch import nn
import torch
from torch.nn.functional import gumbel_softmax
from torch.nn import init
from enhanced_modules import (
    Enhanced_Encoder, Enhanced_Decoder_Recon, Decoder_Class, 
    awgn, normalize, EnhancedResidualBlock
)


def modulation(logits, device, mod_method='bpsk'):
    discrete_code = gumbel_softmax(logits, hard=True, tau=1.5)

    if mod_method == 'bpsk':
        output = discrete_code[:, :, 0] * (-1) + discrete_code[:, :, 1] * 1

    elif mod_method == '4qam':
        const = [1, -1]
        const = torch.tensor(const).to(device)
        temp = discrete_code * const
        output = torch.sum(temp, dim=2)

    elif mod_method == '16qam':
        const = [-3, -1, 1, 3]
        const = torch.tensor(const).to(device)
        temp = discrete_code * const
        output = torch.sum(temp, dim=2)

    elif mod_method == '64qam':
        const = [-7, -5, -3, -1, 1, 3, 5, 7]
        const = torch.tensor(const).to(device)
        temp = discrete_code * const
        output = torch.sum(temp, dim=2)

    else:
        print("Modulation method not defined.")

    return output


class Enhanced_JCM(nn.Module):
    def __init__(self, config, device):
        super(Enhanced_JCM, self).__init__()
        self.config = config
        self.device = device

        # define the number of probability categories
        if self.config.mod_method == 'bpsk':
            self.num_category = 2
        elif self.config.mod_method == '4qam':
            self.num_category = 2
        elif self.config.mod_method == '16qam':
            self.num_category = 4
        elif self.config.mod_method == '64qam':
            self.num_category = 8

        self.encoder = Enhanced_Encoder(self.config)

        if config.mod_method == 'bpsk':
            self.prob_convs = nn.Sequential(
                nn.Linear(config.channel_use * 4 * 4, config.channel_use * self.num_category),
                nn.ReLU(),
            )
        else:
            self.prob_convs = nn.Sequential(
                nn.Linear(config.channel_use * 2 * 4 * 4, config.channel_use * 2 * self.num_category),
                nn.ReLU(),
            )

        self.decoder_recon = Enhanced_Decoder_Recon(self.config)

        if self.config.mod_method == 'bpsk':
            self.decoder_class = Decoder_Class(int(config.channel_use / 2), int(config.channel_use / 8))
        else:
            self.decoder_class = Decoder_Class(int(config.channel_use * 2 / 2), int(config.channel_use * 2 / 8))

        self.initialize_weights()

    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                init.xavier_normal_(m.weight, gain=1)
            elif isinstance(m, nn.Conv2d):
                init.xavier_uniform_(m.weight, gain=1)

    def reparameterize(self, probs):
        mod_method = self.config.mod_method
        code = modulation(probs, self.device, mod_method)
        return code

    def forward(self, x):
        x_encoded, semantic_mask = self.encoder(x)
        x_f = x_encoded.reshape(x.shape[0], -1)
        
        
        z = self.prob_convs(x_f).reshape(x.shape[0], -1, self.num_category)
        
        
        code = self.reparameterize(z)

        power, z_normalized = normalize(code)

        
        if self.config.mode == 'train':
            z_hat = awgn(self.config.snr_train, z_normalized, self.device)
        elif self.config.mode == 'test':
            z_hat = awgn(self.config.snr_test, z_normalized, self.device)

        
        recon_final, progressive_preds = self.decoder_recon(z_hat, semantic_mask)
        
        r_class = self.decoder_class(z_hat)

        
        self.progressive_preds = progressive_preds
        
        
        return code, z_normalized, z_hat, r_class, recon_final

    def compute_progressive_loss(self, progressive_preds, target, loss_weights=None):
       
        if loss_weights is None:
            loss_weights = [0.1, 0.2, 0.3, 1.0]         
        progressive_loss = 0
        mse_loss = nn.MSELoss()
        
        for i, pred in enumerate(progressive_preds):
            if i < len(loss_weights):
                progressive_loss += loss_weights[i] * mse_loss(pred, target)
        
        return progressive_loss

    def load_pretrained_weights(self, checkpoint_path, strict=False):
       
        try:
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            
            if strict:
               
                self.load_state_dict(checkpoint)
            else:
                model_dict = self.state_dict()
                
                matched_dict = {}
                for k, v in checkpoint.items():
                    if k in model_dict and model_dict[k].shape == v.shape:
                        matched_dict[k] = v
                
                model_dict.update(matched_dict)
                self.load_state_dict(model_dict)
                
                
                loaded_keys = set(matched_dict.keys())
                model_keys = set(model_dict.keys())
                missing_keys = model_keys - loaded_keys
                
                print(f"Successfully loaded {len(loaded_keys)} layers from checkpoint")
                if missing_keys:
                    print(f"{len(missing_keys)} new layers will be initialized randomly")
                    
        except Exception as e:
            print(f"Error loading checkpoint: {e}")
            print("Training from scratch...")


class JCM(Enhanced_JCM):
    
    def __init__(self, config, device):
        super(JCM, self).__init__(config, device)
    
    def forward(self, x):
        
        code, z_normalized, z_hat, r_class, recon_final = super().forward(x)
    
        return code, z_normalized, z_hat, r_class, recon_final


class JCM_Enhanced(Enhanced_JCM):
    def forward(self, x):
        code, z_normalized, z_hat, r_class, recon_final = super().forward(x)
        return code, z_normalized, z_hat, r_class, recon_final, self.progressive_preds

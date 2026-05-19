import os
import torch
from torch import optim, nn
from torch.nn.utils import clip_grad_norm_
import numpy as np
import pandas as pd
from tqdm import tqdm
import math
from evaluation import EVAL
from utils import save_checkpoint, PSNR


def train_enhanced(config, net, train_iter, test_iter, device):
    """
    增强的训练函数，支持MSA和渐进式监督
    🚨 EMERGENCY FIXES: 添加权重衰减、标签平滑、早停机制
    """
    learning_rate = config.lr
    epochs = config.train_iters

    # 🚨 MODIFIED: 优化器添加权重衰减
    ignored_params = list(map(id, net.prob_convs.parameters()))
    base_params = filter(lambda p: id(p) not in ignored_params, net.parameters())
    
    weight_decay = getattr(config, 'weight_decay', 1e-4)
    optimizer = optim.Adam([
        {'params': base_params, 'weight_decay': weight_decay},  # 🚨 添加权重衰减
        {'params': net.prob_convs.parameters(), 'lr': learning_rate/2, 'weight_decay': weight_decay/10}  # 🚨 prob_convs也添加较小的权重衰减
    ], learning_rate)

    # 🚨 MODIFIED: 损失函数添加标签平滑
    loss_f1 = nn.CrossEntropyLoss(label_smoothing=0.1)  # 🚨 添加标签平滑
    loss_f2 = nn.MSELoss()
    
    results = {'epoch': [], 'acc': [], 'mse': [], 'psnr': [], 'ssim': [], 'loss': [], 
               'progressive_loss': [], 'semantic_loss': []}
    
    # Choose scheduler based on config
    if config.use_scheduler:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=config.t_max, eta_min=config.min_lr)
    else:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=config.train_iters+1, T_mult=1, eta_min=1e-6, last_epoch=-1)

    # 渐进式监督的权重配置
    progressive_weights = getattr(config, 'progressive_weights', [0.1, 0.2, 0.3, 1.0])
    use_progressive = getattr(config, 'use_progressive_supervision', True)
    progressive_lambda = getattr(config, 'progressive_lambda', 0.5)

    # 🚨 NEW: 早停机制
    best_acc = 0
    patience_counter = 0
    patience = getattr(config, 'early_stopping_patience', 15)
    best_model_state = None
    
    print(f"🚨 EMERGENCY FIXES ACTIVE:")
    print(f"   Weight Decay: {weight_decay}")
    print(f"   Label Smoothing: 0.1")
    print(f"   Early Stopping Patience: {patience}")
    print(f"   Learning Rate: {learning_rate}")
    
    for epoch in range(epochs):
        net.train()
        epoch_loss = []
        epoch_progressive_loss = []
        acc_total_train = 0
        psnr_total_train = 0
        
        for i, (X, Y) in enumerate(tqdm(train_iter)):
            X, Y = X.to(device), Y.to(device)

            optimizer.zero_grad()
            
            # 前向传播 - 根据网络类型决定如何处理返回值
            if net.__class__.__name__ == 'JCM_Enhanced':
                # JCM_Enhanced 返回6个值
                code, z_normalized, z_hat, y_class, y_recon, progressive_preds = net(X)
            else:
                # JCM 或 Enhanced_JCM 返回5个值，progressive_preds存储在属性中
                code, z_normalized, z_hat, y_class, y_recon = net(X)
                progressive_preds = getattr(net, 'progressive_preds', None)

            # 主要损失
            loss_1 = loss_f1(y_class, Y)                      # 🚨 使用带标签平滑的分类损失
            loss_2 = loss_f2(y_recon, X)                      # 重建损失
            
            total_loss = loss_1 + config.tradeoff_lambda * loss_2

            # 渐进式监督损失
            progressive_loss_value = 0
            if use_progressive and progressive_preds is not None:
                progressive_loss_value = compute_progressive_loss(
                    progressive_preds, X, progressive_weights, loss_f2
                )
                total_loss += progressive_lambda * progressive_loss_value

            # 反向传播
            total_loss.backward()
            
            # 梯度裁剪
            if config.use_grad_clip:
                clip_grad_norm_(net.parameters(), config.clip_grad)
            
            optimizer.step()
            
            # 记录损失
            epoch_loss.append(total_loss.cpu().item())
            epoch_progressive_loss.append(progressive_loss_value.cpu().item() if isinstance(progressive_loss_value, torch.Tensor) else progressive_loss_value)

            # 训练集准确率和PSNR
            acc = (y_class.data.max(1)[1] == Y.data).float().sum()
            acc_total_train += acc
            psnr = PSNR(X, y_recon.detach())
            psnr_total_train += psnr

        scheduler.step()

        # 计算平均损失
        avg_loss = sum(epoch_loss) / len(epoch_loss)
        avg_progressive_loss = sum(epoch_progressive_loss) / len(epoch_progressive_loss)
        acc_train = acc_total_train / 50000
        psnr_train = psnr_total_train / 50000

        # 验证
        acc, mse, psnr, ssim = EVAL(net, test_iter, device, config, epoch)
        
        # 🚨 NEW: 早停逻辑
        acc_num = acc.detach().cpu().numpy()
        if acc_num > best_acc:
            best_acc = acc_num
            patience_counter = 0
            best_model_state = net.state_dict().copy()  # 保存最佳模型状态
            print(f'🎯 *** NEW BEST ACCURACY: {best_acc:.3f}% ***')
        else:
            patience_counter += 1
            print(f'⏳ No improvement for {patience_counter}/{patience} epochs (best: {best_acc:.3f}%)')
        
        # 早停检查
        if patience_counter >= patience:
            print(f'🛑 EARLY STOPPING triggered at epoch {epoch}!')
            print(f'🔄 Restoring best model with accuracy: {best_acc:.3f}%')
            if best_model_state:
                net.load_state_dict(best_model_state)
            break
        
        # 🚨 ENHANCED: 过拟合检测 (修复CUDA tensor问题)
        train_acc_cpu = acc_train.cpu().item() if torch.is_tensor(acc_train) else acc_train
        train_test_gap = train_acc_cpu * 100 - acc_num
        if train_test_gap > 15:
            print(f'🚨 WARNING: Large train-test gap detected: {train_test_gap:.1f}%')
        elif train_test_gap > 10:
            print(f'⚠️  Moderate overfitting detected: {train_test_gap:.1f}%')
        
        # 打印训练信息
        print('epoch: {:d}, loss: {:.6f}, progressive_loss: {:.6f}, acc: {:.3f}, mse: {:.6f}, psnr: {:.3f}, ssim: {:.3f}, lr: {:.6f}'.format(
              epoch, avg_loss, avg_progressive_loss, acc, mse, psnr, ssim, optimizer.state_dict()['param_groups'][0]['lr']))
        print('train acc: {:.3f}, train psnr: {:.3f}, train-test gap: {:.1f}%'.format(acc_train, psnr_train, train_test_gap))

        # 记录结果
        results['epoch'].append(epoch)
        results['loss'].append(avg_loss)
        results['progressive_loss'].append(avg_progressive_loss)
        results['semantic_loss'].append(0)  # 预留语义损失
        results['acc'].append(acc_num)
        results['mse'].append(mse)
        results['psnr'].append(psnr)
        results['ssim'].append(ssim)

        # 保存最佳模型 - 🚨 修改保存逻辑，使用早停的最佳模型
        if (epochs - epoch) <= 10 and acc_num == best_acc:  # 只在是当前最佳时保存
            file_name = config.model_path + '/{}/'.format(config.mod_method)
            if not os.path.exists(file_name):
                os.makedirs(file_name)
            model_name = 'CIFAR_SNR{:.3f}_Trans{:d}_{}_enhanced.pth.tar'.format(
                config.snr_train, config.channel_use, config.mod_method)
            save_checkpoint(net.state_dict(), file_name + model_name)
            print(f'💾 Best model saved: {model_name}')

    # 保存训练结果
    data = pd.DataFrame(results)
    file_name = config.result_path + '/{}/'.format(config.mod_method)
    if not os.path.exists(file_name):
        os.makedirs(file_name)

    result_name = 'CIFAR_SNR{:.3f}_Trans{:d}_{}_enhanced.csv'.format(
            config.snr_train, config.channel_use, config.mod_method)
    data.to_csv(file_name + result_name, index=False, header=False)
    
    print(f'🎯 Training completed! Final best accuracy: {best_acc:.3f}%')


def train_standard(config, net, train_iter, test_iter, device):
    """
    原有的标准训练函数
    🚨 EMERGENCY FIXES: 添加权重衰减、标签平滑、早停机制
    """
    learning_rate = config.lr
    epochs = config.train_iters

    # 🚨 MODIFIED: 优化器添加权重衰减
    ignored_params = list(map(id, net.prob_convs.parameters()))
    base_params = filter(lambda p: id(p) not in ignored_params, net.parameters())
    
    weight_decay = getattr(config, 'weight_decay', 1e-4)
    optimizer = optim.Adam([
        {'params': base_params, 'weight_decay': weight_decay},  # 🚨 添加权重衰减
        {'params': net.prob_convs.parameters(), 'lr': learning_rate/2, 'weight_decay': weight_decay/10}  # 🚨 prob_convs也添加权重衰减
    ], learning_rate)

    # 🚨 MODIFIED: 损失函数添加标签平滑
    loss_f1 = nn.CrossEntropyLoss(label_smoothing=0.1)  # 🚨 添加标签平滑
    loss_f2 = nn.MSELoss()
    results = {'epoch': [], 'acc': [], 'mse': [], 'psnr': [], 'ssim': [], 'loss': []}
    
    # Choose scheduler based on config
    if config.use_scheduler:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=config.t_max, eta_min=config.min_lr)
    else:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=config.train_iters+1, T_mult=1, eta_min=1e-6, last_epoch=-1)

    # 🚨 NEW: 早停机制
    best_acc = 0
    patience_counter = 0
    patience = getattr(config, 'early_stopping_patience', 15)
    best_model_state = None
    
    print(f"🚨 STANDARD TRAINING WITH EMERGENCY FIXES:")
    print(f"   Weight Decay: {weight_decay}")
    print(f"   Label Smoothing: 0.1")
    print(f"   Early Stopping Patience: {patience}")
    
    for epoch in range(epochs):
        net.train()
        epoch_loss = []
        acc_total_train = 0
        psnr_total_train = 0
        
        for i, (X, Y) in enumerate(tqdm(train_iter)):
            X, Y = X.to(device), Y.to(device)

            optimizer.zero_grad()
            
            # 确保只获取5个返回值
            outputs = net(X)
            if len(outputs) == 6:
                code, z_normalized, z_hat, y_class, y_recon, _ = outputs
            else:
                code, z_normalized, z_hat, y_class, y_recon = outputs

            loss_1 = loss_f1(y_class, Y)  # 🚨 使用带标签平滑的损失
            loss_2 = loss_f2(y_recon, X)

            loss = loss_1 + config.tradeoff_lambda * loss_2

            loss.backward()
            if config.use_grad_clip:
                clip_grad_norm_(net.parameters(), config.clip_grad)
            optimizer.step()
            epoch_loss.append(loss.cpu().item())

            # acc & psnr of the train set
            acc = (y_class.data.max(1)[1] == Y.data).float().sum()
            acc_total_train += acc
            psnr = PSNR(X, y_recon.detach())
            psnr_total_train += psnr

        scheduler.step()

        loss = sum(epoch_loss) / len(epoch_loss)
        acc_train = acc_total_train / 50000
        psnr_train = psnr_total_train / 50000

        acc, mse, psnr, ssim = EVAL(net, test_iter, device, config, epoch)
        
        # 🚨 NEW: 早停逻辑
        acc_num = acc.detach().cpu().numpy()
        if acc_num > best_acc:
            best_acc = acc_num
            patience_counter = 0
            best_model_state = net.state_dict().copy()
            print(f'🎯 *** NEW BEST ACCURACY: {best_acc:.3f}% ***')
        else:
            patience_counter += 1
            print(f'⏳ No improvement for {patience_counter}/{patience} epochs (best: {best_acc:.3f}%)')
        
        # 早停检查
        if patience_counter >= patience:
            print(f'🛑 EARLY STOPPING triggered at epoch {epoch}!')
            print(f'🔄 Restoring best model with accuracy: {best_acc:.3f}%')
            if best_model_state:
                net.load_state_dict(best_model_state)
            break
        
        # 🚨 ENHANCED: 过拟合检测 (修复CUDA tensor问题)
        train_acc_cpu = acc_train.cpu().item() if torch.is_tensor(acc_train) else acc_train
        train_test_gap = train_acc_cpu * 100 - acc_num
        if train_test_gap > 15:
            print(f'🚨 WARNING: Large train-test gap detected: {train_test_gap:.1f}%')
        elif train_test_gap > 10:
            print(f'⚠️  Moderate overfitting detected: {train_test_gap:.1f}%')
        
        print('epoch: {:d}, loss: {:.6f}, acc: {:.3f}, mse: {:.6f}, psnr: {:.3f}, ssim: {:.3f}, lr: {:.6f}'.format
              (epoch, loss, acc, mse, psnr, ssim, optimizer.state_dict()['param_groups'][0]['lr']))
        print('train acc: {:.3f}, train psnr: {:.3f}, train-test gap: {:.1f}%'.format(acc_train, psnr_train, train_test_gap))

        results['epoch'].append(epoch)
        results['loss'].append(loss)
        results['acc'].append(acc_num)
        results['mse'].append(mse)
        results['psnr'].append(psnr)
        results['ssim'].append(ssim)

        # 保存最佳模型 - 🚨 修改保存逻辑
        if (epochs - epoch) <= 10 and acc_num == best_acc:
            file_name = config.model_path + '/{}/'.format(config.mod_method)
            if not os.path.exists(file_name):
                os.makedirs(file_name)
            model_name = 'CIFAR_SNR{:.3f}_Trans{:d}_{}.pth.tar'.format(
                config.snr_train, config.channel_use, config.mod_method)
            save_checkpoint(net.state_dict(), file_name + model_name)
            print(f'💾 Best model saved: {model_name}')

    # 保存结果
    data = pd.DataFrame(results)
    file_name = config.result_path + '/{}/'.format(config.mod_method)
    if not os.path.exists(file_name):
        os.makedirs(file_name)

    result_name = 'CIFAR_SNR{:.3f}_Trans{:d}_{}.csv'.format(
            config.snr_train, config.channel_use, config.mod_method)
    data.to_csv(file_name + result_name, index=False, header=False)
    
    print(f'🎯 Training completed! Final best accuracy: {best_acc:.3f}%')


def compute_progressive_loss(progressive_preds, target, loss_weights, loss_fn):
    """
    计算渐进式监督损失
    """
    if progressive_preds is None or len(progressive_preds) == 0:
        return 0
    
    progressive_loss = 0
    for i, pred in enumerate(progressive_preds):
        if i < len(loss_weights):
            progressive_loss += loss_weights[i] * loss_fn(pred, target)
    
    return progressive_loss


def train(config, net, train_iter, test_iter, device):
    """
    智能训练函数调度器
    🚨 EMERGENCY FIXES: 所有训练方式都应用正则化措施
    """
    # 检查是否使用增强特性
    use_enhanced = getattr(config, 'use_enhanced_training', True)
    use_progressive = getattr(config, 'use_progressive_supervision', True)
    
    # 根据网络类型和配置决定训练方式
    network_class = net.__class__.__name__
    
    print(f"🚨 EMERGENCY ANTI-OVERFITTING MEASURES ACTIVATED!")
    print(f"Network type: {network_class}")
    print(f"Enhanced training: {use_enhanced}")
    print(f"Progressive supervision: {use_progressive}")
    
    if use_enhanced and use_progressive and network_class in ['Enhanced_JCM', 'JCM_Enhanced', 'JCM']:
        print("Using enhanced training with MSA and progressive supervision...")
        train_enhanced(config, net, train_iter, test_iter, device)
    else:
        print("Using standard training with emergency fixes...")
        train_standard(config, net, train_iter, test_iter, device)

import torchvision
from torch.utils.data import DataLoader
import torch
import torchvision.transforms as transforms
from torch.optim.lr_scheduler import CosineAnnealingLR
from enhanced_network_fixed import JCM, JCM_Enhanced 
from enhanced_train_fixed import train
from evaluation import EVAL
from utils import init_seeds
import os
import argparse


def mischandler(config):
    if not os.path.exists(config.model_path):
        os.makedirs(config.model_path)
    if not os.path.exists(config.result_path):
        os.makedirs(config.result_path)


def smart_load_checkpoint(net, config, device):

    if not config.load_checkpoint:
        print(" Training from scratch...")
        return
        
    
    enhanced_model_name = '/{}/'.format(config.mod_method) + \
                 'CIFAR_SNR{:.3f}_Trans{:d}_{}_enhanced.pth.tar'.format(
                     config.snr_train, config.channel_use, config.mod_method)
    standard_model_name = '/{}/'.format(config.mod_method) + \
                 'CIFAR_SNR{:.3f}_Trans{:d}_{}.pth.tar'.format(
                     config.snr_train, config.channel_use, config.mod_method)
    
    enhanced_path = config.model_path + enhanced_model_name
    standard_path = config.model_path + standard_model_name
    
    checkpoint_loaded = False
    
    
    if os.path.exists(enhanced_path):
        print(f" Found enhanced checkpoint: {enhanced_path}")
        try:
            if hasattr(net, 'load_pretrained_weights'):
                net.load_pretrained_weights(enhanced_path, strict=False)
            else:
                checkpoint = torch.load(enhanced_path, map_location=device)
                net.load_state_dict(checkpoint, strict=False)
            print("Enhanced checkpoint loaded successfully!")
            checkpoint_loaded = True
        except Exception as e:
            print(f"  Failed to load enhanced checkpoint: {e}")
    
    if not checkpoint_loaded and os.path.exists(standard_path):
        print(f" Found standard checkpoint: {standard_path}")
        try:
            if hasattr(net, 'load_pretrained_weights'):
                net.load_pretrained_weights(standard_path, strict=False)
            else:
                checkpoint = torch.load(standard_path, map_location=device)
                
                model_dict = net.state_dict()
                matched_dict = {}
                
                for k, v in checkpoint.items():
                    if k in model_dict and model_dict[k].shape == v.shape:
                        matched_dict[k] = v
                
                model_dict.update(matched_dict)
                net.load_state_dict(model_dict)
                
                print(f"Standard checkpoint loaded! {len(matched_dict)}/{len(model_dict)} layers matched")
            checkpoint_loaded = True
        except Exception as e:
            print(f" Failed to load standard checkpoint: {e}")
    
    if not checkpoint_loaded:
        print(" No compatible checkpoint found, training from scratch...")


def main(config):
    # initialize random seed
    init_seeds()

    # prepare training & test data - 🚨 MODIFIED: Enhanced data augmentation
    transform_train = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=10),  # 新增：随机旋转
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),  # 新增：颜色抖动
        transforms.ToTensor(),
        transforms.RandomErasing(p=0.1, scale=(0.02, 0.2)),  # 🚨 FIXED: 移到ToTensor()之后
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
    ])

    train_data = torchvision.datasets.CIFAR10(
        root=config.dataset_path,
        train=True,
        transform=transform_train,
        download=True
    )
    test_data = torchvision.datasets.CIFAR10(
        root=config.dataset_path,
        train=False,
        transform=transform_test,
        download=True
    )

    train_loader = DataLoader(dataset=train_data, batch_size=config.batch_size, shuffle=True)
    test_loader = DataLoader(dataset=test_data, batch_size=config.batch_size, shuffle=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    
    if config.use_enhanced_network:
        print(f" Using Enhanced JCM with MSA semantic attention...")
        net = JCM_Enhanced(config, device).to(device)
        model_suffix = "_enhanced"
    else:
        print(f"Using standard JCM (with MSA backbone but compatible interface)...")
        net = JCM(config, device).to(device)
        model_suffix = ""

   
    smart_load_checkpoint(net, config, device)

    if config.mode == 'train':
        print("Training with the modulation scheme {}.".format(config.mod_method.upper()))
        if config.use_enhanced_network and config.use_progressive_supervision:
            print("Enhanced training with MSA attention and progressive supervision enabled.")
        
        # 🚨 EMERGENCY FIX: Print regularization settings
        print("\n" + "="*50)
        print("🚨 EMERGENCY OVERFITTING FIXES APPLIED:")
        print(f"   Learning Rate: {config.lr}")
        print(f"   Weight Decay: {getattr(config, 'weight_decay', 'Not set')}")
        print(f"   Label Smoothing: Enabled (0.1)")
        print(f"   Early Stopping: Enabled (patience=15)")
        print(f"   Enhanced Data Augmentation: Enabled")
        print("="*50 + "\n")
        
        train(config, net, train_loader, test_loader, device)

    elif config.mode == 'test':
        print(" Start Testing.")
        acc, mse, psnr, ssim = EVAL(net, test_loader, device, config)
        print(' Results: acc: {:.3f}, mse: {:3f}, psnr: {:.3f}, ssim: {:.3f}'.format(acc, mse, psnr, ssim))

    else:
        print("?Wrong mode input! Please use 'train' or 'test'.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

   
    # model hyper-parameters
    parser.add_argument('--channel_use', type=int, default=128)
    parser.add_argument('--mod_method', type=str, default='64qam', 
                       help='Available: bpsk, 4qam, 16qam, 64qam')
    parser.add_argument('--load_checkpoint', type=int, default=1)

    # training hyper-parameters - 🚨 MODIFIED: Reduced learning rate
    parser.add_argument('--train_iters', type=int, default=300)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-4)  # 🚨 CHANGED: from 2e-4 to 1e-4
    parser.add_argument('--weight_decay', type=float, default=1e-4)  # 🚨 NEW: Weight decay for regularization
    
    # learning rate scheduler parameters
    parser.add_argument('--use_scheduler', type=int, default=1, 
                       help='1: use CosineAnnealingLR, 0: use CosineAnnealingWarmRestarts')     
    parser.add_argument('--min_lr', type=float, default=1e-6, 
                       help='minimum learning rate for scheduler')  
    parser.add_argument('--t_max', type=int, default=100, 
                       help='maximum number of iterations for CosineAnnealingLR')    
    
    parser.add_argument('--snr_train', type=float, default=18)
    
    # gradient clipping parameters
    parser.add_argument('--use_grad_clip', type=int, default=1, 
                       help='1: enable gradient clipping, 0: disable')
    parser.add_argument('--clip_grad', type=float, default=1.0, 
                       help='gradient clipping threshold')

    parser.add_argument('--snr_test', type=float, default=18)
    parser.add_argument('--tradeoff_lambda', type=float, default=100,
                       help='The tradeoff hyperparameter lambda between two tasks')

    # misc
    parser.add_argument('--dataset', type=str, default='cifar')
    parser.add_argument('--mode', type=str, default='train')
    parser.add_argument('--model_path', type=str, default='./models')
    parser.add_argument('--result_path', type=str, default='./results')
    parser.add_argument('--dataset_path', type=str, default='./dataset')

    
    # MSA Enhancement Parameters
    parser.add_argument('--use_enhanced_network', type=int, default=1,
                       help='1: use MSA-enhanced JCM, 0: use standard JCM with MSA backbone')
    parser.add_argument('--use_enhanced_training', type=int, default=1,
                       help='1: enable enhanced training features, 0: standard training')
    
    # Progressive Supervision Parameters  
    parser.add_argument('--use_progressive_supervision', type=int, default=1,
                       help='1: enable progressive supervision, 0: disable')
    parser.add_argument('--progressive_lambda', type=float, default=0.5,
                       help='weight for progressive supervision loss')
    parser.add_argument('--progressive_weights', type=str, default='0.1,0.2,0.3,1.0',
                       help='comma-separated weights for progressive stages')
    
    # MSA Attention Parameters
    parser.add_argument('--msa_num_heads', type=int, default=8,
                       help='number of attention heads in MSA modules')
    parser.add_argument('--msa_ffn_expansion', type=float, default=2.66,
                       help='FFN expansion factor in MSA modules')
    
    # Semantic Enhancement Parameters
    parser.add_argument('--semantic_enhancement', type=int, default=1,
                       help='1: enable semantic enhancement in classifier, 0: disable')

    # 🚨 NEW: Early stopping parameters
    parser.add_argument('--early_stopping_patience', type=int, default=15,
                       help='number of epochs to wait for improvement before early stopping')

    config = parser.parse_args()
    
  
    config.progressive_weights = [float(x) for x in config.progressive_weights.split(',')]
    
    config.use_enhanced_network = bool(config.use_enhanced_network)
    config.use_enhanced_training = bool(config.use_enhanced_training)
    config.use_progressive_supervision = bool(config.use_progressive_supervision)
    config.semantic_enhancement = bool(config.semantic_enhancement)

    print("\n" + "="*60)
    print(" Enhanced Semantic Communication System with MSA")
    print("="*60)
    print(f" Modulation: {config.mod_method.upper()}")
    print(f" Channel Use: {config.channel_use}")
    print(f" SNR Train/Test: {config.snr_train}/{config.snr_test} dB")
    print(f"Enhanced Network: {'ON' if config.use_enhanced_network else 'OFF'}")
    print(f" Progressive Supervision: {'ON' if config.use_progressive_supervision else 'OFF'}")
    print(f" MSA Attention Heads: {config.msa_num_heads}")
    print(f" Progressive Lambda: {config.progressive_lambda}")
    print(f" Load Checkpoint: {'ON' if config.load_checkpoint else 'OFF'}")
    print(f"🚨 Learning Rate: {config.lr} (Reduced for anti-overfitting)")
    print(f"🚨 Weight Decay: {config.weight_decay} (Added for regularization)")
    print(f"🚨 Early Stopping Patience: {config.early_stopping_patience}")
    print("="*60 + "\n")

    mischandler(config)
    main(config)

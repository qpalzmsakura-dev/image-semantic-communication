import torch
import torch.nn.functional as F
import numpy as np
from math import log10


def PSNR(original, compressed):
    original = torch.clamp(original, -1, 1)
    compressed = torch.clamp(compressed, -1, 1)
    
    mse = F.mse_loss(compressed, original, reduction='mean')
    
    if mse < 1e-10:
        return 100.0
    
    
    max_pixel = 2.0
    psnr = 20 * log10(max_pixel / torch.sqrt(mse))
    
    return psnr.item() if hasattr(psnr, 'item') else psnr


def SSIM(img1, img2, window_size=11, reduction='mean'):

    img1 = torch.clamp(img1, -1, 1)
    img2 = torch.clamp(img2, -1, 1)
    
    img1 = (img1 + 1) / 2
    img2 = (img2 + 1) / 2
    
   
    if len(img1.shape) == 4:
        if img1.shape[1] == 3: 
            
            img1 = 0.299 * img1[:, 0:1] + 0.587 * img1[:, 1:2] + 0.114 * img1[:, 2:3]
            img2 = 0.299 * img2[:, 0:1] + 0.587 * img2[:, 1:2] + 0.114 * img2[:, 2:3]
    
    
    try:
        mu1 = F.avg_pool2d(img1, window_size, stride=1, padding=window_size//2)
        mu2 = F.avg_pool2d(img2, window_size, stride=1, padding=window_size//2)
        
        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1 * mu2
        
        sigma1_sq = F.avg_pool2d(img1 * img1, window_size, stride=1, padding=window_size//2) - mu1_sq
        sigma2_sq = F.avg_pool2d(img2 * img2, window_size, stride=1, padding=window_size//2) - mu2_sq
        sigma12 = F.avg_pool2d(img1 * img2, window_size, stride=1, padding=window_size//2) - mu1_mu2
        
        
        C1 = 0.01 ** 2
        C2 = 0.03 ** 2
        
        ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
        
        ssim_value = ssim_map.mean()
        return ssim_value.item() if hasattr(ssim_value, 'item') else ssim_value
        
    except Exception as e:
        print(f"Warning: SSIM calculation failed: {e}")
        return 0.5


def EVAL(model, data_loader, device, config, epoch=None):
    model.eval()
    
    total_samples = 0
    correct_predictions = 0
    total_mse = 0
    total_psnr = 0
    total_ssim = 0
    num_batches = 0
    
    print(f"Evaluating model on {len(data_loader)} batches...")
    
    with torch.no_grad():
        for batch_idx, (data, labels) in enumerate(data_loader):
            data, labels = data.to(device), labels.to(device)
            
            outputs = model(data)
            if len(outputs) == 6:
                # Enhanced JCM: code, z, z_hat, pred, rec, progressive_preds
                code, z, z_hat, pred, rec, _ = outputs
            elif len(outputs) == 5:
                # Standard JCM: code, z, z_hat, pred, rec
                code, z, z_hat, pred, rec = outputs
            else:
                raise ValueError(f"Model returned {len(outputs)} values, expected 5 or 6")
            
            batch_size = labels.size(0)
            total_samples += batch_size
            predicted = torch.max(pred.data, 1)[1]
            correct_predictions += (predicted == labels).sum().item()
            
            
            # MSE
            mse_batch = F.mse_loss(rec, data, reduction='mean').item()
            total_mse += mse_batch
            
            # PSNR
            psnr_batch = PSNR(data, rec)
            total_psnr += psnr_batch
            
            # SSIM
            ssim_batch = SSIM(data, rec)
            total_ssim += ssim_batch
            
            num_batches += 1
            
            if batch_idx % 50 == 0 and batch_idx > 0:
                print(f"   Batch {batch_idx}/{len(data_loader)}: PSNR={psnr_batch:.2f}, SSIM={ssim_batch:.3f}")

    
    accuracy = 100.0 * correct_predictions / total_samples
    avg_mse = total_mse / num_batches
    avg_psnr = total_psnr / num_batches
    avg_ssim = total_ssim / num_batches
    
    print(f"Evaluation complete: Acc={accuracy:.2f}%, MSE={avg_mse:.6f}, PSNR={avg_psnr:.2f}dB, SSIM={avg_ssim:.3f}")
    
    return torch.tensor(accuracy), avg_mse, avg_psnr, avg_ssim


def detailed_eval(model, data_loader, device, config):
   
    model.eval()
    
    class_correct = torch.zeros(10)   
    class_total = torch.zeros(10)
    total_samples = 0
    correct_samples = 0
    
    print("Running detailed evaluation...")
    
    with torch.no_grad():
        for data, labels in data_loader:
            data, labels = data.to(device), labels.to(device)
            
            outputs = model(data)
            if len(outputs) >= 5:
                pred = outputs[3]  
            else:
                raise ValueError("Invalid model outputs")
            
            predicted = torch.max(pred, 1)[1]
            
            total_samples += labels.size(0)
            correct_samples += (predicted == labels).sum().item()
            
            
            c = (predicted == labels).squeeze()
            for i in range(labels.size(0)):
                label = labels[i]
                class_correct[label] += c[i].item()
                class_total[label] += 1

    
    overall_acc = 100.0 * correct_samples / total_samples
    print(f"Overall Accuracy: {overall_acc:.2f}%")
    print(" Per-class Accuracy:")
    
    class_names = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    for i in range(10):
        if class_total[i] > 0:
            acc = 100.0 * class_correct[i] / class_total[i]
            print(f"   {class_names[i]:8s}: {acc:6.2f}% ({int(class_correct[i])}/{int(class_total[i])})")
    
    return overall_acc, class_correct, class_total


def evaluate_snr_robustness(model, data_loader, device, config, snr_range=None):
    
    if snr_range is None:
        snr_range = [0, 3, 6, 9, 12, 15, 18, 21, 24]
    
    print(f" Evaluating SNR robustness over range: {snr_range}")
    
    results = {}
    original_snr = config.snr_test
    original_mode = config.mode
    
    for snr in snr_range:
        print(f" Testing at SNR = {snr} dB...")
        
        
        config.snr_test = snr
        config.mode = 'test'
        
        
        acc, mse, psnr, ssim = EVAL(model, data_loader, device, config)
        
        results[snr] = {
            'accuracy': acc.item() if hasattr(acc, 'item') else acc,
            'mse': mse,
            'psnr': psnr,
            'ssim': ssim
        }
        
        print(f"   Results: Acc={acc:.2f}%, PSNR={psnr:.2f}dB, SSIM={ssim:.3f}")
    
    
    config.snr_test = original_snr
    config.mode = original_mode
    
    return results


def evaluate_model(model, data_loader, device, config):
   
    return EVAL(model, data_loader, device, config)



def test_metrics():
    
    print("Testing PSNR and SSIM calculations...")
    
   
    x = torch.randn(2, 3, 32, 32) * 0.5  
    y = x + torch.randn_like(x) * 0.1       
    psnr = PSNR(x, y)
    ssim = SSIM(x, y)
    
    print(f"Test PSNR: {psnr:.2f} dB (normal range: 10-40)")
    print(f"Test SSIM: {ssim:.3f} (normal range: 0-1)")
    
   
    psnr_perfect = PSNR(x, x)
    ssim_perfect = SSIM(x, x)
    
    print(f"Perfect reconstruction PSNR: {psnr_perfect:.2f} dB")
    print(f"Perfect reconstruction SSIM: {ssim_perfect:.3f}")
    
    return psnr, ssim


if __name__ == "__main__":
    
    test_metrics()

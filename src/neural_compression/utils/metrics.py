import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Tuple, Optional
import math


class CompressionMetrics:
    """압축 모델 평가 메트릭 모음"""
    
    @staticmethod
    def psnr(pred: torch.Tensor, target: torch.Tensor, max_val: float = 1.0) -> torch.Tensor:
        """PSNR 계산"""
        mse = F.mse_loss(pred, target)
        if mse == 0:
            return torch.tensor(float('inf'))
        return 20 * torch.log10(max_val / torch.sqrt(mse))
    
    @staticmethod
    def ssim(pred: torch.Tensor, target: torch.Tensor, max_val: float = 1.0) -> torch.Tensor:
        """SSIM 계산"""
        C1 = (0.01 * max_val) ** 2
        C2 = (0.03 * max_val) ** 2
        
        mu_pred = torch.mean(pred, dim=[2, 3], keepdim=True)
        mu_target = torch.mean(target, dim=[2, 3], keepdim=True)
        
        sigma_pred = torch.var(pred, dim=[2, 3], keepdim=True)
        sigma_target = torch.var(target, dim=[2, 3], keepdim=True)
        sigma_cross = torch.mean((pred - mu_pred) * (target - mu_target), dim=[2, 3], keepdim=True)
        
        ssim_val = ((2 * mu_pred * mu_target + C1) * (2 * sigma_cross + C2)) / \
                   ((mu_pred ** 2 + mu_target ** 2 + C1) * (sigma_pred + sigma_target + C2))
        
        return torch.mean(ssim_val)
    
    @staticmethod
    def compression_ratio(original_size: int, compressed_size: int) -> float:
        """압축률 계산"""
        return original_size / compressed_size
    
    @staticmethod
    def bits_per_pixel(compressed_bits: int, height: int, width: int) -> float:
        """픽셀당 비트 수"""
        return compressed_bits / (height * width)


class PSNRMetric(nn.Module):
    """PSNR 메트릭 모듈"""
    
    def __init__(self, max_val: float = 1.0):
        super().__init__()
        self.max_val = max_val
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return CompressionMetrics.psnr(pred, target, self.max_val)


class SSIMMetric(nn.Module):
    """SSIM 메트릭 모듈"""
    
    def __init__(self, max_val: float = 1.0):
        super().__init__()
        self.max_val = max_val
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return CompressionMetrics.ssim(pred, target, self.max_val)


class LPIPSMetric(nn.Module):
    """LPIPS 메트릭 (간단 버전)"""
    
    def __init__(self):
        super().__init__()
        
        # VGG19 특징 추출기 사용
        import torchvision.models as models
        vgg = models.vgg19(pretrained=True).features[:16]
        
        self.feature_extractor = vgg
        for param in self.feature_extractor.parameters():
            param.requires_grad = False
        
        self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer('std', torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # Normalize
        pred_norm = (pred - self.mean) / self.std
        target_norm = (target - self.mean) / self.std
        
        # Extract features
        pred_features = self.feature_extractor(pred_norm)
        target_features = self.feature_extractor(target_norm)
        
        # L2 distance in feature space
        return F.mse_loss(pred_features, target_features)


class CompressionBenchmark:
    """압축 벤치마크 클래스"""
    
    def __init__(self, model: nn.Module, device: str = 'cuda'):
        self.model = model.to(device)
        self.device = device
        self.model.eval()
    
    def evaluate_dataset(self, dataloader) -> Dict[str, float]:
        """데이터셋 전체 평가"""
        total_psnr = 0.0
        total_ssim = 0.0
        total_compression_ratio = 0.0
        total_samples = 0
        
        psnr_metric = PSNRMetric().to(self.device)
        ssim_metric = SSIMMetric().to(self.device)
        
        with torch.no_grad():
            for batch in dataloader:
                images = batch['image'].to(self.device)
                batch_size = images.shape[0]
                
                # Forward pass
                results = self.model(images)
                reconstructed = results['reconstructed']
                
                # 메트릭 계산
                batch_psnr = psnr_metric(reconstructed, images)
                batch_ssim = ssim_metric(reconstructed, images)
                
                # 압축률 계산
                original_bits = images.numel() * 32
                compressed_bits = results['latent'].numel() * 8
                batch_compression_ratio = original_bits / compressed_bits
                
                total_psnr += batch_psnr.item() * batch_size
                total_ssim += batch_ssim.item() * batch_size
                total_compression_ratio += batch_compression_ratio * batch_size
                total_samples += batch_size
        
        return {
            'psnr': total_psnr / total_samples,
            'ssim': total_ssim / total_samples,
            'compression_ratio': total_compression_ratio / total_samples,
            'samples': total_samples
        }
    
    def compare_with_baselines(self, dataloader, baselines: Dict[str, nn.Module]) -> Dict[str, Dict[str, float]]:
        """베이스라인과 비교"""
        results = {}
        
        # Our model
        results['neural_compression'] = self.evaluate_dataset(dataloader)
        
        # Baseline models
        for name, baseline_model in baselines.items():
            benchmark = CompressionBenchmark(baseline_model, self.device)
            results[name] = benchmark.evaluate_dataset(dataloader)
        
        return results
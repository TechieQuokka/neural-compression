import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from typing import List, Optional
import numpy as np


class MSELoss(nn.Module):
    """Mean Squared Error Loss"""
    
    def __init__(self, reduction: str = 'mean'):
        super().__init__()
        self.reduction = reduction
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return F.mse_loss(pred, target, reduction=self.reduction)


class L1Loss(nn.Module):
    """L1 (Mean Absolute Error) Loss"""
    
    def __init__(self, reduction: str = 'mean'):
        super().__init__()
        self.reduction = reduction
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return F.l1_loss(pred, target, reduction=self.reduction)


class PerceptualLoss(nn.Module):
    """VGG 기반 지각적 손실"""
    
    def __init__(
        self,
        feature_layers: List[int] = [3, 8, 15, 22],
        weights: Optional[List[float]] = None,
        normalize: bool = True
    ):
        super().__init__()
        
        # Load pre-trained VGG19
        vgg = models.vgg19(pretrained=True).features
        self.feature_extractor = nn.ModuleList()
        
        # Extract specific layers
        layer_idx = 0
        for i, layer in enumerate(vgg):
            self.feature_extractor.append(layer)
            if i in feature_layers:
                break
            layer_idx += 1
        
        # Freeze parameters
        for param in self.feature_extractor.parameters():
            param.requires_grad = False
        
        self.feature_layers = feature_layers
        self.weights = weights or [1.0] * len(feature_layers)
        self.normalize = normalize
        
        if normalize:
            self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
            self.register_buffer('std', torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))
    
    def extract_features(self, x: torch.Tensor) -> List[torch.Tensor]:
        """특징 추출"""
        if self.normalize:
            x = (x - self.mean) / self.std
        
        features = []
        layer_idx = 0
        
        for i, layer in enumerate(self.feature_extractor):
            x = layer(x)
            if i in self.feature_layers:
                features.append(x)
        
        return features
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred_features = self.extract_features(pred)
        target_features = self.extract_features(target)
        
        loss = 0.0
        for i, (pred_feat, target_feat, weight) in enumerate(zip(pred_features, target_features, self.weights)):
            loss += weight * F.mse_loss(pred_feat, target_feat)
        
        return loss


class MSSSIMLoss(nn.Module):
    """Multi-Scale Structural Similarity Loss"""
    
    def __init__(
        self,
        weights: List[float] = [0.0448, 0.2856, 0.3001, 0.2363, 0.1333],
        max_val: float = 1.0,
        filter_size: int = 11,
        sigma: float = 1.5
    ):
        super().__init__()
        
        self.weights = weights
        self.max_val = max_val
        self.levels = len(weights)
        
        # Create Gaussian filter
        self.register_buffer('gaussian_filter', self._create_gaussian_filter(filter_size, sigma))
    
    def _create_gaussian_filter(self, filter_size: int, sigma: float) -> torch.Tensor:
        """가우시안 필터 생성"""
        coords = torch.arange(filter_size, dtype=torch.float32)
        coords -= filter_size // 2
        
        g = torch.exp(-(coords**2) / (2 * sigma**2))
        g /= g.sum()
        
        return g.view(1, 1, -1) * g.view(1, -1, 1)
    
    def _gaussian_filter2d(self, x: torch.Tensor, filter_2d: torch.Tensor) -> torch.Tensor:
        """2D 가우시안 필터 적용"""
        B, C, H, W = x.shape
        filter_2d = filter_2d.expand(C, 1, -1, -1)
        
        return F.conv2d(x, filter_2d, padding=filter_2d.shape[-1]//2, groups=C)
    
    def _ssim(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """단일 스케일 SSIM 계산"""
        C1 = (0.01 * self.max_val) ** 2
        C2 = (0.03 * self.max_val) ** 2
        
        mu_x = self._gaussian_filter2d(x, self.gaussian_filter)
        mu_y = self._gaussian_filter2d(y, self.gaussian_filter)
        
        mu_x_sq = mu_x ** 2
        mu_y_sq = mu_y ** 2
        mu_xy = mu_x * mu_y
        
        sigma_x_sq = self._gaussian_filter2d(x ** 2, self.gaussian_filter) - mu_x_sq
        sigma_y_sq = self._gaussian_filter2d(y ** 2, self.gaussian_filter) - mu_y_sq
        sigma_xy = self._gaussian_filter2d(x * y, self.gaussian_filter) - mu_xy
        
        ssim_map = ((2 * mu_xy + C1) * (2 * sigma_xy + C2)) / \
                   ((mu_x_sq + mu_y_sq + C1) * (sigma_x_sq + sigma_y_sq + C2))
        
        return torch.mean(ssim_map, dim=[2, 3])
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        ms_ssim = []
        
        current_pred = pred
        current_target = target
        
        for i in range(self.levels):
            ssim_val = self._ssim(current_pred, current_target)
            ms_ssim.append(ssim_val)
            
            if i < self.levels - 1:
                current_pred = F.avg_pool2d(current_pred, 2)
                current_target = F.avg_pool2d(current_target, 2)
        
        # Weighted combination
        total_loss = 0.0
        for i, (ssim_val, weight) in enumerate(zip(ms_ssim, self.weights)):
            if i == self.levels - 1:
                total_loss += weight * ssim_val
            else:
                total_loss += weight * (ssim_val ** self.weights[i])
        
        return 1.0 - torch.mean(total_loss)


class LPIPSLoss(nn.Module):
    """Learned Perceptual Image Patch Similarity"""
    
    def __init__(self, net: str = 'alex', use_gpu: bool = True):
        super().__init__()
        
        # Import LPIPS if available, otherwise use simplified version
        try:
            import lpips
            self.lpips_model = lpips.LPIPS(net=net)
            self.use_lpips = True
        except ImportError:
            print("LPIPS not available, using VGG-based perceptual loss")
            self.lpips_model = PerceptualLoss()
            self.use_lpips = False
        
        if use_gpu and torch.cuda.is_available():
            self.lpips_model = self.lpips_model.cuda()
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if self.use_lpips:
            # Normalize to [-1, 1] range
            pred = 2 * pred - 1
            target = 2 * target - 1
            return self.lpips_model(pred, target).mean()
        else:
            return self.lpips_model(pred, target)


class CharbonnierLoss(nn.Module):
    """Charbonnier Loss (smooth L1)"""
    
    def __init__(self, epsilon: float = 1e-3):
        super().__init__()
        self.epsilon = epsilon
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        diff = pred - target
        loss = torch.sqrt(diff ** 2 + self.epsilon ** 2)
        return torch.mean(loss)
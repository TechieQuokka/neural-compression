import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Optional


class GANLoss(nn.Module):
    """Standard GAN Loss"""
    
    def __init__(self, real_label: float = 1.0, fake_label: float = 0.0):
        super().__init__()
        self.real_label = real_label
        self.fake_label = fake_label
        self.loss = nn.BCEWithLogitsLoss()
    
    def __call__(
        self, 
        discriminator_output: torch.Tensor, 
        target_is_real: bool
    ) -> torch.Tensor:
        target_tensor = torch.full_like(
            discriminator_output,
            self.real_label if target_is_real else self.fake_label
        )
        return self.loss(discriminator_output, target_tensor)


class LSGANLoss(nn.Module):
    """Least Squares GAN Loss"""
    
    def __init__(self, real_label: float = 1.0, fake_label: float = 0.0):
        super().__init__()
        self.real_label = real_label
        self.fake_label = fake_label
        self.loss = nn.MSELoss()
    
    def __call__(
        self, 
        discriminator_output: torch.Tensor, 
        target_is_real: bool
    ) -> torch.Tensor:
        target_tensor = torch.full_like(
            discriminator_output,
            self.real_label if target_is_real else self.fake_label
        )
        return self.loss(discriminator_output, target_tensor)


class WGANLoss(nn.Module):
    """Wasserstein GAN Loss with Gradient Penalty"""
    
    def __init__(self, lambda_gp: float = 10.0):
        super().__init__()
        self.lambda_gp = lambda_gp
    
    def discriminator_loss(
        self,
        real_output: torch.Tensor,
        fake_output: torch.Tensor
    ) -> torch.Tensor:
        """판별기 손실"""
        return torch.mean(fake_output) - torch.mean(real_output)
    
    def generator_loss(self, fake_output: torch.Tensor) -> torch.Tensor:
        """생성기 손실"""
        return -torch.mean(fake_output)
    
    def gradient_penalty(
        self,
        discriminator: nn.Module,
        real_data: torch.Tensor,
        fake_data: torch.Tensor
    ) -> torch.Tensor:
        """Gradient penalty 계산"""
        B = real_data.shape[0]
        
        # Random interpolation
        alpha = torch.rand(B, 1, 1, 1, device=real_data.device)
        interpolated = alpha * real_data + (1 - alpha) * fake_data
        interpolated.requires_grad_(True)
        
        # Discriminator output for interpolated samples
        d_interpolated = discriminator(interpolated)
        
        # Gradients
        gradients = torch.autograd.grad(
            outputs=d_interpolated,
            inputs=interpolated,
            grad_outputs=torch.ones_like(d_interpolated),
            create_graph=True,
            retain_graph=True,
            only_inputs=True
        )[0]
        
        # Gradient penalty
        gradients = gradients.view(B, -1)
        gradient_norm = torch.sqrt(torch.sum(gradients ** 2, dim=1) + 1e-12)
        penalty = torch.mean((gradient_norm - 1) ** 2)
        
        return self.lambda_gp * penalty


class HingeLoss(nn.Module):
    """Hinge Loss for GAN"""
    
    def discriminator_loss(
        self,
        real_output: torch.Tensor,
        fake_output: torch.Tensor
    ) -> torch.Tensor:
        """판별기 손실"""
        real_loss = torch.mean(F.relu(1.0 - real_output))
        fake_loss = torch.mean(F.relu(1.0 + fake_output))
        return real_loss + fake_loss
    
    def generator_loss(self, fake_output: torch.Tensor) -> torch.Tensor:
        """생성기 손실"""
        return -torch.mean(fake_output)


class FeatureMatchingLoss(nn.Module):
    """Feature Matching Loss"""
    
    def __init__(self, weights: Optional[List[float]] = None):
        super().__init__()
        self.weights = weights
    
    def forward(
        self,
        real_features: List[torch.Tensor],
        fake_features: List[torch.Tensor]
    ) -> torch.Tensor:
        """특징 매칭 손실 계산"""
        loss = 0.0
        weights = self.weights or [1.0] * len(real_features)
        
        for i, (real_feat, fake_feat, weight) in enumerate(zip(real_features, fake_features, weights)):
            loss += weight * F.l1_loss(fake_feat, real_feat.detach())
        
        return loss


class SpectralLoss(nn.Module):
    """Spectral Loss (주파수 도메인)"""
    
    def __init__(self, loss_type: str = 'l1'):
        super().__init__()
        self.loss_type = loss_type
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # FFT 변환
        pred_fft = torch.fft.fft2(pred, norm='ortho')
        target_fft = torch.fft.fft2(target, norm='ortho')
        
        # 크기 스펙트럼
        pred_magnitude = torch.abs(pred_fft)
        target_magnitude = torch.abs(target_fft)
        
        # 위상 스펙트럼
        pred_phase = torch.angle(pred_fft)
        target_phase = torch.angle(target_fft)
        
        if self.loss_type == 'l1':
            magnitude_loss = F.l1_loss(pred_magnitude, target_magnitude)
            phase_loss = F.l1_loss(pred_phase, target_phase)
        else:  # l2
            magnitude_loss = F.mse_loss(pred_magnitude, target_magnitude)
            phase_loss = F.mse_loss(pred_phase, target_phase)
        
        return magnitude_loss + 0.1 * phase_loss


class AdversarialLoss(nn.Module):
    """통합 적대적 손실"""
    
    def __init__(
        self,
        loss_type: str = 'lsgan',  # 'vanilla', 'lsgan', 'wgan', 'hinge'
        lambda_gp: float = 10.0,
        use_feature_matching: bool = True,
        fm_weight: float = 10.0
    ):
        super().__init__()
        
        self.loss_type = loss_type
        self.use_feature_matching = use_feature_matching
        self.fm_weight = fm_weight
        
        if loss_type == 'vanilla':
            self.criterion = GANLoss()
        elif loss_type == 'lsgan':
            self.criterion = LSGANLoss()
        elif loss_type == 'wgan':
            self.criterion = WGANLoss(lambda_gp)
        elif loss_type == 'hinge':
            self.criterion = HingeLoss()
        else:
            raise ValueError(f"Unknown loss type: {loss_type}")
        
        if use_feature_matching:
            self.feature_matching = FeatureMatchingLoss()
    
    def discriminator_loss(
        self,
        real_output: torch.Tensor,
        fake_output: torch.Tensor,
        discriminator: Optional[nn.Module] = None,
        real_data: Optional[torch.Tensor] = None,
        fake_data: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """판별기 손실"""
        if self.loss_type in ['vanilla', 'lsgan']:
            real_loss = self.criterion(real_output, True)
            fake_loss = self.criterion(fake_output, False)
            total_loss = (real_loss + fake_loss) * 0.5
        elif self.loss_type == 'wgan':
            total_loss = self.criterion.discriminator_loss(real_output, fake_output)
            if discriminator is not None and real_data is not None and fake_data is not None:
                gp = self.criterion.gradient_penalty(discriminator, real_data, fake_data)
                total_loss += gp
        elif self.loss_type == 'hinge':
            total_loss = self.criterion.discriminator_loss(real_output, fake_output)
        
        return total_loss
    
    def generator_loss(
        self,
        fake_output: torch.Tensor,
        real_features: Optional[List[torch.Tensor]] = None,
        fake_features: Optional[List[torch.Tensor]] = None
    ) -> torch.Tensor:
        """생성기 손실"""
        if self.loss_type in ['vanilla', 'lsgan']:
            adv_loss = self.criterion(fake_output, True)
        elif self.loss_type in ['wgan', 'hinge']:
            adv_loss = self.criterion.generator_loss(fake_output)
        
        total_loss = adv_loss
        
        # Feature matching loss
        if self.use_feature_matching and real_features is not None and fake_features is not None:
            fm_loss = self.feature_matching(real_features, fake_features)
            total_loss += self.fm_weight * fm_loss
        
        return total_loss
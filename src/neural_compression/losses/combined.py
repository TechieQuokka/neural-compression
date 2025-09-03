import torch
import torch.nn as nn
from typing import Dict, Optional, List
from .reconstruction import MSELoss, PerceptualLoss, MSSSIMLoss, LPIPSLoss
from .adversarial import AdversarialLoss, FeatureMatchingLoss
from .compression import RateDistortionLoss, EntropyLoss


class CombinedLoss(nn.Module):
    """통합 손실 함수"""
    
    def __init__(
        self,
        loss_weights: Dict[str, float] = None,
        use_adversarial: bool = True,
        use_perceptual: bool = True,
        use_rate_distortion: bool = True,
        adversarial_type: str = 'lsgan'
    ):
        super().__init__()
        
        # 기본 가중치 설정
        default_weights = {
            'reconstruction': 1.0,
            'perceptual': 0.1,
            'adversarial': 0.1,
            'rate_distortion': 0.05,
            'kl_divergence': 0.01,
            'ms_ssim': 0.05,
            'lpips': 0.05
        }
        
        self.loss_weights = {**default_weights, **(loss_weights or {})}
        self.use_adversarial = use_adversarial
        self.use_perceptual = use_perceptual
        self.use_rate_distortion = use_rate_distortion
        
        # 손실 함수들 초기화
        self.mse_loss = MSELoss()
        
        if use_perceptual:
            self.perceptual_loss = PerceptualLoss()
            self.ms_ssim_loss = MSSSIMLoss()
            self.lpips_loss = LPIPSLoss()
        
        if use_adversarial:
            self.adversarial_loss = AdversarialLoss(
                loss_type=adversarial_type,
                use_feature_matching=True
            )
        
        if use_rate_distortion:
            self.rate_distortion_loss = RateDistortionLoss()
    
    def compute_vae_loss(
        self,
        mu: torch.Tensor,
        logvar: torch.Tensor
    ) -> torch.Tensor:
        """VAE KL divergence 손실"""
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        kl_loss = kl_loss / (mu.shape[0] * mu.shape[1])  # Normalize
        return kl_loss
    
    def generator_loss(
        self,
        reconstructed: torch.Tensor,
        original: torch.Tensor,
        mu: torch.Tensor,
        logvar: torch.Tensor,
        fake_output: Optional[torch.Tensor] = None,
        real_features: Optional[List[torch.Tensor]] = None,
        fake_features: Optional[List[torch.Tensor]] = None,
        latent: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """생성기 손실 계산"""
        
        losses = {}
        total_loss = 0.0
        
        # 기본 재구성 손실
        recon_loss = self.mse_loss(reconstructed, original)
        losses['reconstruction'] = recon_loss
        total_loss += self.loss_weights['reconstruction'] * recon_loss
        
        # VAE KL divergence
        kl_loss = self.compute_vae_loss(mu, logvar)
        losses['kl_divergence'] = kl_loss
        total_loss += self.loss_weights['kl_divergence'] * kl_loss
        
        # 지각적 손실들
        if self.use_perceptual:
            perceptual_loss = self.perceptual_loss(reconstructed, original)
            losses['perceptual'] = perceptual_loss
            total_loss += self.loss_weights['perceptual'] * perceptual_loss
            
            ms_ssim_loss = self.ms_ssim_loss(reconstructed, original)
            losses['ms_ssim'] = ms_ssim_loss
            total_loss += self.loss_weights['ms_ssim'] * ms_ssim_loss
            
            lpips_loss = self.lpips_loss(reconstructed, original)
            losses['lpips'] = lpips_loss
            total_loss += self.loss_weights['lpips'] * lpips_loss
        
        # 적대적 손실
        if self.use_adversarial and fake_output is not None:
            adv_loss = self.adversarial_loss.generator_loss(
                fake_output, real_features, fake_features
            )
            losses['adversarial'] = adv_loss
            total_loss += self.loss_weights['adversarial'] * adv_loss
        
        # Rate-Distortion 손실
        if self.use_rate_distortion and latent is not None:
            rd_loss, distortion, rate = self.rate_distortion_loss(
                reconstructed, original, latent
            )
            losses['rate_distortion'] = rd_loss
            losses['distortion'] = distortion
            losses['rate'] = rate
            total_loss += self.loss_weights['rate_distortion'] * rd_loss
        
        losses['total'] = total_loss
        return losses
    
    def discriminator_loss(
        self,
        real_output: torch.Tensor,
        fake_output: torch.Tensor,
        discriminator: Optional[nn.Module] = None,
        real_data: Optional[torch.Tensor] = None,
        fake_data: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """판별기 손실 계산"""
        
        if not self.use_adversarial:
            return {'discriminator': torch.tensor(0.0)}
        
        d_loss = self.adversarial_loss.discriminator_loss(
            real_output, fake_output, discriminator, real_data, fake_data
        )
        
        return {'discriminator': d_loss}


class ProgressiveLoss(nn.Module):
    """점진적 훈련을 위한 손실"""
    
    def __init__(
        self,
        base_loss: nn.Module,
        resolution_schedule: List[int] = [64, 128, 256],
        loss_schedule: List[float] = [1.0, 0.5, 0.1]
    ):
        super().__init__()
        
        self.base_loss = base_loss
        self.resolution_schedule = resolution_schedule
        self.loss_schedule = loss_schedule
        self.current_resolution = resolution_schedule[0]
        self.current_loss_weight = loss_schedule[0]
    
    def update_resolution(self, new_resolution: int):
        """해상도 업데이트"""
        if new_resolution in self.resolution_schedule:
            idx = self.resolution_schedule.index(new_resolution)
            self.current_resolution = new_resolution
            self.current_loss_weight = self.loss_schedule[idx]
    
    def forward(self, *args, **kwargs) -> Dict[str, torch.Tensor]:
        losses = self.base_loss(*args, **kwargs)
        
        # 해상도에 따른 손실 가중치 조정
        for key in losses:
            if key != 'total':
                losses[key] *= self.current_loss_weight
        
        # 총 손실 재계산
        total_loss = sum(loss for key, loss in losses.items() if key != 'total')
        losses['total'] = total_loss
        
        return losses


class CurriculumLoss(nn.Module):
    """커리큘럼 학습을 위한 손실"""
    
    def __init__(
        self,
        base_loss: nn.Module,
        difficulty_schedule: List[float] = [0.1, 0.3, 0.5, 0.7, 1.0],
        epochs_per_stage: int = 20
    ):
        super().__init__()
        
        self.base_loss = base_loss
        self.difficulty_schedule = difficulty_schedule
        self.epochs_per_stage = epochs_per_stage
        self.current_epoch = 0
        self.current_difficulty = difficulty_schedule[0]
    
    def update_epoch(self, epoch: int):
        """에폭 업데이트"""
        self.current_epoch = epoch
        stage_idx = min(epoch // self.epochs_per_stage, len(self.difficulty_schedule) - 1)
        self.current_difficulty = self.difficulty_schedule[stage_idx]
    
    def forward(self, *args, **kwargs) -> Dict[str, torch.Tensor]:
        losses = self.base_loss(*args, **kwargs)
        
        # 난이도에 따른 손실 조정
        difficulty_weight = self.current_difficulty
        
        for key in losses:
            if key in ['adversarial', 'perceptual']:
                losses[key] *= difficulty_weight
        
        # 총 손실 재계산
        total_loss = sum(loss for key, loss in losses.items() if key != 'total')
        losses['total'] = total_loss
        
        return losses
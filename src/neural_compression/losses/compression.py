import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Tuple, List


class RateDistortionLoss(nn.Module):
    """Rate-Distortion 손실 함수"""
    
    def __init__(
        self,
        lambda_rate: float = 0.01,
        distortion_metric: str = 'mse',  # 'mse', 'l1', 'perceptual'
        rate_model: str = 'gaussian'  # 'gaussian', 'laplacian', 'learned'
    ):
        super().__init__()
        
        self.lambda_rate = lambda_rate
        self.distortion_metric = distortion_metric
        self.rate_model = rate_model
        
        # Distortion loss
        if distortion_metric == 'mse':
            self.distortion_loss = nn.MSELoss()
        elif distortion_metric == 'l1':
            self.distortion_loss = nn.L1Loss()
        
        # Rate estimation parameters
        if rate_model == 'gaussian':
            self.register_buffer('log_2pi', torch.log(torch.tensor(2 * np.pi)))
        
    def estimate_rate_gaussian(self, z: torch.Tensor) -> torch.Tensor:
        """가우시안 분포 가정하에 비트율 추정"""
        # 평균과 분산 계산 (latent이 2D인 경우)
        if z.dim() == 2:
            mean = torch.mean(z, dim=0, keepdim=True)
            var = torch.var(z, dim=0, keepdim=True) + 1e-8
        else:
            mean = torch.mean(z, dim=[0, 2, 3], keepdim=True)
            var = torch.var(z, dim=[0, 2, 3], keepdim=True) + 1e-8
        
        # 엔트로피 계산 (bits per pixel)
        entropy = 0.5 * (self.log_2pi + torch.log(var)) / np.log(2)
        
        return torch.mean(entropy)
    
    def estimate_rate_laplacian(self, z: torch.Tensor) -> torch.Tensor:
        """라플라시안 분포 가정하에 비트율 추정"""
        # Scale parameter 추정
        scale = torch.mean(torch.abs(z - torch.median(z)), dim=[0, 2, 3], keepdim=True) + 1e-8
        
        # 엔트로피 계산
        entropy = (1 + torch.log(2 * scale)) / np.log(2)
        
        return torch.mean(entropy)
    
    def forward(
        self,
        reconstructed: torch.Tensor,
        original: torch.Tensor,
        latent: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Rate-Distortion 손실 계산"""
        
        # Distortion (복원 품질)
        distortion = self.distortion_loss(reconstructed, original)
        
        # Rate (비트율)
        if self.rate_model == 'gaussian':
            rate = self.estimate_rate_gaussian(latent)
        elif self.rate_model == 'laplacian':
            rate = self.estimate_rate_laplacian(latent)
        else:
            rate = torch.tensor(0.0, device=latent.device)
        
        # 총 손실
        total_loss = distortion + self.lambda_rate * rate
        
        return total_loss, distortion, rate


class EntropyLoss(nn.Module):
    """엔트로피 기반 압축 손실"""
    
    def __init__(self, temperature: float = 1.0):
        super().__init__()
        self.temperature = temperature
    
    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        """소프트맥스 엔트로피 계산"""
        probs = F.softmax(logits / self.temperature, dim=-1)
        entropy = -torch.sum(probs * torch.log(probs + 1e-8), dim=-1)
        return torch.mean(entropy)


class VQVAELoss(nn.Module):
    """VQ-VAE 전용 손실 함수"""
    
    def __init__(
        self,
        commitment_cost: float = 0.25,
        reconstruction_weight: float = 1.0
    ):
        super().__init__()
        self.commitment_cost = commitment_cost
        self.reconstruction_weight = reconstruction_weight
        self.reconstruction_loss = nn.MSELoss()
    
    def forward(
        self,
        reconstructed: torch.Tensor,
        original: torch.Tensor,
        quantized: torch.Tensor,
        encoded: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """VQ-VAE 손실 계산"""
        
        # Reconstruction loss
        recon_loss = self.reconstruction_loss(reconstructed, original)
        
        # Vector quantization losses
        vq_loss = F.mse_loss(quantized.detach(), encoded)
        commitment_loss = F.mse_loss(quantized, encoded.detach())
        
        total_loss = (self.reconstruction_weight * recon_loss + 
                     vq_loss + self.commitment_cost * commitment_loss)
        
        return total_loss, recon_loss, vq_loss


class AdaptiveLambdaRD(nn.Module):
    """적응적 Rate-Distortion 람다 조정"""
    
    def __init__(
        self,
        initial_lambda: float = 0.01,
        target_rate: Optional[float] = None,
        adaptation_rate: float = 0.01
    ):
        super().__init__()
        
        self.lambda_rd = nn.Parameter(torch.tensor(initial_lambda))
        self.target_rate = target_rate
        self.adaptation_rate = adaptation_rate
        
    def forward(
        self,
        distortion: torch.Tensor,
        rate: torch.Tensor,
        current_rate: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """적응적 RD 손실"""
        
        # 기본 RD 손실
        rd_loss = distortion + torch.clamp(self.lambda_rd, min=0.001, max=1.0) * rate
        
        # 목표 비트율이 설정된 경우 람다 조정
        if self.target_rate is not None and current_rate is not None:
            rate_error = current_rate - self.target_rate
            lambda_adjustment = self.adaptation_rate * rate_error
            
            # 람다 업데이트 (gradient-free)
            with torch.no_grad():
                self.lambda_rd.data = torch.clamp(
                    self.lambda_rd.data + lambda_adjustment,
                    min=0.001,
                    max=1.0
                )
        
        return rd_loss


class CompandingLoss(nn.Module):
    """Companding (압신-신장) 손실"""
    
    def __init__(self, mu: float = 255.0):
        super().__init__()
        self.mu = mu
    
    def mu_law_encode(self, x: torch.Tensor) -> torch.Tensor:
        """μ-law 압신"""
        x_norm = torch.clamp(x, -1, 1)
        return torch.sign(x_norm) * torch.log(1 + self.mu * torch.abs(x_norm)) / torch.log(1 + self.mu)
    
    def mu_law_decode(self, x: torch.Tensor) -> torch.Tensor:
        """μ-law 신장"""
        return torch.sign(x) * (torch.exp(torch.abs(x) * torch.log(1 + self.mu)) - 1) / self.mu
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # μ-law 도메인에서 손실 계산
        pred_companded = self.mu_law_encode(pred)
        target_companded = self.mu_law_encode(target)
        
        return F.mse_loss(pred_companded, target_companded)


class MultiScaleLoss(nn.Module):
    """다중 스케일 손실"""
    
    def __init__(
        self,
        scales: List[int] = [1, 2, 4],
        weights: Optional[List[float]] = None,
        loss_fn: nn.Module = nn.MSELoss()
    ):
        super().__init__()
        
        self.scales = scales
        self.weights = weights or [1.0] * len(scales)
        self.loss_fn = loss_fn
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        total_loss = 0.0
        
        for scale, weight in zip(self.scales, self.weights):
            if scale == 1:
                scaled_pred = pred
                scaled_target = target
            else:
                scaled_pred = F.avg_pool2d(pred, scale)
                scaled_target = F.avg_pool2d(target, scale)
            
            loss = self.loss_fn(scaled_pred, scaled_target)
            total_loss += weight * loss
        
        return total_loss


class FrequencyLoss(nn.Module):
    """주파수 도메인 손실"""
    
    def __init__(self, alpha: float = 1.0, beta: float = 0.1):
        super().__init__()
        self.alpha = alpha  # 저주파 가중치
        self.beta = beta    # 고주파 가중치
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # DCT 변환
        pred_dct = torch.fft.dct(torch.fft.dct(pred, dim=-1, norm='ortho'), dim=-2, norm='ortho')
        target_dct = torch.fft.dct(torch.fft.dct(target, dim=-1, norm='ortho'), dim=-2, norm='ortho')
        
        # 주파수 마스크 생성
        H, W = pred_dct.shape[-2:]
        freq_mask = torch.zeros_like(pred_dct)
        
        # 저주파 영역 (좌상단)
        low_freq_h, low_freq_w = H // 4, W // 4
        freq_mask[..., :low_freq_h, :low_freq_w] = self.alpha
        
        # 고주파 영역 (나머지)
        freq_mask[..., low_freq_h:, :] = self.beta
        freq_mask[..., :, low_freq_w:] = self.beta
        
        # 가중 손실
        weighted_loss = freq_mask * (pred_dct - target_dct) ** 2
        
        return torch.mean(weighted_loss)
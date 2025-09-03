import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional, List, Any
import numpy as np

from .encoder import CompressionEncoder, EfficientEncoder
from .decoder import CompressionDecoder, ProgressiveDecoder
from .discriminator import PatchGANDiscriminator, MultiScaleDiscriminator
from .quantization import VectorQuantizer, LearnedQuantizer, UniformQuantizer


class VAEGANCompression(nn.Module):
    """VAE-GAN 하이브리드 압축 모델"""
    
    def __init__(
        self,
        in_channels: int = 3,
        latent_dim: int = 256,
        encoder_type: str = "standard",  # "standard", "efficient"
        decoder_type: str = "standard",  # "standard", "progressive"
        use_discriminator: bool = True,
        discriminator_type: str = "patchgan",  # "patchgan", "multiscale"
        quantization_method: str = "vector",  # "vector", "learned", "uniform"
        compression_ratio: int = 16,
        **kwargs
    ):
        super().__init__()
        
        self.latent_dim = latent_dim
        self.compression_ratio = compression_ratio
        self.use_discriminator = use_discriminator
        self.quantization_method = quantization_method
        
        # Encoder
        if encoder_type == "efficient":
            self.encoder = EfficientEncoder(
                in_channels=in_channels,
                latent_dim=latent_dim,
                use_attention=True
            )
        else:
            self.encoder = CompressionEncoder(
                in_channels=in_channels,
                latent_dim=latent_dim,
                use_attention=True,
                dropout=0.1
            )
        
        # Quantizer
        if quantization_method == "vector":
            self.quantizer = VectorQuantizer(
                num_embeddings=1024,
                embedding_dim=latent_dim,
                commitment_cost=0.25
            )
        elif quantization_method == "learned":
            self.quantizer = LearnedQuantizer(
                channels=latent_dim,
                num_levels=256
            )
        elif quantization_method == "uniform":
            self.quantizer = UniformQuantizer(num_bits=8)
        else:
            self.quantizer = None
        
        # Decoder
        target_size = kwargs.get('target_size', 256)  # 목표 출력 크기
        if decoder_type == "progressive":
            self.decoder = ProgressiveDecoder(
                latent_dim=latent_dim,
                out_channels=in_channels,
                max_resolution=target_size
            )
        else:
            self.decoder = CompressionDecoder(
                latent_dim=latent_dim,
                out_channels=in_channels,
                use_attention=True,
                dropout=0.1,
                target_size=target_size
            )
        
        # Discriminator
        if use_discriminator:
            if discriminator_type == "multiscale":
                self.discriminator = MultiScaleDiscriminator(
                    in_channels=in_channels,
                    ndf=64,
                    n_layers=3
                )
            else:
                self.discriminator = PatchGANDiscriminator(
                    in_channels=in_channels,
                    ndf=64,
                    n_layers=3,
                    use_spectral_norm=True
                )
    
    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """VAE 재매개화 트릭"""
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + eps * std
        else:
            return mu
    
    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """인코딩 수행"""
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        
        # 양자화
        if self.quantizer is not None:
            if isinstance(self.quantizer, VectorQuantizer):
                z_quantized, vq_loss, perplexity = self.quantizer(z.unsqueeze(-1).unsqueeze(-1))
                z_quantized = z_quantized.squeeze(-1).squeeze(-1)
                return z_quantized, mu, logvar, vq_loss
            else:
                z_quantized = self.quantizer(z)
                return z_quantized, mu, logvar, torch.tensor(0.0)
        
        return z, mu, logvar, torch.tensor(0.0)
    
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """디코딩 수행"""
        return self.decoder(z)
    
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """순전파"""
        # Encoding
        z_quantized, mu, logvar, vq_loss = self.encode(x)
        
        # Decoding
        reconstructed = self.decode(z_quantized)
        
        results = {
            'reconstructed': reconstructed,
            'mu': mu,
            'logvar': logvar,
            'latent': z_quantized,
            'vq_loss': vq_loss
        }
        
        # Discriminator forward (훈련 시에만)
        if self.use_discriminator and self.training:
            real_output = self.discriminator(x)
            fake_output = self.discriminator(reconstructed.detach())
            
            results.update({
                'real_output': real_output,
                'fake_output': fake_output
            })
        
        return results
    
    def compress(self, x: torch.Tensor) -> Dict[str, Any]:
        """압축 수행"""
        self.eval()
        with torch.no_grad():
            mu, logvar = self.encoder(x)
            z = self.reparameterize(mu, logvar)
            
            if self.quantizer is not None:
                if isinstance(self.quantizer, VectorQuantizer):
                    z_quantized, _, _ = self.quantizer(z.unsqueeze(-1).unsqueeze(-1))
                    z_quantized = z_quantized.squeeze(-1).squeeze(-1)
                else:
                    z_quantized = self.quantizer(z)
            else:
                z_quantized = z
            
            # 실제 압축에서는 엔트로피 코딩도 수행
            compressed_data = {
                'latent': z_quantized.cpu().numpy(),
                'shape': x.shape,
                'compression_ratio': self.compression_ratio
            }
            
            return compressed_data
    
    def decompress(self, compressed_data: Dict[str, Any]) -> torch.Tensor:
        """압축 해제"""
        self.eval()
        with torch.no_grad():
            latent = torch.from_numpy(compressed_data['latent']).to(next(self.parameters()).device)
            reconstructed = self.decode(latent)
            
            return reconstructed
    
    def calculate_compression_ratio(self, original_size: int, compressed_size: int) -> float:
        """압축률 계산"""
        return original_size / compressed_size
    
    def estimate_bitrate(self, latent: torch.Tensor) -> float:
        """비트율 추정"""
        # 간단한 엔트로피 추정
        latent_flat = latent.view(-1)
        unique_values = torch.unique(latent_flat)
        
        # 히스토그램 계산
        hist = torch.histc(latent_flat, bins=len(unique_values))
        probs = hist / hist.sum()
        probs = probs[probs > 0]
        
        # 엔트로피 계산 (bits)
        entropy = -torch.sum(probs * torch.log2(probs))
        
        return entropy.item()


class AdaptiveVAEGAN(VAEGANCompression):
    """적응적 VAE-GAN 압축 모델"""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        # 적응적 압축률 제어
        self.target_bitrate = kwargs.get('target_bitrate', 1.0)
        self.bitrate_controller = nn.Parameter(torch.tensor(1.0))
        
        # 품질 적응
        self.quality_threshold = kwargs.get('quality_threshold', 0.9)
        self.quality_controller = nn.Parameter(torch.tensor(1.0))
    
    def adaptive_forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """적응적 순전파"""
        results = self.forward(x)
        
        # 품질 측정 (PSNR 근사)
        mse = F.mse_loss(results['reconstructed'], x)
        psnr = 20 * torch.log10(1.0 / torch.sqrt(mse))
        
        # 비트율 추정
        current_bitrate = self.estimate_bitrate(results['latent'])
        
        # 적응적 조정
        if psnr < self.quality_threshold:
            self.quality_controller.data *= 1.01  # 품질 향상
        else:
            self.quality_controller.data *= 0.99  # 압축률 향상
        
        if current_bitrate > self.target_bitrate:
            self.bitrate_controller.data *= 0.99  # 압축률 증가
        else:
            self.bitrate_controller.data *= 1.01  # 품질 향상
        
        results.update({
            'psnr': psnr,
            'bitrate': current_bitrate,
            'quality_factor': self.quality_controller,
            'bitrate_factor': self.bitrate_controller
        })
        
        return results
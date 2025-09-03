import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter
import numpy as np
from typing import Tuple, Optional


class VectorQuantizer(nn.Module):
    """Vector Quantization-VAE (VQ-VAE) quantizer"""
    
    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        commitment_cost: float = 0.25,
        decay: float = 0.99,
        epsilon: float = 1e-5
    ):
        super().__init__()
        
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.commitment_cost = commitment_cost
        self.decay = decay
        self.epsilon = epsilon
        
        # Initialize embeddings
        self.embedding = nn.Embedding(num_embeddings, embedding_dim)
        self.embedding.weight.data.uniform_(-1/num_embeddings, 1/num_embeddings)
        
        # EMA for codebook update
        self.register_buffer('ema_cluster_size', torch.zeros(num_embeddings))
        self.register_buffer('ema_w', torch.zeros(num_embeddings, embedding_dim))
        
    def forward(self, inputs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # Convert inputs from BCHW -> BHWC
        inputs = inputs.permute(0, 2, 3, 1).contiguous()
        input_shape = inputs.shape
        
        # Flatten input
        flat_input = inputs.view(-1, self.embedding_dim)
        
        # Calculate distances
        distances = (torch.sum(flat_input**2, dim=1, keepdim=True) 
                    + torch.sum(self.embedding.weight**2, dim=1)
                    - 2 * torch.matmul(flat_input, self.embedding.weight.t()))
        
        # Encoding
        encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)
        encodings = torch.zeros(encoding_indices.shape[0], self.num_embeddings, device=inputs.device)
        encodings.scatter_(1, encoding_indices, 1)
        
        # Quantize and unflatten
        quantized = torch.matmul(encodings, self.embedding.weight).view(input_shape)
        
        # Update embeddings (EMA)
        if self.training:
            self.ema_cluster_size = self.ema_cluster_size * self.decay + \
                                 (1 - self.decay) * torch.sum(encodings, 0)
            
            n = torch.sum(self.ema_cluster_size.data)
            self.ema_cluster_size = (
                (self.ema_cluster_size + self.epsilon)
                / (n + self.num_embeddings * self.epsilon) * n
            )
            
            dw = torch.matmul(encodings.t(), flat_input)
            self.ema_w = nn.Parameter(self.ema_w * self.decay + (1 - self.decay) * dw)
            
            self.embedding.weight = nn.Parameter(self.ema_w / self.ema_cluster_size.unsqueeze(1))
        
        # Loss calculation
        e_latent_loss = F.mse_loss(quantized.detach(), inputs)
        q_latent_loss = F.mse_loss(quantized, inputs.detach())
        loss = q_latent_loss + self.commitment_cost * e_latent_loss
        
        # Straight through estimator
        quantized = inputs + (quantized - inputs).detach()
        avg_probs = torch.mean(encodings, dim=0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))
        
        # Convert back to BCHW
        quantized = quantized.permute(0, 3, 1, 2).contiguous()
        
        return quantized, loss, perplexity


class LearnedQuantizer(nn.Module):
    """학습된 양자화기"""
    
    def __init__(
        self,
        channels: int,
        num_levels: int = 256,
        learnable_scales: bool = True,
        entropy_bottleneck: bool = True
    ):
        super().__init__()
        
        self.channels = channels
        self.num_levels = num_levels
        self.learnable_scales = learnable_scales
        self.entropy_bottleneck = entropy_bottleneck
        
        if learnable_scales:
            self.scales = Parameter(torch.ones(channels))
        else:
            self.register_buffer('scales', torch.ones(channels))
        
        # Entropy model for rate estimation
        if entropy_bottleneck:
            self.entropy_model = EntropyBottleneck(channels)
        
        # Quantization levels
        self.register_buffer('levels', torch.arange(num_levels).float() - num_levels // 2)
        
    def quantize(self, x: torch.Tensor, training: bool = True) -> torch.Tensor:
        """양자화 수행"""
        if training:
            # Soft quantization (differentiable)
            x_scaled = x / self.scales.view(1, -1, 1, 1)
            
            # Gumbel softmax for differentiable quantization
            logits = -torch.abs(x_scaled.unsqueeze(-1) - self.levels.view(1, 1, 1, 1, -1))
            
            # Temperature annealing
            temperature = max(0.5, 1.0 - (self.current_epoch / 100) * 0.5) if hasattr(self, 'current_epoch') else 1.0
            
            soft_quantized = F.gumbel_softmax(logits, tau=temperature, hard=False, dim=-1)
            quantized = torch.sum(soft_quantized * self.levels.view(1, 1, 1, 1, -1), dim=-1)
            
        else:
            # Hard quantization (inference)
            x_scaled = x / self.scales.view(1, -1, 1, 1)
            quantized = torch.round(torch.clamp(x_scaled, -self.num_levels//2, self.num_levels//2-1))
        
        # Scale back
        quantized = quantized * self.scales.view(1, -1, 1, 1)
        
        return quantized
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        quantized = self.quantize(x, self.training)
        
        # Rate estimation
        if self.entropy_bottleneck and self.training:
            rate_loss = self.entropy_model(x)
        else:
            rate_loss = torch.tensor(0.0, device=x.device)
        
        return quantized, rate_loss


class EntropyBottleneck(nn.Module):
    """엔트로피 병목 레이어"""
    
    def __init__(self, channels: int, filters: Tuple[int, ...] = (3, 3, 3)):
        super().__init__()
        
        self.channels = channels
        self.filters = filters
        
        # Learnable parameters for entropy model
        self._medians = Parameter(torch.zeros(channels))
        self._scales = Parameter(torch.ones(channels))
        
        # Convolution layers for context modeling
        conv_layers = []
        in_ch = channels
        
        for filter_size in filters:
            conv_layers.extend([
                nn.Conv1d(in_ch, channels, filter_size, padding=filter_size//2),
                nn.LeakyReLU(0.2, inplace=True)
            ])
            in_ch = channels
        
        self.context_model = nn.Sequential(*conv_layers)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Estimate rate using Gaussian assumption
        B, C, H, W = x.shape
        
        # Flatten spatial dimensions
        x_flat = x.view(B, C, -1)
        
        # Context modeling
        context = self.context_model(x_flat)
        
        # Rate calculation (negative log-likelihood under Gaussian)
        medians = self._medians.view(1, -1, 1)
        scales = F.softplus(self._scales.view(1, -1, 1))
        
        # Gaussian log-likelihood
        log_likelihood = -0.5 * torch.log(2 * np.pi * scales**2) - \
                        0.5 * ((x_flat - medians) / scales)**2
        
        # Average rate per pixel
        rate = -torch.mean(log_likelihood)
        
        return rate


class UniformQuantizer(nn.Module):
    """균등 양자화기"""
    
    def __init__(self, num_bits: int = 8, learnable: bool = False):
        super().__init__()
        
        self.num_bits = num_bits
        self.num_levels = 2 ** num_bits
        self.learnable = learnable
        
        if learnable:
            self.step_size = Parameter(torch.tensor(1.0))
        else:
            self.register_buffer('step_size', torch.tensor(1.0))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.training:
            # Straight-through estimator
            x_scaled = x / self.step_size
            x_quantized = torch.round(torch.clamp(x_scaled, -self.num_levels//2, self.num_levels//2-1))
            x_quantized = x_quantized * self.step_size
            
            # Gradient pass-through
            return x + (x_quantized - x).detach()
        else:
            # Hard quantization
            x_scaled = x / self.step_size
            x_quantized = torch.round(torch.clamp(x_scaled, -self.num_levels//2, self.num_levels//2-1))
            return x_quantized * self.step_size
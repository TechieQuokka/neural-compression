import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Optional
from .attention import MultiHeadAttention, CBAM, SelfAttention


class TransposeConvBlock(nn.Module):
    """Transpose convolution block with residual connection"""
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 4,
        stride: int = 2,
        padding: int = 1,
        use_attention: bool = False,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.transpose_conv = nn.ConvTranspose2d(
            in_channels, out_channels, kernel_size, stride, padding, bias=False
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout2d(dropout)
        
        # Skip connection adapter
        if in_channels != out_channels or stride != 1:
            self.skip_conv = nn.Sequential(
                nn.ConvTranspose2d(in_channels, out_channels, 1, stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.skip_conv = nn.Identity()
        
        # Attention mechanism
        if use_attention:
            self.attention = CBAM(out_channels)
        else:
            self.attention = nn.Identity()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Transpose convolution
        out = self.activation(self.bn(self.transpose_conv(x)))
        out = self.dropout(out)
        
        # Skip connection
        skip = self.skip_conv(x)
        if skip.shape != out.shape:
            skip = F.interpolate(skip, size=out.shape[2:], mode='bilinear', align_corners=False)
        
        out = out + skip
        
        # Attention
        out = self.attention(out)
        
        return out


class PixelShuffleBlock(nn.Module):
    """Pixel shuffle upsampling block"""
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        upscale_factor: int = 2,
        use_attention: bool = False
    ):
        super().__init__()
        
        self.upscale_factor = upscale_factor
        intermediate_channels = out_channels * (upscale_factor ** 2)
        
        self.conv = nn.Conv2d(in_channels, intermediate_channels, 3, 1, 1)
        self.pixel_shuffle = nn.PixelShuffle(upscale_factor)
        self.bn = nn.BatchNorm2d(out_channels)
        self.activation = nn.GELU()
        
        if use_attention:
            self.attention = SelfAttention(out_channels)
        else:
            self.attention = nn.Identity()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.pixel_shuffle(x)
        x = self.activation(self.bn(x))
        x = self.attention(x)
        return x


class CompressionDecoder(nn.Module):
    """고급 압축 디코더"""
    
    def __init__(
        self,
        latent_dim: int = 256,
        out_channels: int = 3,
        base_channels: int = 512,
        num_stages: int = 4,
        use_attention: bool = True,
        attention_stages: List[int] = [1, 2],
        upsampling_mode: str = "transpose",  # "transpose" or "pixel_shuffle"
        dropout: float = 0.1,
        target_size: int = 256  # 목표 출력 크기
    ):
        super().__init__()
        
        self.latent_dim = latent_dim
        self.out_channels = out_channels
        self.base_channels = base_channels
        self.num_stages = num_stages
        self.use_attention = use_attention
        self.upsampling_mode = upsampling_mode
        self.target_size = target_size
        
        # 목표 크기에 따라 초기 크기와 스테이지 수 계산
        self.num_upsampling_stages = int(np.log2(target_size)) - 2  # 4x4부터 시작
        initial_size = 4
        self.initial_size = initial_size
        
        self.latent_projection = nn.Sequential(
            nn.Linear(latent_dim, base_channels * initial_size * initial_size),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        
        # Global attention on latent features
        if use_attention:
            self.latent_attention = MultiHeadAttention(
                d_model=base_channels,
                num_heads=8,
                dropout=dropout
            )
        
        # Decoder stages
        self.stages = nn.ModuleList()
        current_channels = base_channels
        
        # 목표 크기에 맞는 스테이지 수 사용
        for stage_idx in range(self.num_upsampling_stages):
            out_channels_stage = max(base_channels // (2 ** (stage_idx + 1)), 32)
            use_attn = use_attention and (stage_idx in attention_stages)
            
            if upsampling_mode == "transpose":
                stage = TransposeConvBlock(
                    current_channels,
                    out_channels_stage,
                    use_attention=use_attn,
                    dropout=dropout
                )
            else:  # pixel_shuffle
                stage = PixelShuffleBlock(
                    current_channels,
                    out_channels_stage,
                    use_attention=use_attn
                )
            
            self.stages.append(stage)
            current_channels = out_channels_stage
        
        # Output head
        self.output_head = nn.Sequential(
            nn.Conv2d(current_channels, current_channels, 3, 1, 1),
            nn.BatchNorm2d(current_channels),
            nn.GELU(),
            nn.Conv2d(current_channels, out_channels, 3, 1, 1),
            nn.Tanh()  # Output in [-1, 1] range
        )
        
        # Refinement layers
        self.refinement = nn.Sequential(
            nn.Conv2d(out_channels, out_channels * 2, 3, 1, 1),
            nn.GELU(),
            nn.Conv2d(out_channels * 2, out_channels, 3, 1, 1),
            nn.Sigmoid()  # Gate for refinement
        )
    
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        B = z.shape[0]
        
        # Project latent to feature map
        x = self.latent_projection(z)
        x = x.view(B, self.base_channels, self.initial_size, self.initial_size)
        
        # Global attention on latent features
        if hasattr(self, 'latent_attention'):
            x = self.latent_attention(x)
        
        # Decoder stages
        for stage in self.stages:
            x = stage(x)
        
        # Output generation
        base_output = self.output_head(x)
        
        # Refinement
        refinement_gate = self.refinement(base_output)
        refined_output = base_output * refinement_gate
        
        return refined_output


class ProgressiveDecoder(nn.Module):
    """점진적 디코더 (Progressive GAN style)"""
    
    def __init__(
        self,
        latent_dim: int = 256,
        out_channels: int = 3,
        max_resolution: int = 256,
        base_channels: int = 512
    ):
        super().__init__()
        
        self.latent_dim = latent_dim
        self.out_channels = out_channels
        self.max_resolution = max_resolution
        self.base_channels = base_channels
        
        # Calculate number of stages needed
        self.num_stages = int(np.log2(max_resolution)) - 2  # Start from 4x4
        
        # Initial block (latent -> 4x4)
        self.initial_block = nn.Sequential(
            nn.Linear(latent_dim, base_channels * 4 * 4),
            nn.GELU(),
            nn.Unflatten(1, (base_channels, 4, 4))
        )
        
        # Progressive blocks
        self.progressive_blocks = nn.ModuleList()
        self.to_rgb_layers = nn.ModuleList()
        
        current_channels = base_channels
        
        for stage in range(self.num_stages):
            next_channels = max(current_channels // 2, 64)
            
            # Upsampling block
            block = nn.Sequential(
                nn.ConvTranspose2d(current_channels, next_channels, 4, 2, 1, bias=False),
                nn.BatchNorm2d(next_channels),
                nn.GELU(),
                nn.Conv2d(next_channels, next_channels, 3, 1, 1, bias=False),
                nn.BatchNorm2d(next_channels),
                nn.GELU()
            )
            
            # RGB output layer for this resolution
            to_rgb = nn.Conv2d(next_channels, out_channels, 1)
            
            self.progressive_blocks.append(block)
            self.to_rgb_layers.append(to_rgb)
            
            current_channels = next_channels
        
        self.current_stage = 0
        self.alpha = 1.0  # Fade-in parameter
    
    def set_stage(self, stage: int, alpha: float = 1.0):
        """현재 스테이지 설정"""
        self.current_stage = min(stage, len(self.progressive_blocks) - 1)
        self.alpha = alpha
    
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        x = self.initial_block(z)
        
        # Progressive generation
        for stage in range(self.current_stage + 1):
            x = self.progressive_blocks[stage]
            
            if stage == self.current_stage:
                # Current resolution output
                out = self.to_rgb_layers[stage](x)
                
                # Fade-in with previous resolution
                if stage > 0 and self.alpha < 1.0:
                    prev_out = self.to_rgb_layers[stage - 1](
                        F.interpolate(x_prev, scale_factor=2, mode='bilinear')
                    )
                    out = self.alpha * out + (1 - self.alpha) * prev_out
                
                return torch.tanh(out)
            
            x_prev = x
        
        return torch.tanh(self.to_rgb_layers[self.current_stage](x))


class AdaptiveDecoder(nn.Module):
    """적응적 디코더 (입력에 따라 구조 조정)"""
    
    def __init__(
        self,
        latent_dim: int = 256,
        out_channels: int = 3,
        base_channels: int = 256
    ):
        super().__init__()
        
        self.latent_dim = latent_dim
        self.out_channels = out_channels
        self.base_channels = base_channels
        
        # Adaptive routing network
        self.router = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.GELU(),
            nn.Linear(128, 4),  # 4 different decoder paths
            nn.Softmax(dim=1)
        )
        
        # Multiple decoder paths
        self.decoders = nn.ModuleList([
            self._create_decoder_path(complexity_level=i)
            for i in range(4)
        ])
        
    def _create_decoder_path(self, complexity_level: int) -> nn.Module:
        """복잡도 수준에 따른 디코더 경로 생성"""
        
        num_layers = 3 + complexity_level
        channels = self.base_channels // (2 ** complexity_level)
        
        layers = [
            nn.Linear(self.latent_dim, channels * 8 * 8),
            nn.GELU(),
            nn.Unflatten(1, (channels, 8, 8))
        ]
        
        current_channels = channels
        for _ in range(num_layers):
            next_channels = max(current_channels // 2, 32)
            
            layers.extend([
                nn.ConvTranspose2d(current_channels, next_channels, 4, 2, 1),
                nn.BatchNorm2d(next_channels),
                nn.GELU()
            ])
            
            current_channels = next_channels
        
        # Output layer
        layers.append(nn.Conv2d(current_channels, self.out_channels, 3, 1, 1))
        layers.append(nn.Tanh())
        
        return nn.Sequential(*layers)
    
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        # Route selection
        routing_weights = self.router(z)
        
        # Generate outputs from all paths
        outputs = []
        for decoder in self.decoders:
            output = decoder(z)
            outputs.append(output)
        
        # Weighted combination
        final_output = torch.zeros_like(outputs[0])
        for i, output in enumerate(outputs):
            weight = routing_weights[:, i:i+1, None, None]
            final_output += weight * output
        
        return final_output
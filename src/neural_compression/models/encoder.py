import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, List
from .attention import MultiHeadAttention, CBAM, SelfAttention


class ResidualBlock(nn.Module):
    """Residual Block with optional attention"""
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        use_attention: bool = False,
        attention_type: str = "cbam"
    ):
        super().__init__()
        
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.downsample = None
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        
        self.attention = None
        if use_attention:
            if attention_type == "cbam":
                self.attention = CBAM(out_channels)
            elif attention_type == "self":
                self.attention = SelfAttention(out_channels)
        
        self.activation = nn.GELU()
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        
        out = self.activation(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        
        if self.downsample is not None:
            residual = self.downsample(x)
        
        out += residual
        
        if self.attention is not None:
            out = self.attention(out)
        
        out = self.activation(out)
        return out


class CompressionEncoder(nn.Module):
    """고급 압축 인코더"""
    
    def __init__(
        self,
        in_channels: int = 3,
        latent_dim: int = 256,
        base_channels: int = 64,
        num_stages: int = 4,
        use_attention: bool = True,
        attention_stages: List[int] = [2, 3],
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.in_channels = in_channels
        self.latent_dim = latent_dim
        self.base_channels = base_channels
        self.num_stages = num_stages
        self.use_attention = use_attention
        
        # Initial convolution
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, 7, 2, 3, bias=False),
            nn.BatchNorm2d(base_channels),
            nn.GELU(),
            nn.MaxPool2d(3, 2, 1)
        )
        
        # Encoder stages
        self.stages = nn.ModuleList()
        current_channels = base_channels
        
        for stage_idx in range(num_stages):
            out_channels = base_channels * (2 ** stage_idx)
            stride = 2 if stage_idx > 0 else 1
            use_attn = use_attention and (stage_idx in attention_stages)
            
            stage = self._make_stage(
                current_channels,
                out_channels,
                stride=stride,
                use_attention=use_attn,
                num_blocks=2
            )
            
            self.stages.append(stage)
            current_channels = out_channels
        
        # Global attention layer
        if use_attention:
            self.global_attention = MultiHeadAttention(
                d_model=current_channels,
                num_heads=8,
                dropout=dropout
            )
        
        # Final compression layers
        self.compression_head = nn.Sequential(
            nn.AdaptiveAvgPool2d((8, 8)),
            nn.Flatten(),
            nn.Linear(current_channels * 64, latent_dim * 2),  # mu and logvar
            nn.Dropout(dropout)
        )
        
    def _make_stage(
        self,
        in_channels: int,
        out_channels: int,
        stride: int,
        use_attention: bool,
        num_blocks: int = 2
    ) -> nn.Sequential:
        """스테이지 생성"""
        layers = []
        
        # First block with potential stride
        layers.append(ResidualBlock(
            in_channels,
            out_channels,
            stride=stride,
            use_attention=use_attention
        ))
        
        # Additional blocks
        for _ in range(num_blocks - 1):
            layers.append(ResidualBlock(
                out_channels,
                out_channels,
                stride=1,
                use_attention=use_attention
            ))
        
        return nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # Initial processing
        x = self.stem(x)
        
        # Encoder stages
        for stage in self.stages:
            x = stage(x)
        
        # Global attention
        if hasattr(self, 'global_attention'):
            x = self.global_attention(x)
        
        # Final compression
        compressed = self.compression_head(x)
        
        # Split into mu and logvar for VAE
        mu, logvar = compressed.chunk(2, dim=1)
        
        return mu, logvar


class EfficientEncoder(nn.Module):
    """EfficientNet 기반 효율적인 인코더"""
    
    def __init__(
        self,
        in_channels: int = 3,
        latent_dim: int = 256,
        compound_coeff: int = 0,
        use_attention: bool = True
    ):
        super().__init__()
        
        # EfficientNet-like scaling
        width_mult, depth_mult, resolution = self._get_scaling_params(compound_coeff)
        
        base_channels = int(32 * width_mult)
        
        # Stem
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, 3, 2, 1, bias=False),
            nn.BatchNorm2d(base_channels),
            nn.SiLU(inplace=True)
        )
        
        # MBConv blocks
        self.blocks = nn.ModuleList()
        
        # Stage configurations: (expand_ratio, channels, repeats, stride)
        stage_configs = [
            (1, 16, 1, 1),
            (6, 24, 2, 2),
            (6, 40, 2, 2),
            (6, 80, 3, 2),
            (6, 112, 3, 1),
            (6, 192, 4, 2),
            (6, 320, 1, 1)
        ]
        
        current_channels = base_channels
        
        for expand_ratio, out_channels, repeats, stride in stage_configs:
            out_channels = int(out_channels * width_mult)
            repeats = int(repeats * depth_mult)
            
            for i in range(repeats):
                block_stride = stride if i == 0 else 1
                self.blocks.append(
                    MBConvBlock(
                        current_channels,
                        out_channels,
                        expand_ratio=expand_ratio,
                        stride=block_stride,
                        use_attention=use_attention and (len(self.blocks) > 10)
                    )
                )
                current_channels = out_channels
        
        # Head
        head_channels = int(1280 * width_mult)
        self.head = nn.Sequential(
            nn.Conv2d(current_channels, head_channels, 1, bias=False),
            nn.BatchNorm2d(head_channels),
            nn.SiLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(0.2),
            nn.Linear(head_channels, latent_dim * 2)
        )
    
    def _get_scaling_params(self, compound_coeff: int) -> Tuple[float, float, int]:
        """EfficientNet 스케일링 파라미터"""
        scaling_configs = {
            0: (1.0, 1.0, 224),
            1: (1.0, 1.1, 240),
            2: (1.1, 1.2, 260),
            3: (1.2, 1.4, 300),
            4: (1.4, 1.8, 380),
            5: (1.6, 2.2, 456),
            6: (1.8, 2.6, 528),
            7: (2.0, 3.1, 600)
        }
        return scaling_configs.get(compound_coeff, (1.0, 1.0, 224))
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.stem(x)
        
        for block in self.blocks:
            x = block(x)
        
        x = self.head(x)
        
        mu, logvar = x.chunk(2, dim=1)
        return mu, logvar


class MBConvBlock(nn.Module):
    """Mobile Inverted Bottleneck Convolution Block"""
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        expand_ratio: int = 6,
        kernel_size: int = 3,
        stride: int = 1,
        se_ratio: float = 0.25,
        use_attention: bool = False,
        dropout_rate: float = 0.2
    ):
        super().__init__()
        
        self.use_residual = (stride == 1 and in_channels == out_channels)
        expanded_channels = in_channels * expand_ratio
        
        # Expansion phase
        if expand_ratio != 1:
            self.expand_conv = nn.Sequential(
                nn.Conv2d(in_channels, expanded_channels, 1, bias=False),
                nn.BatchNorm2d(expanded_channels),
                nn.SiLU(inplace=True)
            )
        else:
            self.expand_conv = nn.Identity()
        
        # Depthwise convolution
        self.depthwise_conv = nn.Sequential(
            nn.Conv2d(
                expanded_channels, expanded_channels, kernel_size,
                stride, kernel_size // 2, groups=expanded_channels, bias=False
            ),
            nn.BatchNorm2d(expanded_channels),
            nn.SiLU(inplace=True)
        )
        
        # Squeeze-and-Excitation
        if se_ratio > 0:
            se_channels = max(1, int(in_channels * se_ratio))
            self.se = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Conv2d(expanded_channels, se_channels, 1),
                nn.SiLU(inplace=True),
                nn.Conv2d(se_channels, expanded_channels, 1),
                nn.Sigmoid()
            )
        else:
            self.se = nn.Identity()
        
        # Output projection
        self.project_conv = nn.Sequential(
            nn.Conv2d(expanded_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels)
        )
        
        # Attention mechanism
        if use_attention:
            self.attention = SelfAttention(out_channels)
        else:
            self.attention = nn.Identity()
        
        # Dropout
        if dropout_rate > 0 and self.use_residual:
            self.dropout = nn.Dropout2d(dropout_rate)
        else:
            self.dropout = nn.Identity()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        
        # Expansion
        x = self.expand_conv(x)
        
        # Depthwise
        x = self.depthwise_conv(x)
        
        # Squeeze-and-Excitation
        if not isinstance(self.se, nn.Identity):
            se_out = self.se(x)
            x = x * se_out
        
        # Projection
        x = self.project_conv(x)
        
        # Attention
        x = self.attention(x)
        
        # Residual connection
        if self.use_residual:
            x = self.dropout(x)
            x = x + residual
        
        return x
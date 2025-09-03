import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List
from .attention import SelfAttention


class SpectralNorm(nn.Module):
    """Spectral Normalization wrapper"""
    
    def __init__(self, module: nn.Module, name: str = 'weight', n_power_iterations: int = 1):
        super().__init__()
        self.module = module
        self.name = name
        self.n_power_iterations = n_power_iterations
        
        if not self._made_params():
            self._make_params()
    
    def _made_params(self):
        try:
            getattr(self.module, self.name + "_u")
            getattr(self.module, self.name + "_v")
            getattr(self.module, self.name + "_bar")
            return True
        except AttributeError:
            return False
    
    def _make_params(self):
        weight = getattr(self.module, self.name)
        
        height = weight.data.shape[0]
        width = weight.data.view(height, -1).shape[1]
        
        u = nn.Parameter(weight.data.new(height).normal_(0, 1), requires_grad=False)
        v = nn.Parameter(weight.data.new(width).normal_(0, 1), requires_grad=False)
        weight_bar = nn.Parameter(weight.data)
        
        del self.module._parameters[self.name]
        
        self.module.register_parameter(self.name + "_u", u)
        self.module.register_parameter(self.name + "_v", v)  
        self.module.register_parameter(self.name + "_bar", weight_bar)
    
    def _update_u_v(self):
        weight = getattr(self.module, self.name + "_bar")
        u = getattr(self.module, self.name + "_u")
        v = getattr(self.module, self.name + "_v")
        
        height = weight.data.shape[0]
        weight_mat = weight.data.view(height, -1)
        
        for _ in range(self.n_power_iterations):
            v.data = F.normalize(torch.mv(weight_mat.t(), u.data), dim=0, eps=1e-12)
            u.data = F.normalize(torch.mv(weight_mat, v.data), dim=0, eps=1e-12)
        
        sigma = torch.dot(u.data, torch.mv(weight_mat, v.data))
        setattr(self.module, self.name, weight / sigma.expand_as(weight))
    
    def forward(self, *args):
        self._update_u_v()
        return self.module.forward(*args)


class PatchGANDiscriminator(nn.Module):
    """PatchGAN 판별기"""
    
    def __init__(
        self,
        in_channels: int = 3,
        ndf: int = 64,
        n_layers: int = 3,
        use_spectral_norm: bool = True,
        use_attention: bool = False
    ):
        super().__init__()
        
        self.use_spectral_norm = use_spectral_norm
        self.use_attention = use_attention
        
        # Initial layer
        layers = [
            nn.Conv2d(in_channels, ndf, 4, 2, 1),
            nn.LeakyReLU(0.2, inplace=True)
        ]
        
        # Intermediate layers
        nf_mult = 1
        for n in range(1, n_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            
            conv = nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, 4, 2, 1, bias=False)
            if use_spectral_norm:
                conv = SpectralNorm(conv)
            
            layers.extend([
                conv,
                nn.BatchNorm2d(ndf * nf_mult),
                nn.LeakyReLU(0.2, inplace=True)
            ])
            
            # Add attention in middle layers
            if use_attention and n == n_layers // 2:
                layers.append(SelfAttention(ndf * nf_mult))
        
        # Final layer
        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        
        final_conv = nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, 4, 1, 1, bias=False)
        if use_spectral_norm:
            final_conv = SpectralNorm(final_conv)
        
        layers.extend([
            final_conv,
            nn.BatchNorm2d(ndf * nf_mult),
            nn.LeakyReLU(0.2, inplace=True)
        ])
        
        # Output layer
        output_conv = nn.Conv2d(ndf * nf_mult, 1, 4, 1, 1)
        if use_spectral_norm:
            output_conv = SpectralNorm(output_conv)
        
        layers.append(output_conv)
        
        self.model = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


class MultiScaleDiscriminator(nn.Module):
    """다중 해상도 판별기"""
    
    def __init__(
        self,
        in_channels: int = 3,
        num_scales: int = 3,
        ndf: int = 64,
        n_layers: int = 3,
        use_spectral_norm: bool = True
    ):
        super().__init__()
        
        self.num_scales = num_scales
        
        self.discriminators = nn.ModuleList([
            PatchGANDiscriminator(
                in_channels=in_channels,
                ndf=ndf,
                n_layers=n_layers,
                use_spectral_norm=use_spectral_norm,
                use_attention=(i == 0)  # Only use attention in finest scale
            )
            for i in range(num_scales)
        ])
        
        self.downsample = nn.AvgPool2d(3, stride=2, padding=1, count_include_pad=False)
    
    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        outputs = []
        current_x = x
        
        for i, discriminator in enumerate(self.discriminators):
            outputs.append(discriminator(current_x))
            
            if i < self.num_scales - 1:
                current_x = self.downsample(current_x)
        
        return outputs


class FeatureMatchingDiscriminator(nn.Module):
    """Feature matching을 위한 판별기"""
    
    def __init__(
        self,
        in_channels: int = 3,
        ndf: int = 64,
        n_layers: int = 4,
        use_spectral_norm: bool = True
    ):
        super().__init__()
        
        self.features = nn.ModuleList()
        
        # Build feature extraction layers
        current_channels = in_channels
        
        for i in range(n_layers):
            out_channels = ndf * min(2 ** i, 8)
            stride = 2 if i > 0 else 1
            
            conv = nn.Conv2d(current_channels, out_channels, 4, stride, 1, bias=False)
            if use_spectral_norm:
                conv = SpectralNorm(conv)
            
            if i == 0:
                layer = nn.Sequential(
                    conv,
                    nn.LeakyReLU(0.2, inplace=True)
                )
            else:
                layer = nn.Sequential(
                    conv,
                    nn.BatchNorm2d(out_channels),
                    nn.LeakyReLU(0.2, inplace=True)
                )
            
            self.features.append(layer)
            current_channels = out_channels
        
        # Final classification layer
        final_conv = nn.Conv2d(current_channels, 1, 4, 1, 1)
        if use_spectral_norm:
            final_conv = SpectralNorm(final_conv)
        
        self.classifier = final_conv
    
    def forward(self, x: torch.Tensor, return_features: bool = False):
        features = []
        
        for layer in self.features:
            x = layer(x)
            if return_features:
                features.append(x)
        
        output = self.classifier(x)
        
        if return_features:
            return output, features
        else:
            return output
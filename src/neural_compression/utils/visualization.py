import matplotlib.pyplot as plt
import torch
import numpy as np
from typing import List, Tuple, Optional
import seaborn as sns


def plot_compression_results(
    original: torch.Tensor,
    reconstructed: torch.Tensor,
    save_path: Optional[str] = None
) -> plt.Figure:
    """압축 결과 시각화"""
    
    # Tensor를 numpy로 변환
    if original.dim() == 4:
        original = original[0]
    if reconstructed.dim() == 4:
        reconstructed = reconstructed[0]
    
    original = original.detach().cpu().permute(1, 2, 0).numpy()
    reconstructed = reconstructed.detach().cpu().permute(1, 2, 0).numpy()
    
    # 정규화
    original = np.clip(original, 0, 1)
    reconstructed = np.clip(reconstructed, 0, 1)
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # 원본 이미지
    axes[0].imshow(original)
    axes[0].set_title('Original')
    axes[0].axis('off')
    
    # 재구성 이미지
    axes[1].imshow(reconstructed)
    axes[1].set_title('Reconstructed')
    axes[1].axis('off')
    
    # 차이 이미지
    diff = np.abs(original - reconstructed)
    im = axes[2].imshow(diff, cmap='hot')
    axes[2].set_title('Difference')
    axes[2].axis('off')
    plt.colorbar(im, ax=axes[2])
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig


def visualize_latent_space(
    latent_codes: torch.Tensor,
    labels: Optional[torch.Tensor] = None,
    method: str = 'pca',
    save_path: Optional[str] = None
) -> plt.Figure:
    """잠재 공간 시각화"""
    
    latent_np = latent_codes.detach().cpu().numpy()
    
    if method == 'pca':
        from sklearn.decomposition import PCA
        reducer = PCA(n_components=2)
    elif method == 'tsne':
        from sklearn.manifold import TSNE
        reducer = TSNE(n_components=2, random_state=42)
    else:
        raise ValueError(f"Unknown method: {method}")
    
    # 차원 축소
    reduced = reducer.fit_transform(latent_np.reshape(latent_np.shape[0], -1))
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    if labels is not None:
        labels_np = labels.detach().cpu().numpy()
        scatter = ax.scatter(reduced[:, 0], reduced[:, 1], c=labels_np, cmap='tab10')
        plt.colorbar(scatter)
    else:
        ax.scatter(reduced[:, 0], reduced[:, 1], alpha=0.6)
    
    ax.set_title(f'Latent Space Visualization ({method.upper()})')
    ax.set_xlabel('Component 1')
    ax.set_ylabel('Component 2')
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig


def plot_training_curves(
    metrics_history: dict,
    save_path: Optional[str] = None
) -> plt.Figure:
    """훈련 곡선 시각화"""
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    
    # Loss curves
    if 'train_loss' in metrics_history and 'val_loss' in metrics_history:
        axes[0, 0].plot(metrics_history['train_loss'], label='Train')
        axes[0, 0].plot(metrics_history['val_loss'], label='Validation')
        axes[0, 0].set_title('Loss')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
    
    # PSNR
    if 'val_psnr' in metrics_history:
        axes[0, 1].plot(metrics_history['val_psnr'])
        axes[0, 1].set_title('PSNR')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('PSNR (dB)')
        axes[0, 1].grid(True)
    
    # SSIM
    if 'val_ssim' in metrics_history:
        axes[1, 0].plot(metrics_history['val_ssim'])
        axes[1, 0].set_title('SSIM')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('SSIM')
        axes[1, 0].grid(True)
    
    # Compression Ratio
    if 'val_compression_ratio' in metrics_history:
        axes[1, 1].plot(metrics_history['val_compression_ratio'])
        axes[1, 1].set_title('Compression Ratio')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Ratio')
        axes[1, 1].grid(True)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig


def plot_rate_distortion_curve(
    bitrates: List[float],
    psnr_values: List[float],
    ssim_values: List[float],
    save_path: Optional[str] = None
) -> plt.Figure:
    """Rate-Distortion 곡선 시각화"""
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # PSNR vs Bitrate
    ax1.plot(bitrates, psnr_values, 'b-o', linewidth=2, markersize=6)
    ax1.set_xlabel('Bitrate (bpp)')
    ax1.set_ylabel('PSNR (dB)')
    ax1.set_title('Rate-Distortion (PSNR)')
    ax1.grid(True, alpha=0.3)
    
    # SSIM vs Bitrate
    ax2.plot(bitrates, ssim_values, 'r-s', linewidth=2, markersize=6)
    ax2.set_xlabel('Bitrate (bpp)')
    ax2.set_ylabel('SSIM')
    ax2.set_title('Rate-Distortion (SSIM)')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig
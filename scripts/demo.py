#!/usr/bin/env python3
"""
Neural Compression 데모 스크립트
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent / "src"))

import torch
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np

from neural_compression.models.vae_gan_compression import VAEGANCompression
from neural_compression.utils.metrics import CompressionMetrics
from neural_compression.utils.visualization import plot_compression_results


def demo_compression():
    """압축 데모 실행"""
    
    print("Neural Compression Demo")
    print("=" * 30)
    
    # 모델 생성
    print("모델 초기화 중...")
    model = VAEGANCompression(
        latent_dim=256,
        use_discriminator=False,
        quantization_method="vector"
    )
    model.eval()
    
    # 테스트 이미지 로드 (샘플 이미지 사용)
    sample_image_path = "data/flower_photos/daisy/5547758_eea9edfd54_n.jpg"
    
    if Path(sample_image_path).exists():
        print(f"이미지 로드: {sample_image_path}")
        image = Image.open(sample_image_path).convert('RGB')
        image = image.resize((256, 256))
        
        # 텐서로 변환
        image_tensor = torch.from_numpy(np.array(image)).permute(2, 0, 1).float() / 255.0
        image_tensor = image_tensor.unsqueeze(0)
    else:
        print("샘플 이미지 생성...")
        image_tensor = torch.randn(1, 3, 256, 256).clamp(0, 1)
    
    print(f"입력 이미지 크기: {image_tensor.shape}")
    
    # 압축 및 복원
    print("압축 수행 중...")
    with torch.no_grad():
        # Forward pass
        results = model(image_tensor)
        reconstructed = results['reconstructed']
        
        # 메트릭 계산
        psnr = CompressionMetrics.psnr(reconstructed, image_tensor)
        ssim = CompressionMetrics.ssim(reconstructed, image_tensor)
        
        # 압축률 계산
        original_bits = image_tensor.numel() * 32
        compressed_bits = results['latent'].numel() * 8
        compression_ratio = CompressionMetrics.compression_ratio(original_bits, compressed_bits)
        bpp = CompressionMetrics.bits_per_pixel(compressed_bits, 256, 256)
    
    # 결과 출력
    print("\n압축 결과:")
    print(f"  PSNR: {psnr:.2f} dB")
    print(f"  SSIM: {ssim:.4f}")
    print(f"  압축률: {compression_ratio:.1f}:1")
    print(f"  비트/픽셀: {bpp:.3f}")
    
    # 시각화
    print("\n결과 시각화 중...")
    fig = plot_compression_results(image_tensor, reconstructed, "results/demo_result.png")
    
    # 결과 저장
    Path("results").mkdir(exist_ok=True)
    plt.savefig("results/compression_demo.png", dpi=150, bbox_inches='tight')
    print("결과가 results/compression_demo.png에 저장되었습니다.")
    
    return {
        'psnr': psnr.item(),
        'ssim': ssim.item(),
        'compression_ratio': compression_ratio,
        'bpp': bpp
    }


if __name__ == "__main__":
    demo_compression()
#!/usr/bin/env python3
"""간단한 훈련 테스트"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent / "src"))

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np

from neural_compression.models.vae_gan_compression import VAEGANCompression
from neural_compression.data.datasets import CIFARDataset
from neural_compression.data.transforms import CompressionTransforms


def simple_train():
    """간단한 훈련 루프"""
    
    print("간단한 훈련 테스트 시작")
    
    # 디바이스 설정
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"사용 디바이스: {device}")
    
    # 모델 생성 (VAE만, 판별기 없이)
    model = VAEGANCompression(
        latent_dim=128,
        use_discriminator=False,
        quantization_method=None  # 양자화도 비활성화
    ).to(device)
    
    # 데이터 생성 (256x256 크기로 변경)
    transforms = CompressionTransforms(image_size=(256, 256), augment=False)
    dataset = CIFARDataset(
        cifar_type="cifar10",
        train=True,
        transforms=transforms.get_train_transforms()
    )
    
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True, num_workers=0)  # 256x256이므로 배치 크기 축소
    
    # 옵티마이저
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    
    print(f"데이터셋 크기: {len(dataset)}")
    print(f"모델 파라미터 수: {sum(p.numel() for p in model.parameters()):,}")
    
    # 훈련 루프
    model.train()
    
    for epoch in range(2):
        epoch_loss = 0.0
        num_batches = 0
        
        for batch_idx, batch in enumerate(dataloader):
            if batch_idx >= 10:  # 10개 배치만 테스트
                break
                
            images = batch['image'].to(device)
            
            # Forward pass
            optimizer.zero_grad()
            
            results = model(images)
            reconstructed = results['reconstructed']
            mu = results['mu']
            logvar = results['logvar']
            
            # 간단한 손실만 계산
            recon_loss = F.mse_loss(reconstructed, images)
            kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / images.numel()
            
            total_loss = recon_loss + 0.001 * kl_loss
            
            # Backward pass
            total_loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            
            optimizer.step()
            
            epoch_loss += total_loss.item()
            num_batches += 1
            
            if batch_idx % 5 == 0:
                print(f"Epoch {epoch}, Batch {batch_idx}: Loss = {total_loss.item():.4f}, "
                      f"Recon = {recon_loss.item():.4f}, KL = {kl_loss.item():.6f}")
        
        avg_loss = epoch_loss / num_batches
        print(f"Epoch {epoch} 완료: 평균 손실 = {avg_loss:.4f}")
    
    print("간단한 훈련 테스트 완료!")
    
    # 테스트 이미지로 결과 확인
    model.eval()
    with torch.no_grad():
        test_batch = next(iter(dataloader))
        test_images = test_batch['image'][:2].to(device)
        
        results = model(test_images)
        reconstructed = results['reconstructed']
        
        # PSNR 계산
        mse = F.mse_loss(reconstructed, test_images)
        psnr = 20 * torch.log10(1.0 / torch.sqrt(mse))
        
        print(f"테스트 PSNR: {psnr:.2f} dB")
        print(f"입력 형태: {test_images.shape}")
        print(f"출력 형태: {reconstructed.shape}")


if __name__ == "__main__":
    simple_train()
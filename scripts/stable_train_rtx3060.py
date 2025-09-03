#!/usr/bin/env python3
"""RTX 3060 12GB 안정화된 훈련 스크립트"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent / "src"))

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
import os
from tqdm import tqdm

from neural_compression.models.vae_gan_compression import VAEGANCompression
from neural_compression.data.datasets import CIFARDataset
from neural_compression.data.transforms import CompressionTransforms


def initialize_weights(m):
    """안정적인 가중치 초기화"""
    if isinstance(m, (torch.nn.Conv2d, torch.nn.ConvTranspose2d)):
        torch.nn.init.xavier_normal_(m.weight, gain=0.02)
        if m.bias is not None:
            torch.nn.init.constant_(m.bias, 0)
    elif isinstance(m, torch.nn.BatchNorm2d):
        torch.nn.init.constant_(m.weight, 1)
        torch.nn.init.constant_(m.bias, 0)
    elif isinstance(m, torch.nn.Linear):
        torch.nn.init.xavier_normal_(m.weight, gain=0.02)
        if m.bias is not None:
            torch.nn.init.constant_(m.bias, 0)


def stable_rtx3060_train():
    """안정화된 RTX 3060 12GB 훈련"""
    
    print("안정화된 RTX 3060 훈련 시작")
    
    # GPU 설정
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"디바이스: {device}")
    
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name()}")
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
    
    # 더 작은 모델로 시작
    model = VAEGANCompression(
        latent_dim=64,  # 더 작은 latent dimension
        use_discriminator=False,
        quantization_method=None,
        base_channels=32,  # 더 작은 채널 수
        compression_ratio=8
    ).to(device)
    
    # 안정적인 가중치 초기화
    model.apply(initialize_weights)
    
    # 간단한 transforms (augmentation 비활성화)
    transforms = CompressionTransforms(image_size=(128, 128), augment=False)  # 더 작은 이미지 크기
    dataset = CIFARDataset(
        cifar_type="cifar10",
        train=True,
        transforms=transforms.get_train_transforms()
    )
    
    # 작은 배치 크기
    dataloader = DataLoader(
        dataset, 
        batch_size=4,
        shuffle=True, 
        num_workers=2,
        pin_memory=True
    )
    
    # 보수적인 옵티마이저 설정
    optimizer = torch.optim.Adam(
        model.parameters(), 
        lr=1e-4,  # 더 작은 학습률
        betas=(0.9, 0.999),
        eps=1e-8
    )
    
    print(f"데이터셋 크기: {len(dataset):,}")
    print(f"모델 파라미터 수: {sum(p.numel() for p in model.parameters()):,}")
    
    # 훈련 루프
    model.train()
    best_psnr = 0.0
    
    for epoch in range(5):  # 짧은 테스트
        epoch_loss = 0.0
        epoch_psnr = 0.0
        num_batches = 0
        
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/5")
        
        for batch_idx, batch in enumerate(pbar):
            if batch_idx >= 20:  # 20개 배치만 테스트
                break
                
            images = batch['image'].to(device)
            
            optimizer.zero_grad()
            
            try:
                # Forward pass
                results = model(images)
                reconstructed = results['reconstructed']
                mu = results['mu']
                logvar = results['logvar']
                
                # 안정적인 손실 계산
                recon_loss = F.mse_loss(reconstructed, images)
                
                # KL divergence with numerical stability
                kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
                kl_loss = torch.clamp(kl_loss, min=0, max=10)  # 클램핑으로 안정성 확보
                
                total_loss = recon_loss + 0.0001 * kl_loss  # 더 작은 KL 가중치
                
                # NaN 체크
                if torch.isnan(total_loss) or torch.isinf(total_loss):
                    print(f"NaN 감지! 건너뛰기...")
                    continue
                
                # Backward pass
                total_loss.backward()
                
                # 그래디언트 클리핑
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
                
                optimizer.step()
                
                # PSNR 계산
                with torch.no_grad():
                    mse = F.mse_loss(reconstructed, images)
                    if mse > 0:
                        psnr = 20 * torch.log10(1.0 / torch.sqrt(mse))
                        if not torch.isnan(psnr) and not torch.isinf(psnr):
                            epoch_psnr += psnr.item()
                
                epoch_loss += total_loss.item()
                num_batches += 1
                
                pbar.set_postfix({
                    'Loss': f'{total_loss.item():.4f}',
                    'Recon': f'{recon_loss.item():.4f}',
                    'KL': f'{kl_loss.item():.6f}',
                    'GradNorm': f'{grad_norm:.3f}'
                })
                
            except Exception as e:
                print(f"배치 {batch_idx}에서 오류: {e}")
                continue
        
        if num_batches > 0:
            avg_loss = epoch_loss / num_batches
            avg_psnr = epoch_psnr / num_batches if epoch_psnr > 0 else 0
            
            print(f"Epoch {epoch+1} 완료:")
            print(f"  평균 손실: {avg_loss:.4f}")
            print(f"  평균 PSNR: {avg_psnr:.2f} dB")
            
            if torch.cuda.is_available():
                memory_used = torch.cuda.memory_allocated() / 1024**3
                print(f"  GPU 메모리: {memory_used:.1f}GB")
                torch.cuda.empty_cache()
            
            if avg_psnr > best_psnr:
                best_psnr = avg_psnr
                print(f"  새로운 최고 PSNR: {avg_psnr:.2f} dB")
    
    print(f"\n훈련 완료! 최고 PSNR: {best_psnr:.2f} dB")
    
    # 최종 테스트
    model.eval()
    with torch.no_grad():
        test_batch = next(iter(dataloader))
        test_images = test_batch['image'][:2].to(device)
        
        results = model(test_images)
        reconstructed = results['reconstructed']
        
        print(f"\n최종 결과:")
        print(f"입력 형태: {test_images.shape}")
        print(f"출력 형태: {reconstructed.shape}")
        
        # 안정적인 PSNR 계산
        mse = F.mse_loss(reconstructed, test_images)
        if mse > 0 and not torch.isnan(mse):
            psnr = 20 * torch.log10(1.0 / torch.sqrt(mse))
            print(f"최종 PSNR: {psnr:.2f} dB")
        else:
            print("PSNR 계산 불가 (MSE가 0이거나 NaN)")


if __name__ == "__main__":
    stable_rtx3060_train()
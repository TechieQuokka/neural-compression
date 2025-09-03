#!/usr/bin/env python3
"""RTX 3060 12GB 최적화 훈련 스크립트"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent / "src"))

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
import os
from tqdm import tqdm
import matplotlib.pyplot as plt

from neural_compression.models.vae_gan_compression import VAEGANCompression
from neural_compression.data.datasets import CIFARDataset
from neural_compression.data.transforms import CompressionTransforms


def rtx3060_optimized_train():
    """RTX 3060 12GB 최적화 훈련"""
    
    print("RTX 3060 12GB 최적화 훈련 시작")
    
    # GPU 설정 및 최적화
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"사용 디바이스: {device}")
    
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name()}")
        print(f"GPU 메모리: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
        
        # RTX 3060 최적화 설정
        torch.backends.cudnn.benchmark = True  # 성능 최적화
        torch.backends.cuda.matmul.allow_tf32 = True  # Tensor Core 활용
        torch.backends.cudnn.allow_tf32 = True
    
    # 모델 생성 (RTX 3060 최적화)
    model = VAEGANCompression(
        latent_dim=128,  # 메모리 절약
        use_discriminator=False,
        quantization_method=None,
        base_channels=64,  # 메모리 절약
        compression_ratio=16
    ).to(device)
    
    # Mixed precision 설정
    scaler = torch.cuda.amp.GradScaler()
    
    # 데이터 생성 (RTX 3060 최적화)
    transforms = CompressionTransforms(image_size=(256, 256), augment=True)
    dataset = CIFARDataset(
        cifar_type="cifar10",
        train=True,
        transforms=transforms.get_train_transforms()
    )
    
    # RTX 3060에 최적화된 데이터로더 설정
    dataloader = DataLoader(
        dataset, 
        batch_size=6,  # 12GB GPU에 적합
        shuffle=True, 
        num_workers=4,  # 멀티프로세싱
        pin_memory=True,  # GPU 전송 최적화
        persistent_workers=True  # 워커 재사용
    )
    
    # 옵티마이저 (RTX 3060 최적화)
    optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr=2e-4,  # RTX 3060에 적합한 학습률
        weight_decay=1e-4,
        betas=(0.9, 0.999)
    )
    
    # 학습률 스케줄러
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=50, eta_min=1e-6
    )
    
    print(f"데이터셋 크기: {len(dataset):,}")
    print(f"모델 파라미터 수: {sum(p.numel() for p in model.parameters()):,}")
    print(f"배치 크기: {dataloader.batch_size}")
    print(f"배치당 이미지 수: {dataloader.batch_size}")
    
    # 훈련 루프
    model.train()
    best_psnr = 0.0
    train_losses = []
    psnr_history = []
    
    epochs = 20  # RTX 3060에 적합한 에폭 수
    
    for epoch in range(epochs):
        epoch_loss = 0.0
        epoch_psnr = 0.0
        num_batches = 0
        
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}")
        
        for batch_idx, batch in enumerate(pbar):
            if batch_idx >= 100:  # 메모리 절약을 위해 제한
                break
                
            images = batch['image'].to(device, non_blocking=True)
            
            optimizer.zero_grad()
            
            # Mixed precision forward pass
            with torch.cuda.amp.autocast():
                results = model(images)
                reconstructed = results['reconstructed']
                mu = results['mu']
                logvar = results['logvar']
                
                # 손실 계산
                recon_loss = F.mse_loss(reconstructed, images)
                kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / images.numel()
                
                total_loss = recon_loss + 0.001 * kl_loss
            
            # Mixed precision backward pass
            scaler.scale(total_loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            
            # PSNR 계산
            with torch.no_grad():
                mse = F.mse_loss(reconstructed, images)
                psnr = 20 * torch.log10(1.0 / torch.sqrt(mse))
                epoch_psnr += psnr.item()
            
            epoch_loss += total_loss.item()
            num_batches += 1
            
            pbar.set_postfix({
                'Loss': f'{total_loss.item():.4f}',
                'PSNR': f'{psnr.item():.2f}',
                'LR': f'{optimizer.param_groups[0]["lr"]:.6f}'
            })
            
            # 메모리 정리
            if batch_idx % 20 == 0:
                torch.cuda.empty_cache()
        
        scheduler.step()
        
        avg_loss = epoch_loss / num_batches
        avg_psnr = epoch_psnr / num_batches
        
        train_losses.append(avg_loss)
        psnr_history.append(avg_psnr)
        
        print(f"Epoch {epoch+1} 완료:")
        print(f"  평균 손실: {avg_loss:.4f}")
        print(f"  평균 PSNR: {avg_psnr:.2f} dB")
        print(f"  학습률: {optimizer.param_groups[0]['lr']:.6f}")
        
        # GPU 메모리 사용량 출력
        if torch.cuda.is_available():
            memory_used = torch.cuda.memory_allocated() / 1024**3
            memory_cached = torch.cuda.memory_reserved() / 1024**3
            print(f"  GPU 메모리: {memory_used:.1f}GB 사용 / {memory_cached:.1f}GB 캐시")
        
        # 최고 성능 모델 저장
        if avg_psnr > best_psnr:
            best_psnr = avg_psnr
            checkpoint_dir = Path("checkpoints/rtx3060")
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss,
                'psnr': avg_psnr
            }, checkpoint_dir / f"best_model_epoch_{epoch}.pth")
            print(f"  새로운 최고 PSNR! 모델 저장됨: {avg_psnr:.2f} dB")
    
    print(f"\n훈련 완료!")
    print(f"최고 PSNR: {best_psnr:.2f} dB")
    
    # 훈련 결과 시각화
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(train_losses)
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    plt.plot(psnr_history)
    plt.title('PSNR Progress')
    plt.xlabel('Epoch')
    plt.ylabel('PSNR (dB)')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('rtx3060_training_results.png', dpi=150, bbox_inches='tight')
    print("훈련 결과 그래프 저장: rtx3060_training_results.png")
    
    # 최종 테스트
    model.eval()
    with torch.no_grad():
        test_batch = next(iter(dataloader))
        test_images = test_batch['image'][:2].to(device)
        
        with torch.cuda.amp.autocast():
            results = model(test_images)
            reconstructed = results['reconstructed']
        
        # 최종 PSNR 계산
        mse = F.mse_loss(reconstructed, test_images)
        final_psnr = 20 * torch.log10(1.0 / torch.sqrt(mse))
        
        print(f"\n최종 테스트 결과:")
        print(f"입력 형태: {test_images.shape}")
        print(f"출력 형태: {reconstructed.shape}")
        print(f"최종 PSNR: {final_psnr:.2f} dB")
        
        # 압축률 계산
        original_size = test_images.numel() * 32  # 32-bit float
        compressed_size = results['latent'].numel() * 8  # 8-bit quantized estimate
        compression_ratio = original_size / compressed_size
        print(f"추정 압축률: {compression_ratio:.1f}:1")


if __name__ == "__main__":
    rtx3060_optimized_train()
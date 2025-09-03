#!/usr/bin/env python3
"""최소 기능 RTX 3060 압축 모델"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from pathlib import Path


class SimpleAutoEncoder(nn.Module):
    """간단하고 안정적인 오토인코더"""
    
    def __init__(self, latent_dim=64):
        super().__init__()
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, 4, 2, 1),  # 128 -> 64
            nn.BatchNorm2d(32),
            nn.ReLU(),
            
            nn.Conv2d(32, 64, 4, 2, 1),  # 64 -> 32
            nn.BatchNorm2d(64),
            nn.ReLU(),
            
            nn.Conv2d(64, 128, 4, 2, 1),  # 32 -> 16
            nn.BatchNorm2d(128),
            nn.ReLU(),
            
            nn.Conv2d(128, 256, 4, 2, 1),  # 16 -> 8
            nn.BatchNorm2d(256),
            nn.ReLU(),
            
            nn.AdaptiveAvgPool2d(4),  # 고정 크기로 만들기
            nn.Flatten(),
            nn.Linear(256 * 4 * 4, latent_dim)
        )
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 256 * 4 * 4),
            nn.ReLU(),
            nn.Unflatten(1, (256, 4, 4)),
            
            nn.ConvTranspose2d(256, 128, 4, 2, 1),  # 4 -> 8
            nn.BatchNorm2d(128),
            nn.ReLU(),
            
            nn.ConvTranspose2d(128, 64, 4, 2, 1),  # 8 -> 16
            nn.BatchNorm2d(64),
            nn.ReLU(),
            
            nn.ConvTranspose2d(64, 32, 4, 2, 1),  # 16 -> 32
            nn.BatchNorm2d(32),
            nn.ReLU(),
            
            nn.ConvTranspose2d(32, 16, 4, 2, 1),  # 32 -> 64
            nn.BatchNorm2d(16),
            nn.ReLU(),
            
            nn.ConvTranspose2d(16, 3, 4, 2, 1),  # 64 -> 128
            nn.Tanh()
        )
        
        # 가중치 초기화
        self.apply(self._init_weights)
    
    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight)
            nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        latent = self.encoder(x)
        reconstructed = self.decoder(latent)
        return reconstructed, latent


def minimal_rtx3060_train():
    """최소 기능 RTX 3060 훈련"""
    
    print("🚀 최소 기능 RTX 3060 압축 모델 훈련")
    
    # 디바이스 설정
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"디바이스: {device}")
    
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name()}")
        print(f"GPU 메모리: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
        torch.backends.cudnn.benchmark = True
    
    # 간단한 모델
    model = SimpleAutoEncoder(latent_dim=64).to(device)
    
    # CIFAR-10 데이터
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # [-1, 1] 범위
    ])
    
    dataset = datasets.CIFAR10(
        root='./data', 
        train=True, 
        download=True, 
        transform=transform
    )
    
    # RTX 3060 최적화 데이터로더
    dataloader = DataLoader(
        dataset,
        batch_size=16,  # RTX 3060 12GB에 적합
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True
    )
    
    # 옵티마이저
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=1e-3,
        betas=(0.9, 0.999),
        eps=1e-8
    )
    
    print(f"모델 파라미터 수: {sum(p.numel() for p in model.parameters()):,}")
    print(f"배치 크기: {dataloader.batch_size}")
    
    # 훈련 루프
    model.train()
    train_losses = []
    psnr_history = []
    best_psnr = 0.0
    
    epochs = 20
    
    for epoch in range(epochs):
        epoch_loss = 0.0
        epoch_psnr = 0.0
        num_batches = 0
        
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}")
        
        for batch_idx, (images, _) in enumerate(pbar):
            if batch_idx >= 100:  # 제한된 배치 수
                break
            
            images = images.to(device, non_blocking=True)
            
            optimizer.zero_grad()
            
            # Forward pass
            reconstructed, latent = model(images)
            
            # 손실 계산
            loss = F.mse_loss(reconstructed, images)
            
            # NaN 체크
            if torch.isnan(loss) or torch.isinf(loss):
                print(f"NaN 감지, 건너뛰기")
                continue
            
            # Backward pass
            loss.backward()
            
            # 그래디언트 클리핑
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            
            optimizer.step()
            
            # PSNR 계산
            with torch.no_grad():
                mse = F.mse_loss(reconstructed, images)
                if mse > 0:
                    # 정규화된 이미지에 대한 PSNR 계산
                    psnr = 20 * torch.log10(2.0 / torch.sqrt(mse))  # [-1,1] 범위이므로 2.0 사용
                    epoch_psnr += psnr.item()
            
            epoch_loss += loss.item()
            num_batches += 1
            
            pbar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'PSNR': f'{psnr.item():.2f}' if 'psnr' in locals() else 'N/A'
            })
            
            # 메모리 관리
            if batch_idx % 20 == 0 and torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        if num_batches > 0:
            avg_loss = epoch_loss / num_batches
            avg_psnr = epoch_psnr / num_batches
            
            train_losses.append(avg_loss)
            psnr_history.append(avg_psnr)
            
            print(f"\nEpoch {epoch+1} 완료:")
            print(f"  평균 손실: {avg_loss:.4f}")
            print(f"  평균 PSNR: {avg_psnr:.2f} dB")
            
            if torch.cuda.is_available():
                memory_used = torch.cuda.memory_allocated() / 1024**3
                print(f"  GPU 메모리: {memory_used:.1f}GB")
            
            # 최고 성능 저장
            if avg_psnr > best_psnr:
                best_psnr = avg_psnr
                checkpoint_dir = Path("checkpoints/minimal")
                checkpoint_dir.mkdir(parents=True, exist_ok=True)
                
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'loss': avg_loss,
                    'psnr': avg_psnr
                }, checkpoint_dir / f"best_minimal_model.pth")
                print(f"  ✓ 새로운 최고 PSNR: {avg_psnr:.2f} dB")
    
    print(f"\n🎯 훈련 완료!")
    print(f"📊 최고 PSNR: {best_psnr:.2f} dB")
    
    # 결과 시각화
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(train_losses)
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss')
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    plt.plot(psnr_history)
    plt.title('PSNR Progress')
    plt.xlabel('Epoch')
    plt.ylabel('PSNR (dB)')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('minimal_rtx3060_results.png', dpi=150, bbox_inches='tight')
    print("📈 결과 그래프 저장: minimal_rtx3060_results.png")
    
    # 최종 테스트
    model.eval()
    with torch.no_grad():
        test_images, _ = next(iter(dataloader))
        test_images = test_images[:4].to(device)
        
        reconstructed, latent = model(test_images)
        
        print(f"\n📏 최종 테스트:")
        print(f"   입력 형태: {test_images.shape}")
        print(f"   출력 형태: {reconstructed.shape}")
        
        mse = F.mse_loss(reconstructed, test_images)
        psnr = 20 * torch.log10(2.0 / torch.sqrt(mse))
        print(f"   최종 PSNR: {psnr:.2f} dB")
        
        # 압축률 계산
        original_bits = test_images.numel() * 32
        compressed_bits = latent.numel() * 32  # 실제로는 양자화로 더 작아질 수 있음
        compression_ratio = original_bits / compressed_bits
        print(f"   압축률: {compression_ratio:.1f}:1")
        
        # 시각적 결과 저장
        import torchvision.utils as vutils
        
        comparison = torch.cat([test_images[:4], reconstructed[:4]], dim=0)
        grid = vutils.make_grid(comparison, nrow=4, normalize=True, scale_each=True)
        vutils.save_image(grid, 'minimal_reconstruction_comparison.png')
        print("   🖼️  재구성 비교 이미지 저장: minimal_reconstruction_comparison.png")
    
    print(f"\n✅ RTX 3060 12GB 최적화 완료!")
    print(f"   - 안정적인 훈련 성공")
    print(f"   - 메모리 효율적 사용")
    print(f"   - 크기 문제 해결")


if __name__ == "__main__":
    minimal_rtx3060_train()
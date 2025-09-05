#!/usr/bin/env python3
"""훈련된 모델로 이미지 압축/복원 검증"""

import torch
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# 모델 import
import sys
sys.path.append('.')
from scripts.minimal_rtx3060 import SimpleAutoEncoder

def test_compression():
    """압축/복원 검증"""
    print("🔍 훈련된 모델로 압축/복원 검증 시작")
    
    # 디바이스 설정
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"디바이스: {device}")
    
    # 모델 로드
    model = SimpleAutoEncoder(latent_dim=64).to(device)
    
    checkpoint_path = "checkpoints/minimal/best_minimal_model.pth"
    if Path(checkpoint_path).exists():
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"✅ 모델 로드 완료: PSNR {checkpoint['psnr']:.2f} dB")
    else:
        print("❌ 체크포인트 파일을 찾을 수 없습니다!")
        return
    
    # 테스트 데이터 준비
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    test_dataset = datasets.CIFAR10(
        root='./data', 
        train=False, 
        download=True, 
        transform=transform
    )
    
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)
    
    # 모델 평가 모드
    model.eval()
    
    # 테스트 실행
    total_psnr = 0
    total_mse = 0
    num_batches = 0
    
    print("\n📊 압축/복원 테스트 진행 중...")
    
    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(test_loader):
            if batch_idx >= 10:  # 10개 배치만 테스트
                break
                
            images = images.to(device)
            
            # 압축 (인코딩)
            latent = model.encoder(images)
            
            # 복원 (디코딩)  
            reconstructed = model.decoder(latent)
            
            # 성능 측정
            mse = F.mse_loss(reconstructed, images)
            psnr = 20 * torch.log10(2.0 / torch.sqrt(mse))
            
            total_psnr += psnr.item()
            total_mse += mse.item()
            num_batches += 1
            
            # 첫 번째 배치 시각화
            if batch_idx == 0:
                save_comparison(images[:8], reconstructed[:8], labels[:8])
    
    # 결과 출력
    avg_psnr = total_psnr / num_batches
    avg_mse = total_mse / num_batches
    
    print(f"\n🎯 압축/복원 검증 결과:")
    print(f"   평균 PSNR: {avg_psnr:.2f} dB")
    print(f"   평균 MSE: {avg_mse:.6f}")
    print(f"   압축률: 768:1")
    
    # 압축 효율성 계산
    original_size = 128 * 128 * 3 * 32  # 32bit float
    compressed_size = 64 * 32  # latent dimension * 32bit
    compression_ratio = original_size / compressed_size
    
    print(f"   실제 압축률: {compression_ratio:.1f}:1")
    print(f"   원본 크기: {original_size/1024:.1f} KB")
    print(f"   압축 크기: {compressed_size/1024:.1f} KB")

def save_comparison(originals, reconstructed, labels):
    """원본과 재구성 이미지 비교 저장"""
    
    # CIFAR-10 클래스명
    classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 
               'dog', 'frog', 'horse', 'ship', 'truck']
    
    # 정규화 해제 ([-1,1] -> [0,1])
    originals = (originals + 1) / 2
    reconstructed = (reconstructed + 1) / 2
    
    # 클램핑
    originals = torch.clamp(originals, 0, 1)
    reconstructed = torch.clamp(reconstructed, 0, 1)
    
    # 시각화
    fig, axes = plt.subplots(2, 8, figsize=(16, 4))
    fig.suptitle('압축/복원 검증 결과 (위: 원본, 아래: 복원)', fontsize=14)
    
    for i in range(8):
        # 원본 이미지
        orig_img = originals[i].cpu().permute(1, 2, 0).numpy()
        axes[0, i].imshow(orig_img)
        axes[0, i].set_title(f'{classes[labels[i]]}', fontsize=10)
        axes[0, i].axis('off')
        
        # 복원 이미지
        recon_img = reconstructed[i].cpu().permute(1, 2, 0).numpy()
        axes[1, i].imshow(recon_img)
        
        # PSNR 계산
        mse = F.mse_loss(reconstructed[i], originals[i])
        psnr = 20 * torch.log10(1.0 / torch.sqrt(mse))
        axes[1, i].set_title(f'PSNR: {psnr:.1f}dB', fontsize=10)
        axes[1, i].axis('off')
    
    plt.tight_layout()
    plt.savefig('compression_test_results.png', dpi=150, bbox_inches='tight')
    print("🖼️  비교 이미지 저장: compression_test_results.png")

if __name__ == "__main__":
    test_compression()
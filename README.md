# 🚀 Neural Compression

**RTX 3060 최적화된 고급 딥러닝 압축 모델** - VAE-GAN 하이브리드 아키텍처

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![CUDA](https://img.shields.io/badge/CUDA-11.8+-green.svg)](https://developer.nvidia.com/cuda-downloads)

## ✨ 주요 특징

- **🧠 하이브리드 아키텍처**: VAE + GAN + Multi-Head Attention 결합
- **⚡ RTX 3060 최적화**: 12GB GPU 메모리 효율적 활용
- **🎯 고급 양자화**: Vector Quantization + Learned Quantization
- **📊 손실 함수 최적화**: Perceptual + Rate-Distortion + MS-SSIM Loss
- **🔧 Mixed Precision**: FP16 훈련으로 메모리 및 속도 최적화
- **📈 실시간 모니터링**: TensorBoard + WandB 지원

## 🛠️ 설치

### 전제 조건
- Python 3.8+
- CUDA 11.8+ (RTX 3060 최적화)
- 12GB+ GPU 메모리

```bash
# 저장소 클론
git clone https://github.com/TechieQuokka/neural-compression.git
cd neural-compression

# 가상환경 생성
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate  # Windows

# 의존성 설치
pip install -r requirements.txt
pip install -e .
```

## 🚀 빠른 시작

### RTX 3060 최적화 훈련
```bash
# RTX 3060 12GB 최적화된 안정적인 훈련
python scripts/minimal_rtx3060.py

# 고급 설정으로 훈련 (Hydra 기반)
python scripts/train.py --config-name=rtx3060_config
```

### 모델 사용
```python
from neural_compression.models.vae_gan_compression import VAEGANCompression
import torch

# RTX 3060 최적화 모델 생성
model = VAEGANCompression(
    latent_dim=64,
    use_discriminator=False,
    base_channels=64,
    target_size=128
)

# 이미지 압축 및 복원
with torch.no_grad():
    results = model(images)
    compressed = results['latent']
    reconstructed = results['reconstructed']
```

## 📊 성능 결과

### RTX 3060 12GB 최적화 결과
- **PSNR**: 19.09 dB
- **압축률**: 768:1
- **GPU 메모리 사용량**: 0.1GB / 12GB
- **훈련 속도**: ~30 it/s (배치 크기 16)

## 📁 프로젝트 구조

```
neural-compression/
├── src/neural_compression/
│   ├── models/              # 🧠 신경망 모델
│   │   ├── vae_gan_compression.py    # 메인 VAE-GAN 모델
│   │   ├── encoder.py               # ResNet 기반 인코더
│   │   ├── decoder.py               # 트랜스포즈 컨볼루션 디코더
│   │   ├── attention.py             # 어텐션 메커니즘
│   │   ├── quantization.py          # 양자화 방법들
│   │   └── discriminator.py         # PatchGAN 판별기
│   ├── losses/              # 📊 손실 함수
│   │   ├── combined.py              # 결합된 손실 함수
│   │   ├── reconstruction.py        # 재구성 손실
│   │   ├── compression.py           # 압축 특화 손실
│   │   └── adversarial.py           # 적대적 손실
│   ├── training/            # 🎯 훈련 로직
│   │   └── lightning_module.py      # PyTorch Lightning 모듈
│   └── utils/               # 🛠️ 유틸리티
├── configs/                 # ⚙️ Hydra 설정
│   ├── rtx3060_config.yaml          # RTX 3060 최적화 설정
│   └── config.yaml                  # 기본 설정
├── scripts/                 # 📝 실행 스크립트
│   ├── minimal_rtx3060.py           # ✅ 안정적인 RTX 3060 훈련
│   ├── train.py                     # Hydra 기반 훈련
│   └── prepare_data.py              # 데이터 준비
├── docs/                    # 📚 문서
│   └── architecture.md              # 아키텍처 설계 문서
└── *.png                    # 📈 훈련 결과 및 비교 이미지
```

## 🏗️ 아키텍처 개요

### VAE-GAN 하이브리드 모델
```
Input Image (128×128) 
    ↓
🔄 ResNet Encoder + CBAM Attention
    ↓
📦 Vector Quantization (64D latent)
    ↓
🔄 Transpose Conv Decoder + Self-Attention
    ↓
Output Image (128×128)
```

### 핵심 구성 요소
1. **인코더**: ResNet 백본 + CBAM 어텐션
2. **양자화**: 벡터 양자화로 압축률 향상
3. **디코더**: 6단계 업샘플링 + 셀프 어텐션
4. **손실 함수**: MSE + KL Divergence + Perceptual Loss

## 🎯 RTX 3060 특화 최적화

### 메모리 최적화
- **Mixed Precision (FP16)**: 메모리 사용량 50% 절감
- **Gradient Checkpointing**: 메모리 효율적 백프롭
- **배치 크기 최적화**: 16 (12GB GPU 최적)

### 성능 최적화
- **CUDA 최적화**: `torch.backends.cudnn.benchmark = True`
- **TensorFloat-32**: `allow_tf32 = True`
- **Pin Memory**: 빠른 GPU 전송
- **Persistent Workers**: 멀티프로세싱 최적화

## 🔧 사용법

### 1️⃣ 빠른 훈련 (추천)
```bash
# RTX 3060 최적화된 안정적인 훈련
python scripts/minimal_rtx3060.py
```

### 2️⃣ 고급 설정 훈련
```bash
# Hydra 설정으로 커스터마이징
python scripts/train.py --config-name=rtx3060_config

# 배치 크기 조정
python scripts/train.py data.batch_size=8 training.max_epochs=50
```

### 3️⃣ 모델 로드 및 사용
```python
import torch
from neural_compression.models.vae_gan_compression import VAEGANCompression

# 훈련된 모델 로드
checkpoint = torch.load('checkpoints/minimal/best_minimal_model.pth')
model = VAEGANCompression(latent_dim=64, target_size=128)
model.load_state_dict(checkpoint['model_state_dict'])

# 압축 및 복원
model.eval()
with torch.no_grad():
    results = model(input_images)
    compressed_latent = results['latent']
    reconstructed_images = results['reconstructed']
```

## 📈 실험 결과

### 훈련 성과 (RTX 3060 12GB)
| 메트릭 | 값 |
|--------|-----|
| **최고 PSNR** | 19.09 dB |
| **압축률** | 768:1 |
| **GPU 메모리** | 0.1GB / 12GB |
| **훈련 속도** | 30+ it/s |
| **안정성** | ✅ NaN 없음 |

### 모델 사양
- **파라미터 수**: 1.9M (메모리 효율적)
- **Latent Dimension**: 64
- **입력 크기**: 128×128×3
- **압축 비율**: 16:1

## 🎨 결과 이미지

프로젝트 루트의 이미지 파일들:
- `minimal_reconstruction_comparison.png`: 원본 vs 재구성 비교
- `minimal_rtx3060_results.png`: 훈련 손실 및 PSNR 그래프

## ⚡ 성능 최적화 팁

### RTX 3060 사용자
```bash
# 최적 설정으로 훈련
export CUDA_VISIBLE_DEVICES=0
python scripts/minimal_rtx3060.py
```

### 메모리 부족 시
```yaml
# configs/rtx3060_config.yaml 수정
data:
  batch_size: 8      # 16 → 8로 축소
  image_size: [64, 64]  # 128 → 64로 축소
```

## 🏆 주요 성과

- ✅ **크기 문제 해결**: 입출력 크기 일치 보장
- ✅ **안정적인 훈련**: NaN/Inf 문제 해결
- ✅ **RTX 3060 최적화**: 12GB 메모리 효율적 활용
- ✅ **Mixed Precision**: FP16으로 성능 향상
- ✅ **실용적인 압축률**: 768:1 달성

## 📚 참고 문서

- [아키텍처 설계](docs/architecture.md): 상세한 모델 아키텍처 설명
- [Hydra 설정](configs/): 훈련 설정 파일들

## 🤝 기여

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 라이센스

MIT License - 자세한 내용은 [LICENSE](LICENSE) 파일 참조
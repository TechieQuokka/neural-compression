# Neural Compression

고급 딥러닝 기법을 활용한 데이터 압축 및 복원 시스템

## 특징

- **하이브리드 아키텍처**: VAE + GAN + Attention 메커니즘 결합
- **고급 양자화**: Learned Vector Quantization
- **손실 함수 최적화**: Perceptual Loss, Rate-Distortion Loss 
- **실시간 처리**: GPU 가속 압축/복원
- **확장 가능**: 다양한 데이터 타입 지원

## 설치

```bash
# 가상환경 생성
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate  # Windows

# 의존성 설치
pip install -r requirements.txt
pip install -e .
```

## 빠른 시작

```python
from neural_compression import VAEGANCompression
import torch

# 모델 로드
model = VAEGANCompression.load_from_checkpoint("checkpoints/best.ckpt")

# 압축
compressed = model.compress(image)

# 복원  
reconstructed = model.decompress(compressed)
```

## 훈련

```bash
# 기본 훈련
python scripts/train.py

# 설정 커스터마이징
python scripts/train.py model=vae_gan_compression data=imagenet training.max_epochs=200
```

## 프로젝트 구조

```
src/neural_compression/
├── models/          # 신경망 모델
├── data/           # 데이터 처리
├── training/       # 훈련 로직
├── losses/         # 손실 함수
└── utils/          # 유틸리티

configs/            # Hydra 설정 파일
experiments/        # 실험 결과
notebooks/          # Jupyter 노트북
tests/             # 테스트 코드
```

## 라이센스

MIT License
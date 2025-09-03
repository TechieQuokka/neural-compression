#!/usr/bin/env python3
"""
데이터 준비 스크립트
Flower Photos 데이터셋을 압축 모델 훈련용으로 준비
"""

import os
import shutil
from pathlib import Path
import random
from typing import List
import argparse


def split_dataset(
    source_dir: str,
    output_dir: str,
    train_ratio: float = 0.7,
    val_ratio: float = 0.2,
    test_ratio: float = 0.1,
    seed: int = 42
):
    """데이터셋을 train/val/test로 분할"""
    
    random.seed(seed)
    source_path = Path(source_dir)
    output_path = Path(output_dir)
    
    # 출력 디렉토리 생성
    train_dir = output_path / "train"
    val_dir = output_path / "val" 
    test_dir = output_path / "test"
    
    for dir_path in [train_dir, val_dir, test_dir]:
        dir_path.mkdir(parents=True, exist_ok=True)
    
    # 모든 이미지 파일 수집
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
    all_images = []
    
    for ext in image_extensions:
        all_images.extend(list(source_path.rglob(f"*{ext}")))
        all_images.extend(list(source_path.rglob(f"*{ext.upper()}")))
    
    print(f"총 {len(all_images)}개의 이미지를 발견했습니다.")
    
    # 이미지 셔플
    random.shuffle(all_images)
    
    # 분할 인덱스 계산
    total_images = len(all_images)
    train_end = int(total_images * train_ratio)
    val_end = train_end + int(total_images * val_ratio)
    
    train_images = all_images[:train_end]
    val_images = all_images[train_end:val_end]
    test_images = all_images[val_end:]
    
    print(f"분할 결과:")
    print(f"  Train: {len(train_images)}개")
    print(f"  Val: {len(val_images)}개") 
    print(f"  Test: {len(test_images)}개")
    
    # 파일 복사
    def copy_images(images: List[Path], dest_dir: Path):
        for i, img_path in enumerate(images):
            dest_path = dest_dir / f"image_{i:06d}{img_path.suffix}"
            shutil.copy2(img_path, dest_path)
            if (i + 1) % 500 == 0:
                print(f"  {i + 1}/{len(images)} 복사 완료")
    
    print("Train 이미지 복사 중...")
    copy_images(train_images, train_dir)
    
    print("Validation 이미지 복사 중...")
    copy_images(val_images, val_dir)
    
    print("Test 이미지 복사 중...")
    copy_images(test_images, test_dir)
    
    print("데이터 준비 완료!")


def download_sample_datasets():
    """샘플 데이터셋 다운로드"""
    import urllib.request
    import tarfile
    import zipfile
    
    datasets = {
        "flowers": {
            "url": "https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz",
            "filename": "flower_photos.tgz"
        },
        "places": {
            "url": "http://data.csail.mit.edu/places/places365/places365standard_easyformat.tar",
            "filename": "places365_small.tar"
        }
    }
    
    data_dir = Path("data")
    data_dir.mkdir(exist_ok=True)
    
    for name, info in datasets.items():
        print(f"{name} 데이터셋 다운로드 중...")
        filepath = data_dir / info["filename"]
        
        if not filepath.exists():
            try:
                urllib.request.urlretrieve(info["url"], filepath)
                print(f"다운로드 완료: {filepath}")
                
                # 압축 해제
                if filepath.suffix == ".tgz":
                    with tarfile.open(filepath, "r:gz") as tar:
                        tar.extractall(data_dir)
                elif filepath.suffix == ".zip":
                    with zipfile.ZipFile(filepath, "r") as zip_ref:
                        zip_ref.extractall(data_dir)
                        
                print(f"압축 해제 완료: {name}")
                
            except Exception as e:
                print(f"다운로드 실패 {name}: {e}")
        else:
            print(f"이미 존재함: {filepath}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="데이터 준비")
    parser.add_argument("--source", default="data/flower_photos", help="소스 디렉토리")
    parser.add_argument("--output", default="data/processed", help="출력 디렉토리")
    parser.add_argument("--download", action="store_true", help="샘플 데이터셋 다운로드")
    
    args = parser.parse_args()
    
    if args.download:
        download_sample_datasets()
    
    if Path(args.source).exists():
        split_dataset(args.source, args.output)
    else:
        print(f"소스 디렉토리가 존재하지 않습니다: {args.source}")
        print("--download 옵션을 사용하여 샘플 데이터를 다운로드하세요.")
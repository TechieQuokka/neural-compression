import torch
import json
from pathlib import Path
from typing import Dict, Any, Optional


def save_compression_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    metrics: Dict[str, float],
    save_path: str,
    metadata: Optional[Dict[str, Any]] = None
) -> None:
    """압축 모델 체크포인트 저장"""
    
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'metrics': metrics,
        'metadata': metadata or {}
    }
    
    # 체크포인트 저장
    torch.save(checkpoint, save_path)
    
    # 메타데이터 JSON 저장
    metadata_path = Path(save_path).with_suffix('.json')
    with open(metadata_path, 'w') as f:
        json.dump({
            'epoch': epoch,
            'metrics': metrics,
            'metadata': metadata or {}
        }, f, indent=2)


def load_compression_checkpoint(
    model: torch.nn.Module,
    checkpoint_path: str,
    optimizer: Optional[torch.optim.Optimizer] = None,
    device: str = 'cpu'
) -> Dict[str, Any]:
    """압축 모델 체크포인트 로드"""
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # 모델 가중치 로드
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # 옵티마이저 상태 로드
    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    return {
        'epoch': checkpoint.get('epoch', 0),
        'metrics': checkpoint.get('metrics', {}),
        'metadata': checkpoint.get('metadata', {})
    }
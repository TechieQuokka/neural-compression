import pytorch_lightning as pl
from typing import Optional, Dict, Any


class CompressionTrainer:
    """압축 모델 전용 트레이너 래퍼"""
    
    def __init__(
        self,
        model: pl.LightningModule,
        datamodule: pl.LightningDataModule,
        config: Dict[str, Any]
    ):
        self.model = model
        self.datamodule = datamodule
        self.config = config
        
        # PyTorch Lightning 트레이너 설정
        self.trainer = pl.Trainer(**config)
    
    def fit(self):
        """모델 훈련"""
        self.trainer.fit(self.model, self.datamodule)
    
    def test(self, ckpt_path: Optional[str] = "best"):
        """모델 테스트"""
        self.trainer.test(self.model, self.datamodule, ckpt_path=ckpt_path)
    
    def validate(self):
        """모델 검증"""
        self.trainer.validate(self.model, self.datamodule)
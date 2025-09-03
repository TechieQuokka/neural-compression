import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback
import torch
import wandb
from typing import Any


class CompressionCallbacks:
    """압축 모델용 콜백 컬렉션"""
    
    @staticmethod
    def get_default_callbacks():
        """기본 콜백들 반환"""
        return [
            ImageLoggingCallback(),
            MetricsCallback(),
            CompressionRatioCallback()
        ]


class ImageLoggingCallback(Callback):
    """이미지 로깅 콜백"""
    
    def __init__(self, log_every_n_epochs: int = 5):
        self.log_every_n_epochs = log_every_n_epochs
    
    def on_validation_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        if trainer.current_epoch % self.log_every_n_epochs == 0:
            self._log_reconstruction_images(trainer, pl_module)
    
    def _log_reconstruction_images(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        """재구성 이미지 로깅"""
        if not hasattr(trainer, 'datamodule'):
            return
        
        try:
            val_batch = next(iter(trainer.datamodule.val_dataloader()))
            images = val_batch['image'][:4].to(pl_module.device)
            
            with torch.no_grad():
                results = pl_module.model(images)
                reconstructed = results['reconstructed']
            
            # WandB 로깅
            for logger in trainer.loggers:
                if isinstance(logger, pl.loggers.WandbLogger):
                    import torchvision.utils as vutils
                    
                    comparison = torch.cat([images, reconstructed], dim=0)
                    grid = vutils.make_grid(comparison, nrow=4, normalize=True)
                    
                    logger.experiment.log({
                        "reconstructions": wandb.Image(grid),
                        "epoch": trainer.current_epoch
                    })
        except Exception as e:
            print(f"이미지 로깅 오류: {e}")


class MetricsCallback(Callback):
    """메트릭 추적 콜백"""
    
    def __init__(self):
        self.best_psnr = 0.0
        self.best_ssim = 0.0
    
    def on_validation_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        current_psnr = trainer.callback_metrics.get('val/psnr', 0)
        current_ssim = trainer.callback_metrics.get('val/ssim', 0)
        
        if current_psnr > self.best_psnr:
            self.best_psnr = current_psnr
            trainer.logger.log_metrics({'val/psnr_best': self.best_psnr})
        
        if current_ssim > self.best_ssim:
            self.best_ssim = current_ssim
            trainer.logger.log_metrics({'val/ssim_best': self.best_ssim})


class CompressionRatioCallback(Callback):
    """압축률 모니터링 콜백"""
    
    def on_validation_batch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        outputs: Any,
        batch: Any,
        batch_idx: int
    ) -> None:
        if batch_idx == 0:  # 첫 번째 배치에서만 계산
            images = batch['image']
            
            with torch.no_grad():
                results = pl_module.model(images)
                
                # 압축률 계산
                original_bits = images.numel() * 32  # 32-bit float
                compressed_bits = results['latent'].numel() * 8  # 8-bit quantized
                compression_ratio = original_bits / compressed_bits
                
                trainer.logger.log_metrics({
                    'val/compression_ratio': compression_ratio,
                    'val/bits_per_pixel': compressed_bits / (images.shape[0] * images.shape[2] * images.shape[3])
                })
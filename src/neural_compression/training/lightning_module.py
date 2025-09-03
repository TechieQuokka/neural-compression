import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from torch.optim import AdamW, Adam
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau
from typing import Dict, Any, Optional, List
import wandb

from ..models.vae_gan_compression import VAEGANCompression
from ..losses.combined import CombinedLoss


class CompressionLightningModule(pl.LightningModule):
    """PyTorch Lightning 압축 모듈"""
    
    def __init__(
        self,
        model_config: Dict[str, Any],
        optimizer_config: Dict[str, Any],
        scheduler_config: Dict[str, Any],
        loss_config: Dict[str, Any],
        **kwargs
    ):
        super().__init__()
        
        self.save_hyperparameters()
        
        # 모델 초기화
        self.model = VAEGANCompression(**model_config)
        
        # 손실 함수
        self.criterion = CombinedLoss(**loss_config)
        
        # 메트릭 저장
        self.automatic_optimization = True  # Automatic optimization
        
        # 학습률 스케줄러 설정
        self.optimizer_config = optimizer_config
        self.scheduler_config = scheduler_config
        
        # 검증 메트릭
        self.val_psnr_best = 0.0
        self.val_ssim_best = 0.0
        
    def configure_optimizers(self):
        """옵티마이저 및 스케줄러 설정"""
        
        # Generator parameters
        gen_params = list(self.model.encoder.parameters()) + list(self.model.decoder.parameters())
        if hasattr(self.model, 'quantizer') and self.model.quantizer is not None:
            gen_params += list(self.model.quantizer.parameters())
        
        # Generator optimizer
        if self.optimizer_config['name'] == 'adamw':
            gen_optimizer = AdamW(
                gen_params,
                lr=self.optimizer_config['lr'],
                weight_decay=self.optimizer_config.get('weight_decay', 1e-5),
                betas=self.optimizer_config.get('betas', [0.9, 0.999])
            )
        else:
            gen_optimizer = Adam(
                gen_params,
                lr=self.optimizer_config['lr'],
                betas=self.optimizer_config.get('betas', [0.9, 0.999])
            )
        
        optimizers = [gen_optimizer]
        schedulers = []
        
        # Generator scheduler
        if self.scheduler_config['name'] == 'cosine':
            gen_scheduler = CosineAnnealingLR(
                gen_optimizer,
                T_max=self.scheduler_config.get('T_max', 100),
                eta_min=self.scheduler_config.get('eta_min', 1e-6)
            )
            schedulers.append(gen_scheduler)
        elif self.scheduler_config['name'] == 'plateau':
            gen_scheduler = ReduceLROnPlateau(
                gen_optimizer,
                mode='min',
                patience=10,
                factor=0.5
            )
            schedulers.append(gen_scheduler)
        
        # Discriminator optimizer (if using GAN)
        if hasattr(self.model, 'discriminator'):
            disc_optimizer = Adam(
                self.model.discriminator.parameters(),
                lr=self.optimizer_config['lr'] * 0.5,  # 판별기는 더 느리게
                betas=[0.5, 0.999]
            )
            optimizers.append(disc_optimizer)
            
            if len(schedulers) > 0:
                disc_scheduler = CosineAnnealingLR(
                    disc_optimizer,
                    T_max=self.scheduler_config.get('T_max', 100),
                    eta_min=self.scheduler_config.get('eta_min', 1e-6)
                )
                schedulers.append(disc_scheduler)
        
        if len(schedulers) == 0:
            return optimizers
        else:
            return optimizers, schedulers
    
    def training_step(self, batch: Dict[str, torch.Tensor], batch_idx: int):
        """훈련 단계"""
        images = batch['image']
        
        # Forward pass
        results = self.model(images)
        
        # Calculate losses
        losses = self.criterion.generator_loss(
            reconstructed=results['reconstructed'],
            original=images,
            mu=results['mu'],
            logvar=results['logvar'],
            fake_output=results.get('fake_output'),
            latent=results['latent']
        )
        
        # 메트릭 로깅
        self.log('train/total_loss', losses['total'], prog_bar=True)
        self.log('train/recon_loss', losses['reconstruction'])
        self.log('train/kl_loss', losses['kl_divergence'])
        
        if 'perceptual' in losses:
            self.log('train/perceptual_loss', losses['perceptual'])
        
        # PSNR 계산
        with torch.no_grad():
            mse = F.mse_loss(results['reconstructed'], images)
            psnr = 20 * torch.log10(1.0 / torch.sqrt(mse))
            self.log('train/psnr', psnr)
        
        return losses['total']
    
    def validation_step(self, batch: Dict[str, torch.Tensor], batch_idx: int):
        """검증 단계"""
        images = batch['image']
        
        results = self.model(images)
        
        # 손실 계산
        val_losses = self.criterion.generator_loss(
            reconstructed=results['reconstructed'],
            original=images,
            mu=results['mu'],
            logvar=results['logvar'],
            latent=results['latent']
        )
        
        # 메트릭 계산
        with torch.no_grad():
            mse = F.mse_loss(results['reconstructed'], images)
            psnr = 20 * torch.log10(1.0 / torch.sqrt(mse))
            
            # SSIM 근사 계산
            ssim = self._calculate_ssim(results['reconstructed'], images)
            
            # 압축률 추정
            compression_ratio = self.model.calculate_compression_ratio(
                images.numel() * 32,  # 32-bit float
                results['latent'].numel() * 8  # 8-bit quantized
            )
        
        # 로깅
        self.log('val/total_loss', val_losses['total'], prog_bar=True)
        self.log('val/recon_loss', val_losses['reconstruction'])
        self.log('val/psnr', psnr, prog_bar=True)
        self.log('val/ssim', ssim)
        self.log('val/compression_ratio', compression_ratio)
        
        # Best 메트릭 업데이트
        if psnr > self.val_psnr_best:
            self.val_psnr_best = psnr
            self.log('val/psnr_best', self.val_psnr_best)
        
        if ssim > self.val_ssim_best:
            self.val_ssim_best = ssim
            self.log('val/ssim_best', self.val_ssim_best)
        
        return val_losses['total']
    
    def _calculate_ssim(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """SSIM 근사 계산"""
        C1 = 0.01 ** 2
        C2 = 0.03 ** 2
        
        mu_pred = torch.mean(pred, dim=[2, 3], keepdim=True)
        mu_target = torch.mean(target, dim=[2, 3], keepdim=True)
        
        sigma_pred = torch.var(pred, dim=[2, 3], keepdim=True)
        sigma_target = torch.var(target, dim=[2, 3], keepdim=True)
        sigma_cross = torch.mean((pred - mu_pred) * (target - mu_target), dim=[2, 3], keepdim=True)
        
        ssim = ((2 * mu_pred * mu_target + C1) * (2 * sigma_cross + C2)) / \
               ((mu_pred ** 2 + mu_target ** 2 + C1) * (sigma_pred + sigma_target + C2))
        
        return torch.mean(ssim)
    
    def test_step(self, batch: Dict[str, torch.Tensor], batch_idx: int):
        """테스트 단계"""
        return self.validation_step(batch, batch_idx)
    
    def on_validation_epoch_end(self):
        """검증 에폭 종료 시"""
        # 스케줄러 업데이트
        schedulers = self.lr_schedulers()
        if schedulers and isinstance(schedulers, list):
            for scheduler in schedulers:
                if isinstance(scheduler, ReduceLROnPlateau):
                    scheduler.step(self.trainer.callback_metrics.get('val/total_loss', 0))
        elif schedulers and isinstance(schedulers, ReduceLROnPlateau):
            schedulers.step(self.trainer.callback_metrics.get('val/total_loss', 0))
    
    def on_train_epoch_end(self):
        """훈련 에폭 종료 시"""
        # 이미지 로깅 (WandB)
        if self.current_epoch % 10 == 0 and self.logger:
            self._log_images()
    
    def _log_images(self):
        """이미지 로깅"""
        if hasattr(self.trainer, 'datamodule') and hasattr(self.trainer.datamodule, 'val_dataloader'):
            val_batch = next(iter(self.trainer.datamodule.val_dataloader()))
            images = val_batch['image'][:4].to(self.device)
            
            with torch.no_grad():
                results = self.model(images)
                reconstructed = results['reconstructed']
            
            # WandB 이미지 로깅
            if isinstance(self.logger, pl.loggers.WandbLogger):
                import torchvision.utils as vutils
                
                # 원본과 재구성 이미지 나란히 배치
                comparison = torch.cat([images, reconstructed], dim=0)
                grid = vutils.make_grid(comparison, nrow=4, normalize=True, scale_each=True)
                
                self.logger.experiment.log({
                    "reconstruction_comparison": wandb.Image(grid),
                    "epoch": self.current_epoch
                })
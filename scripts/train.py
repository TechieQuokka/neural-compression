#!/usr/bin/env python3
"""
Neural Compression 훈련 스크립트
"""

import hydra
from omegaconf import DictConfig, OmegaConf
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger, TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
import torch
import os
from pathlib import Path

# 프로젝트 모듈 import
import sys
sys.path.append(str(Path(__file__).parent.parent / "src"))

from neural_compression.training.lightning_module import CompressionLightningModule
from neural_compression.data.datamodules import CustomDataModule


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    """메인 훈련 함수"""
    
    print("=" * 50)
    print("Neural Compression Training")
    print("=" * 50)
    print(f"Configuration:\n{OmegaConf.to_yaml(cfg)}")
    
    # 시드 설정
    pl.seed_everything(42, workers=True)
    
    # 데이터 모듈
    print("데이터 모듈 초기화 중...")
    if os.path.exists("data/processed/train"):
        datamodule = CustomDataModule(
            train_dir="data/processed/train",
            val_dir="data/processed/val",
            test_dir="data/processed/test",
            **cfg.data
        )
    else:
        print("처리된 데이터가 없습니다. 먼저 scripts/prepare_data.py를 실행하세요.")
        return
    
    # 모델 모듈
    print("모델 초기화 중...")
    
    # 모델 설정 간소화 (판별기 없이)
    model_config = {
        'latent_dim': cfg.model.get('latent_dim', 256),
        'use_discriminator': False,  # 첫 훈련에서는 판별기 비활성화
        'quantization_method': cfg.model.quantization.get('method', 'vector'),
        'compression_ratio': cfg.model.get('compression_ratio', 16),
        'base_channels': 64,
        'num_stages': 4
    }
    
    model = CompressionLightningModule(
        model_config=model_config,
        optimizer_config=cfg.optimizer,
        scheduler_config=cfg.scheduler,
        loss_config={'loss_weights': cfg.get('loss_weights', {})}
    )
    
    # 콜백 설정
    callbacks = []
    
    # 체크포인트 콜백
    checkpoint_callback = ModelCheckpoint(
        dirpath=cfg.paths.checkpoint_dir,
        filename='{epoch:02d}-{val_total_loss:.4f}',
        monitor=cfg.logging.monitor,
        mode=cfg.logging.mode,
        save_top_k=cfg.logging.save_top_k,
        save_last=True,
        verbose=True
    )
    callbacks.append(checkpoint_callback)
    
    # 조기 종료
    if cfg.training.get('early_stopping'):
        early_stop_callback = EarlyStopping(
            monitor=cfg.training.early_stopping.monitor,
            patience=cfg.training.early_stopping.patience,
            mode=cfg.training.early_stopping.mode,
            min_delta=cfg.training.early_stopping.min_delta,
            verbose=True
        )
        callbacks.append(early_stop_callback)
    
    # 학습률 모니터링
    lr_monitor = LearningRateMonitor(logging_interval='step')
    callbacks.append(lr_monitor)
    
    # 로거 설정
    loggers = []
    
    # WandB 로거
    if cfg.get('wandb') and cfg.wandb.get('project'):
        wandb_logger = WandbLogger(
            project=cfg.wandb.project,
            entity=cfg.wandb.get('entity'),
            name=f"{cfg.experiment.name}_v{cfg.experiment.version}",
            save_dir=cfg.paths.log_dir
        )
        loggers.append(wandb_logger)
    
    # TensorBoard 로거
    tb_logger = TensorBoardLogger(
        save_dir=cfg.paths.log_dir,
        name=cfg.experiment.name,
        version=cfg.experiment.version
    )
    loggers.append(tb_logger)
    
    # 트레이너 설정
    print("트레이너 초기화 중...")
    trainer = pl.Trainer(
        max_epochs=cfg.training.max_epochs,
        accelerator=cfg.training.accelerator,
        devices=cfg.training.devices,
        precision=cfg.training.precision,
        gradient_clip_val=cfg.training.gradient_clip_val,
        accumulate_grad_batches=cfg.training.accumulate_grad_batches,
        val_check_interval=cfg.training.val_check_interval,
        log_every_n_steps=cfg.logging.log_every_n_steps,
        callbacks=callbacks,
        logger=loggers,
        deterministic=False,
        benchmark=True,
        enable_progress_bar=True,
        enable_model_summary=True
    )
    
    # 훈련 시작
    print("훈련 시작...")
    try:
        trainer.fit(model, datamodule=datamodule)
        
        # 최고 성능 모델로 테스트
        print("테스트 시작...")
        trainer.test(model, datamodule=datamodule, ckpt_path="best")
        
    except Exception as e:
        print(f"훈련 중 오류 발생: {e}")
        raise
    
    print("훈련 완료!")
    print(f"최고 PSNR: {model.val_psnr_best:.2f}")
    print(f"최고 SSIM: {model.val_ssim_best:.4f}")


if __name__ == "__main__":
    main()
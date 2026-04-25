#!/usr/bin/env bash
# Stage 1 NTP pre-training — single GPU (or CPU/MPS on Mac).
#
# Usage:
#   bash scripts/train_ntp_single_gpu.sh \
#       <DATA_ROOT> \
#       <VIDEOMAE_PT> \
#       [EXTRA_OVERRIDES...]
#
# Example (Mac CPU, small batch for testing):
#   bash scripts/train_ntp_single_gpu.sh \
#       data/AutoGaze-Training-Data/InternVid_res448_250K \
#       weights/VideoMAE_AutoGaze/videomae.pt
#
# Example (single GPU server):
#   bash scripts/train_ntp_single_gpu.sh \
#       /data/InternVid,...,/data/scanning_idl \
#       /weights/VideoMAE_AutoGaze/videomae.pt \
#       trainer.batch_size=256 trainer.per_gpu_max_batch_size=16

set -euo pipefail

DATA_ROOT="${1:?Usage: $0 <DATA_ROOT> <VIDEOMAE_PT> [overrides...]}"
VIDEOMAE_PT="${2:?Usage: $0 <DATA_ROOT> <VIDEOMAE_PT> [overrides...]}"
shift 2
EXTRA_OVERRIDES="$*"

python -m autogaze.train \
    --config-name video_folder_video_mae_reconstruction_ar_gaze_ntp \
    dataset.root="'${DATA_ROOT}'" \
    dataset.gt_gazing_pos_paths.train="'${DATA_ROOT%/*}/gazing_labels.json'" \
    dataset.clip_len=16 \
    model.gazing_ratio_config.sample_strategy_during_training=fixed \
    model.gazing_ratio_config.sample_strategy_during_inference=fixed \
    model.gazing_ratio_config.fixed.gazing_ratio=0.1 \
    model.gazing_ratio_each_frame_config.sample_strategy_during_inference=dirichlet \
    "model.gazing_ratio_each_frame_config.dirichlet.alpha='10,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3'" \
    model.scales=32+64+112+224 \
    model.num_vision_tokens_each_frame=265 \
    model.has_task_loss_requirement_during_training=False \
    model.has_task_loss_requirement_during_inference=False \
    model.gaze_model_config.gaze_decoder_config.num_multi_token_pred=10 \
    task.recon_model=facebook/vit-mae-large \
    task.recon_sample_rate=0.125 \
    task.recon_model_config.loss_type=l1+dinov2_reg+siglip2 \
    task.recon_model_config.loss_weights=1+0.3+0.3 \
    task.scales=32+64+112+224 \
    algorithm.optimize_task_loss_prediction=True \
    trainer.train_gaze=True \
    trainer.train_task=False \
    trainer.detach_task=True \
    trainer.lr=5e-4 \
    trainer.lr_schedule=linear \
    trainer.optimizer=adam \
    trainer.n_epochs=150 \
    trainer.batch_size=32 \
    trainer.per_gpu_max_batch_size=4 \
    trainer.temp_schedule_args.exp.temp_end=0.3 \
    trainer.temp_schedule_args.exp.temp_start=3 \
    trainer.val_nsteps=500 \
    trainer.save_nsteps=200 \
    trainer.task_weights="${VIDEOMAE_PT}" \
    trainer.exp_name=ntp_single_gpu \
    ${EXTRA_OVERRIDES}

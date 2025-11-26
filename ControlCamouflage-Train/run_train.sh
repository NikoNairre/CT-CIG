#!/bin/bash
export CUDA_VISIBLE_DEVICES=3,4,5,6         #specific GPU devices
accelerate launch --num_processes=4 train_CTCIG.py \
    --pretrained_model_name_or_path "stabilityai/stable-diffusion-xl-base-1.0" \
    --pretrained_vae_model_name_or_path "madebyollin/sdxl-vae-fp16-fix" \
    --variant fp16 \
    --use_safetensors \
    --output_dir "train/training_exp" \
    --checkpoint_dir "train/training_exp/checkpoints" \
    --logging_dir "logs" \
    --resolution 512 \
    --num_train_epochs 80 \
    --controlnet_lr_decay_start_epoch 40 \
    --controlnet_lr_decay_end_epoch 60 \
    --validation_steps 1000 \
    --checkpointing_steps 10000 \
    --set_grads_to_none \
    --proportion_empty_prompts 0.1 \
    --controlnet_scale_factor 1.2 \
    --save_weights_increaments "True" \
    --mixed_precision fp16 \
    --custom_jsonl_file /cver/yhqian/Datasets/LAKERED_DATASET/train/vlm_jsons_train/processed_jsons/processed_vl25_mask_camed_0729.json \
    --masks_folder /cver/yhqian/Datasets/LAKERED_DATASET/train/masks \
    --images_folder /cver/yhqian/Datasets/LAKERED_DATASET/train/images \
    --image_column "image" \
    --conditioning_image_column "conditioning_image" \
    --caption_column "text" \
    --validation_prompt "Brown owl on gravel with brown and white speckled feathers" \
    --validation_image "examples/camouflage/COD_CAMO_camourflage_00208.png" \
    --learning_rate 5e-6 \
    --positive_prompt_prefix "photorealistic, high quality, camouflaged" \
    --report_to tensorboard \
    #--gradient_checkpointing \
    #--enable_xformers_memory_efficient_attention \
    #--dataset_name "Nahrawy/VIDIT-Depth-ControlNet" \      #use custom camouflage datasets here
    #original controlnet_scale_factor=1，which I modify to 1.2
    #original proportion_empty_prompts=0.2, which I modify to 0.1
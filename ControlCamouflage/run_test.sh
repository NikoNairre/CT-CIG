#!/bin/bash
export CUDA_VISIBLE_DEVICES=2
python test_CTCIG.py \
    --pretrained_model_name_or_path "stabilityai/stable-diffusion-xl-base-1.0" \
    --unet_model_name_or_path "/cver/yhqian/CIG/CT-CIG/ControlCamouflage-Train/train/training_exp/checkpoints/checkpoint-10000/unet_weight_increasements.safetensors" \
    --controlnet_model_name_or_path "/cver/yhqian/CIG/CT-CIG/ControlCamouflage-Train/train/training_exp/checkpoints/checkpoint-10000/controlnet.safetensors" \
    --controlnet_scale "1.2" \
    --vae_model_name_or_path "madebyollin/sdxl-vae-fp16-fix" \
    --output_dir  "test_output"\
    --width "512" \
    --height "512" \
    --load_weight_increasement \
    --jsonl_path "/cver/yhqian/Datasets/LAKERED_DATASET/validation/vlm_jsons_validation/processed_jsons/processed_vl25_validation.json" \
    --image_folder "/cver/yhqian/Datasets/LAKERED_DATASET/validation/images" \
    --mask_folder "/cver/yhqian/Datasets/LAKERED_DATASET/validation/masks" \
    #--positive_prompt "photorealistic, high quality, camouflaged"
    #--seed "0" \
    #"--enable_xformers_memory_efficient_attention" 
    #"examples"
    # --validation_prompt "A cat in a field with dried grass and leaves." \
    # --validation_image "/cver/yhqian/Datasets/LAKERED_DATASET/validation/masks/COD_COD10K_COD10K-CAM-2-Terrestrial-23-Cat-1442.png" \
    
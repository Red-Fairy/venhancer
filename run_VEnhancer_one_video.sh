#!/bin/bash

input_video_path=$1
output_video_path=$2

python enhance_a_video.py \
--version v2 \
--up_scale 2 --target_fps 10 --noise_aug 150 \
--solver_mode 'fast' --steps 15 \
--input_path $input_video_path \
--prompt_path "A clear, high-resolution 360-degree panorama. Sharp details, natural colors, and seamless stitching throughout the entire view." \
--save_dir $output_video_path


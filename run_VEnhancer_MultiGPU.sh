N_PARALLEL_GPUS=4

torchrun --nproc_per_node=${N_PARALLEL_GPUS} enhance_a_video_MultiGPU.py \
--version v2 \
--up_scale 2 --target_fps 5 --noise_aug 250 \
--solver_mode 'fast' --steps 15 \
--input_path /home/rl897/VEnhancer/input_videos/1-SVD.mp4 \
--prompt_path input_videos/prompts.txt \
--save_dir 'results/' \
--blended_decoding \
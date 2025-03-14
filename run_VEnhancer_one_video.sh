
input_video_path=$1
output_video_path=$2

python enhance_a_video.py \
--version v2 \
--up_scale 2 --target_fps 10 --noise_aug 150 \
--solver_mode 'fast' --steps 15 \
--input_path $input_video_path \
--prompt_path input_videos/prompts.txt \
--save_dir $output_video_path



root=$1
num_processes=${2:-1}
process_id=${3:-0}
frame_rate=${4:-5}

echo "num_processes: ${num_processes}"
echo "process_id: ${process_id}"
echo "frame_rate: ${frame_rate}"

video_paths=(${root}/*.mp4)
selected_paths=()

for i in "${!video_paths[@]}"; do
  if (( i % num_processes == process_id )); then
    selected_paths+=("${video_paths[$i]}")
  fi
done

output_root=${root}_enhanced


for video_path in ${selected_paths[@]}; do

    echo "Processing video ${video_path}"

    # if the video is already enhanced, skip
    if [ -f "${output_root}_${frame_rate}/${video_path##*/}" ]; then
        echo "Video ${video_path} already enhanced, skipping"
        continue
    fi

    python enhance_a_video.py \
    --version v2 \
    --up_scale 2 --target_fps ${frame_rate} --noise_aug 100 \
    --solver_mode 'fast' --steps 15 \
    --input_path ${video_path} \
    --prompt_path input_videos/prompts.txt \
    --save_dir ${output_root}_${frame_rate} \
    # --blended_decoding \
    # --rotation_decoding 
    
done

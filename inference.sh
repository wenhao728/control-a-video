CUDA_VISIBLE_DEVICES=1 python3 inference.py \
    --prompt "a bear walking through stars, artstation" \
    --input_video bear.mp4 \
    --control_mode depth \
    --num_sample_frames 16 \
    --each_sample_frame 8 
CUDA_VISIBLE_DEVICES=0 python main.py --output_path vcm/clip_text_base  --weight_a 0.0 --dataset vcm --steps 300
CUDA_VISIBLE_DEVICES=1 python main.py --output_path vcm/clip_text_0.05  --weight_a 0.05 --dataset vcm --steps 300
CUDA_VISIBLE_DEVICES=2 python main.py --output_path vcm/clip_text_0.1  --weight_a 0.01 --dataset vcm --steps 300

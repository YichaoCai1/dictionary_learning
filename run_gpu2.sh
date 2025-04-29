echo "running dictionary learning..."

CUDA_VISIBLE_DEVICES=2 python train_sae_baselines.py --experiment_name topk_24k --device cuda
wait
CUDA_VISIBLE_DEVICES=2 python train_sae_lora.py --experiment_name panneallora_1_24k --device cuda --lora_coeff_scale 1.0
wait

echo "training end."
echo "running dictionary learning..."

CUDA_VISIBLE_DEVICES=1 python train_sae_baselines.py --experiment_name panneal_24k --device cuda
wait
CUDA_VISIBLE_DEVICES=1 python train_sae_lora.py --experiment_name panneallora_1e-1_24k --device cuda --lora_coeff_scale 0.1
wait

echo "training end."
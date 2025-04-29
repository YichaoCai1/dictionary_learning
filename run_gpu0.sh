echo "running dictionary learning..."

CUDA_VISIBLE_DEVICES=0 python train_sae_lora.py --experiment_name panneallora_1e-3_24k --device cuda --lora_coeff_scale 0.001
wait

echo "training end."
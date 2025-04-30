echo "running dictionary learning..."

CUDA_VISIBLE_DEVICES=1 python train_sae_lora.py --experiment_name panneallora_1e-1_24k --device cuda --lora_coeff_scale 0.1
wait

echo "training end."

# python train_sae_lora.py --experiment_name panneallora_1e-2_24k --device cuda --lora_coeff_scale 0.01
# python train_sae_lora.py --experiment_name panneallora_10_24k --device cuda --lora_coeff_scale 10
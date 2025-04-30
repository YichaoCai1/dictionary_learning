echo "running dictionary learning..."

CUDA_VISIBLE_DEVICES=3 python train_sae_lora.py --experiment_name panneallora_1e-2_24k --device cuda --lora_coeff_scale 0.01
wait

echo "training end."
echo "running dictionary learning..."

CUDA_VISIBLE_DEVICES=3 python train_sae_baselines.py --experiment_name batchtopk_24k --device cuda
wait

echo "training end."
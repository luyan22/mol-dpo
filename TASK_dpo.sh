# ref model training for lumo 
export CUDA_VISIBLE_DEVICES=6
export LD_LIBRARY_PATH=/home/admin01/SKData/miniconda3/envs/MOLuni/lib
python main_qm9.py --exp_name lumo_context_embedding_cond  --model egnn_dynamics --lr 1e-4  --nf 192 --n_layers 9 --save_model True --diffusion_steps 1000 --sin_embedding False --n_epochs 3000 --n_stability_samples 500 --diffusion_noise_schedule polynomial_2 --diffusion_noise_precision 1e-5 --dequantization deterministic --include_charges False --diffusion_loss_type l2 --batch_size 64 --normalize_factors [1,8,1] --conditioning lumo --dataset qm9_second_half

# DPO training for lumo 
export CUDA_VISIBLE_DEVICES=6
export LD_LIBRARY_PATH=/home/admin01/SKData/miniconda3/envs/MOLuni/lib
srun --gres=gpu:a100:2 --time 8-12:00:00 python -m DPO.main_dpo --ref_fold outputs/train_prop_pred4condition_only --model_path outputs/train_prop_pred4condition_only/generative_model_ema_940.npy --finetune_fold DPO/finetune/lumo --exp_name DPO_lumo_train 

# train unconditional diffusion model on the task of predicting the next token in a sequence of SMILES strings
srun --gres=gpu:a100:2 --time 5-12:00:00  python -u main_qm9.py --n_epochs 3000 --exp_name PAT/PAT_gen --n_stability_samples 1000 --diffusion_noise_schedule polynomial_2 --diffusion_noise_precision 1e-5 --diffusion_steps 1000 --diffusion_loss_type l2 --batch_size 64 --nf 256 --n_layers 9 --lr 1e-4 --normalize_factors [1,4,10] --test_epochs 20 --ema_decay 0.9999 --mode PAT > logs/PAT/train_edm_qm9.log 2>&1 &


#eval
srun --gres=gpu:a100:2 --time 1-12:00:00 python eval_analyze.py --model_path outputs/PAT/PAT_gen --n_samples 10_000 > logs/PAT/eval_PAT.log 2>&1 &
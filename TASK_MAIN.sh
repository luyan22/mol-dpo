#demo task
CUDA_VISIBLE_DEVICES=6 
python -u main_qm9.py --exp_name exp_cond_lumo_pretrained_only_denoise --model egnn_dynamics --lr 1e-4 --nf 192 --n_layers 9 --save_model True --diffusion_steps 1000 --sin_embedding False --n_epochs 3000 --n_stability_samples 500 --diffusion_noise_schedule polynomial_2 --diffusion_noise_precision 1e-5 --dequantization deterministic --include_charges False --diffusion_loss_type l2 --batch_size 64 --normalize_factors \[1,8,1\] --conditioning lumo --dataset qm9_second_half --model torchmdnet --nf 256 --finetune 1 --pretrained_model outputs/pcq_torchmd_pretrain_4gpu_only_denoising/generative_model_18.npy 
#eval_pretrain_01 2990
srun --gres=gpu:a100:2 --time 3-12:00:00 python eval_conditional_qm9.py --generators_path outputs/exp_cond_lumo_pretrained_only_denoise --classifiers_path qm9/property_prediction/outputs/exp_class_lumo_22_one_hot --property lumo  --iterations 100  --batch_size 100 --task edm --finetune 1 > /home/AI4Science/luy2402/e3_diffusion_for_molecules/logs/exp_cond_lumo_pretrained_only_denoise.log 2>&1
#eval_pretrain_01 1460
srun --gres=gpu:a100:2 --time 3-12:00:00 python -u eval_conditional_qm9.py --generators_path outputs/exp_cond_lumo_pretrained_only_denoise_1460 --classifiers_path qm9/property_prediction/outputs/exp_class_lumo_22_one_hot --property lumo  --iterations 100  --batch_size 100 --task edm --finetune 1 > /home/AI4Science/luy2402/e3_diffusion_for_molecules/logs/exp_cond_lumo_pretrained_only_denoise_1460.log 2>&1
#eval_pretrain_01 550
srun --gres=gpu:a100:2 --time 3-12:00:00 python -u eval_conditional_qm9.py --generators_path outputs/exp_cond_lumo_pretrained_only_denoise_550 --classifiers_path qm9/property_prediction/outputs/exp_class_lumo_22_one_hot --property lumo  --iterations 100  --batch_size 100 --task edm --finetune 1 > /home/AI4Science/luy2402/e3_diffusion_for_molecules/logs/exp_cond_lumo_pretrained_only_denoise_550.log 2>&1

#eval_pretrain_02
srun --gres=gpu:a100:2 --time 1-12:00:00 python -u eval_conditional_qm9.py --generators_path outputs/exp_cond_lumo_pretrained_resume --classifiers_path qm9/property_prediction/outputs/exp_class_lumo_22_one_hot --property lumo  --iterations 100  --batch_size 100 --task edm --finetune 1 > /home/AI4Science/luy2402/e3_diffusion_for_molecules/logs/exp_cond_lumo_pretrained_resume.log 2>&1
#eval_pretrain_02 550
srun --gres=gpu:a100:2 --time 1-12:00:00 python -u eval_conditional_qm9.py --generators_path outputs/exp_cond_lumo_pretrained_resume_550 --classifiers_path qm9/property_prediction/outputs/exp_class_lumo_22_one_hot --property lumo  --iterations 100  --batch_size 100 --task edm --finetune 1 > /home/AI4Science/luy2402/e3_diffusion_for_molecules/logs/exp_cond_lumo_pretrained_resume_550.log 2>&1

#eval_scratch
srun --gres=gpu:a100:2 --time 1-12:00:00 python -u eval_conditional_qm9.py --generators_path outputs/exp_cond_lumo_baseline_resume --classifiers_path qm9/property_prediction/outputs/exp_class_lumo_22_one_hot --property lumo  --iterations 100  --batch_size 100 --task edm --finetune 1 > /home/AI4Science/luy2402/e3_diffusion_for_molecules/logs/exp_cond_lumo_baseline_resume.log 2>&1
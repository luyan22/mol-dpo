#train script unconditional
srun --gres=gpu:a100:2 --time 3-12:00:00 python -u main_qm9.py --n_epochs 3000 --exp_name edm_qm9_DGAP --n_stability_samples 1000 --diffusion_noise_schedule polynomial_2 --diffusion_noise_precision 1e-5 --diffusion_steps 1000 --diffusion_loss_type l2 --batch_size 64 --nf 256 --n_layers 9 --lr 1e-4 --normalize_factors [1,4,10] --test_epochs 20 --ema_decay 0.9999  --property_pred 1 --prediction_threshold_t 10 --model DGAP --target_property lumo > logs/DGAP/edm_qm9_DGAP.log 2>&1
#train without property prediction loss
srun --gres=gpu:a100:2 --time 5-12:00:00 python -u main_qm9.py --n_epochs 3000 --exp_name DGAP_no_prop_pred --n_stability_samples 1000 --diffusion_noise_schedule polynomial_2 --diffusion_noise_precision 1e-5 --diffusion_steps 1000 --diffusion_loss_type l2 --batch_size 64 --nf 256 --n_layers 9 --lr 1e-4 --normalize_factors [1,4,10] --test_epochs 20 --ema_decay 0.9999  --property_pred 1 --prediction_threshold_t 10 --model DGAP --target_property lumo --use_prop_pred 0 > logs/DGAP/edm_qm9_DGAP_no_prop_pred.log 2>&1

#train DGAP with normal time sampling(这组实验不正常 不能复现baseline)
srun --gres=gpu:a100:2 --time 5-12:00:00 python -u main_qm9.py --n_epochs 3000 --exp_name DGAP_normal_time_sampling --n_stability_samples 1000 --diffusion_noise_schedule polynomial_2 --diffusion_noise_precision 1e-5 --diffusion_steps 1000 --diffusion_loss_type l2 --batch_size 64 --nf 256 --n_layers 9 --lr 1e-4 --normalize_factors [1,4,10] --test_epochs 20 --ema_decay 0.9999  --property_pred 1 --prediction_threshold_t 10 --model DGAP --target_property lumo > logs/DGAP/train_normal_time_sampling.log 2>&1
#train DGAP with normal time sampling and no property prediction loss(和上一组对比 看看能不能复现baseline)
srun --gres=gpu:a100:2 --time 5-12:00:00 python -u main_qm9.py --n_epochs 3000 --exp_name DGAP_normal_time_sampling_no_loss --n_stability_samples 1000 --diffusion_noise_schedule polynomial_2 --diffusion_noise_precision 1e-5 --diffusion_steps 1000 --diffusion_loss_type l2 --batch_size 64 --nf 256 --n_layers 9 --lr 1e-4 --normalize_factors [1,4,10] --test_epochs 20 --ema_decay 0.9999  --property_pred 1 --prediction_threshold_t 10 --model DGAP --target_property lumo --use_prop_pred 0 > logs/DGAP/train_normal_time_sampling_no_loss.log 2>&1


#script for debug
srun --gres=gpu:a100:2 --time 3-12:00:00 python main_qm9.py --n_epochs 3000 --exp_name edm_qm9_DGAP_test --n_stability_samples 1000 --diffusion_noise_schedule polynomial_2 --diffusion_noise_precision 1e-5 --diffusion_steps 1000 --diffusion_loss_type l2 --batch_size 64 --nf 256 --n_layers 9 --lr 1e-4 --normalize_factors [1,4,10] --test_epochs 20 --ema_decay 0.9999  --property_pred 1 --prediction_threshold_t 10 --model DGAP --target_property lumo --start_epoch 19 > logs/DGAP/edm_qm9_DGAP_test.log 2>&1


srun --gres=gpu:a100:2 --time 3-12:00:00 python -u main_qm9.py --n_epochs 3000 --exp_name edm_qm9_DGAP --n_stability_samples 1000 --diffusion_noise_schedule polynomial_2 --diffusion_noise_precision 1e-5 --diffusion_steps 1000 --diffusion_loss_type l2 --batch_size 64 --nf 256 --n_layers 9 --lr 1e-4 --normalize_factors [1,4,10] --test_epochs 20 --ema_decay 0.9999  --property_pred 1 --prediction_threshold_t 10 --model DGAP --target_property lumo --resume outputs/edm_qm9_DGAP --start_epoch=60 > logs/DGAP/edm_qm9_DGAP_60.log 2>&1
# > logs/edm_qm9_DGAP.log 2>&1

#eval generation script
srun --gres=gpu:a100:2 --time 3-12:00:00 python -u eval_analyze.py --model_path outputs/edm_qm9_DGAP_resume --n_samples 10_000 --save_to_xyz 0 > logs/DGAP/edm_qm9_DGAP_eval_gen.log 2>&1

srun --gres=gpu:a100:2 --time 3-12:00:00 python -u eval_analyze.py --model_path outputs/DGAP_baseline_gen_resume --n_samples 10_000 --save_to_xyz 0 > logs/DGAP/baseline_DGAP_eval_gen.log 2>&1

#eval generation DGAP 900
srun --gres=gpu:a100:2 --time 3-12:00:00 python -u eval_analyze.py --model_path outputs/edm_qm9_DGAP_resume --n_samples 10_000 --save_to_xyz 0 --checkpoint_epoch 900 > logs/DGAP/edm_qm9_DGAP_eval_gen_900.log 2>&1


#eval generation baseline 450
srun --gres=gpu:a100:2 --time 3-12:00:00 python -u eval_analyze.py --model_path outputs/DGAP_baseline_gen_resume --n_samples 10_000 --save_to_xyz 0 --checkpoint_epoch 440 > logs/DGAP/baseline_DGAP_eval_gen_440.log 2>&1


#eval generation baseline 900
srun --gres=gpu:a100:2 --time 3-12:00:00 python -u eval_analyze.py --model_path outputs/DGAP_baseline_gen_resume --n_samples 10_000 --save_to_xyz 0 --checkpoint_epoch 900 > logs/DGAP/baseline_DGAP_eval_gen_900.log 2>&1

#eval property prediction script DONE
cd qm9/property_prediction
srun --gres=gpu:a100:2 --time 3-12:00:00 python -u eval_prop_pred.py --num_workers 2 --lr 5e-4 --property lumo --model_name egnn > ../../logs/DGAP/prop_pred.log 2>&1

#eval DGAP property prediction baseline(normal time sampling)
cd qm9/property_prediction
srun --gres=gpu:a100:2 --time 3-12:00:00 python -u eval_prop_pred.py --num_workers 2 --lr 5e-4 --property lumo --model_name egnn --generators_path ../../outputs/DGAP_normal_time_sampling --model_path generative_model_ema_2980.npy > ../../logs/DGAP/prop_pred_normal_time_sampling.log 2>&1

#eval DGAP property prediction 0~10 timestamp for training
cd qm9/property_prediction
srun --gres=gpu:a100:2 --time 3-12:00:00 python -u eval_prop_pred.py --num_workers 2 --lr 5e-4 --property lumo --model_name egnn --generators_path ../../outputs/fengsk_noisy_node_gen --model_path generative_model_ema_720.npy > ../../logs/DGAP/prop_pred_fengsk_noisy_node_gen.log 2>&1

#eval DGAP generation baseline(normal time sampling) use the last epoch
srun --gres=gpu:a100:2 --time 3-12:00:00 python -u eval_analyze.py --model_path outputs/DGAP_normal_time_sampling --n_samples 10_000 --save_to_xyz 0 --checkpoint_epoch 2980 > logs/DGAP/DGAP_normal_time_sampling_eval_gen_2980.log 2>&1

#eval DGAP generation baseline(normal time sampling no loss) use the last epoch
srun --gres=gpu:a100:2 --time 3-12:00:00 python -u eval_analyze.py --model_path outputs/DGAP_normal_time_sampling_no_loss --n_samples 10_000 --save_to_xyz 0 --checkpoint_epoch 2980 > logs/DGAP/DGAP_normal_time_sampling_no_loss_eval_gen_2980.log 2>&1


# eval no property prediction 440
srun --gres=gpu:a100:2 --time 3-12:00:00 python -u eval_analyze.py --model_path outputs/DGAP_no_prop_pred --n_samples 10_000 --save_to_xyz 0 --checkpoint_epoch 440 > logs/DGAP/DGAP_no_prop_pred_eval_gen_440.log 2>&1
#--property lumo


#train classifier lumo 500 epochs DONE
cd qm9/property_prediction
srun --gres=gpu:a100:2 --time 3-12:00:00 python -u main_qm9_prop.py --num_workers 2 --lr 5e-4 --property lumo --exp_name exp_class_lumo_500 --model_name egnn --epochs 500 > ../../logs/DGAP/baseline_exp_class_lumo_500.log 2>&1

#train baseline unconditional generator
srun --gres=gpu:a100:2 --time 3-12:00:00 python main_qm9.py --n_epochs 3000 --exp_name DGAP_baseline_gen --n_stability_samples 1000 --diffusion_noise_schedule polynomial_2 --diffusion_noise_precision 1e-5 --diffusion_steps 1000 --diffusion_loss_type l2 --batch_size 64 --nf 256 --n_layers 9 --lr 1e-4 --normalize_factors [1,4,10] --test_epochs 20 --ema_decay 0.9999  > logs/DGAP/DGAP_baseline_gen.log 2>&1

srun --gres=gpu:a100:2 --time 3-12:00:00 python main_qm9.py --n_epochs 3000 --exp_name DGAP_baseline_gen --n_stability_samples 1000 --diffusion_noise_schedule polynomial_2 --diffusion_noise_precision 1e-5 --diffusion_steps 1000 --diffusion_loss_type l2 --batch_size 64 --nf 256 --n_layers 9 --lr 1e-4 --normalize_factors [1,4,10] --test_epochs 20 --ema_decay 0.9999 --resume outputs/DGAP_baseline_gen_resume_140 --start_epoch 140 > logs/DGAP/DGAP_baseline_gen_resume_140.log 2>&1

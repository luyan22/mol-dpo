#跑之前需要新建一个logs文件夹
#--use_prop_pred 0 表示不使用property prediction loss(默认是1)

#train without property prediction loss【无梯度 + 0.5timestamp】
srun --gres=gpu:a100:2 --time 5-12:00:00 python -u main_qm9.py --n_epochs 3000 --exp_name DGAP_no_prop_pred --n_stability_samples 1000 --diffusion_noise_schedule polynomial_2 --diffusion_noise_precision 1e-5 --diffusion_steps 1000 --diffusion_loss_type l2 --batch_size 64 --nf 256 --n_layers 9 --lr 1e-4 --normalize_factors [1,4,10] --test_epochs 20 --ema_decay 0.9999  --property_pred 1 --prediction_threshold_t 10 --model DGAP --target_property lumo --use_prop_pred 0 > logs/DGAP/edm_qm9_DGAP_no_prop_pred.log 2>&1

#train DGAP with normal time sampling(这组实验不正常 不能复现baseline)【梯度 + 正常时间采样】
srun --gres=gpu:a100:2 --time 5-12:00:00 python -u main_qm9.py --n_epochs 3000 --exp_name DGAP_normal_time_sampling --n_stability_samples 1000 --diffusion_noise_schedule polynomial_2 --diffusion_noise_precision 1e-5 --diffusion_steps 1000 --diffusion_loss_type l2 --batch_size 64 --nf 256 --n_layers 9 --lr 1e-4 --normalize_factors [1,4,10] --test_epochs 20 --ema_decay 0.9999  --property_pred 1 --prediction_threshold_t 10 --model DGAP --target_property lumo > logs/DGAP/train_normal_time_sampling.log 2>&1
#train DGAP with normal time sampling and no property prediction loss(和上一组对比 看看能不能复现baseline)
#【无梯度 + 正常时间采样】
srun --gres=gpu:a100:2 --time 5-12:00:00 python -u main_qm9.py --n_epochs 3000 --exp_name DGAP_normal_time_sampling_no_loss --n_stability_samples 1000 --diffusion_noise_schedule polynomial_2 --diffusion_noise_precision 1e-5 --diffusion_steps 1000 --diffusion_loss_type l2 --batch_size 64 --nf 256 --n_layers 9 --lr 1e-4 --normalize_factors [1,4,10] --test_epochs 20 --ema_decay 0.9999  --property_pred 1 --prediction_threshold_t 10 --model DGAP --target_property lumo --use_prop_pred 0 > logs/DGAP/train_normal_time_sampling_no_loss.log 2>&1



# 
# random_number < -1: hard code
# training the code baseline
CUDA_VISIBLE_DEVICES=7 python -u main_qm9.py --n_epochs 3000 --exp_name DGAP_normal_time_sampling_no_loss --n_stability_samples 1000 --diffusion_noise_schedule polynomial_2 --diffusion_noise_precision 1e-5 --diffusion_steps 1000 --diffusion_loss_type l2 --batch_size 256 --nf 256 --n_layers 9 --lr 1e-4 --normalize_factors [1,4,10] --test_epochs 20 --ema_decay 0.9999  --property_pred 1 --prediction_threshold_t 10 --model DGAP --target_property lumo --use_prop_pred 0 > train_qm9_baseline.log 2>&1 &

# training with the molecule property prediction without the change the time step
CUDA_VISIBLE_DEVICES=6 python -u main_qm9.py --n_epochs 3000 --exp_name DGAP_normal_time_sampling --n_stability_samples 1000 --diffusion_noise_schedule polynomial_2 --diffusion_noise_precision 1e-5 --diffusion_steps 1000 --diffusion_loss_type l2 --batch_size 256 --nf 256 --n_layers 9 --lr 1e-4 --normalize_factors [1,4,10] --test_epochs 20 --ema_decay 0.9999  --property_pred 1 --prediction_threshold_t 10 --model DGAP --target_property lumo > train_normal_time_sampling_with_prediction.log 2>&1 &


# freeze weight,  half prediciton(gradient don't influence generation), half generation:
# if random_number < -1 --> if random_number < 0.5:
CUDA_VISIBLE_DEVICES=4 python -u main_qm9.py --n_epochs 3000 --exp_name DGAP_freeze_weight --n_stability_samples 1000 --diffusion_noise_schedule polynomial_2 --diffusion_noise_precision 1e-5 --diffusion_steps 1000 --diffusion_loss_type l2 --batch_size 256 --nf 256 --n_layers 9 --lr 1e-4 --normalize_factors [1,4,10] --test_epochs 20 --ema_decay 0.9999  --property_pred 1 --prediction_threshold_t 10 --model DGAP --target_property lumo --freeze_gradient 1 > DGAP_freeze_weight.log 2>&1 &



# unnormal sampling without the property prediction loss
# if random_number < -1 --> if random_number < 0.5:
CUDA_VISIBLE_DEVICES=5 python -u main_qm9.py --n_epochs 3000 --exp_name DGAP_unnormal_time_wo_prop --n_stability_samples 1000 --diffusion_noise_schedule polynomial_2 --diffusion_noise_precision 1e-5 --diffusion_steps 1000 --diffusion_loss_type l2 --batch_size 256 --nf 256 --n_layers 9 --lr 1e-4 --normalize_factors [1,4,10] --test_epochs 20 --ema_decay 0.9999  --property_pred 1 --prediction_threshold_t 10 --model DGAP --target_property lumo --use_prop_pred 0 > DGAP_unnormal_time_wo_prop.log 2>&1 &
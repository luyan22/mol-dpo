#add atom type to the property prediction task: cat
srun --gres=gpu:a100:2 --time 8-12:00:00 python -u main_qm9.py --n_epochs 3000 --exp_name atom_type_cat4_prop_pred --n_stability_samples 1000 --diffusion_noise_schedule polynomial_2 --diffusion_noise_precision 1e-5 --diffusion_steps 1000 --diffusion_loss_type l2 --batch_size 64 --nf 256 --n_layers 9 --lr 1e-4 --normalize_factors [1,4,10] --test_epochs 20 --ema_decay 0.9999 --property_pred 1 --prediction_threshold_t 10 --model DGAP --sep_noisy_node 1 --target_property U --atom_type_pred 1 --branch_layers_num 8 --use_prop_pred 1 --atom_type4prop_pred 1 --atom_connect_fn cat > logs/atom_type_cat_prop_pred.log 2>&1 &

#command for debug:
#add atom type to the property prediction task: cat
python -u main_qm9.py --n_epochs 3000 --exp_name atom_type_cat4_prop_pred --n_stability_samples 1000 --diffusion_noise_schedule polynomial_2 --diffusion_noise_precision 1e-5 --diffusion_steps 1000 --diffusion_loss_type l2 --batch_size 64 --nf 256 --n_layers 9 --lr 1e-4 --normalize_factors [1,4,10] --test_epochs 20 --ema_decay 0.9999 --property_pred 1 --prediction_threshold_t 10 --model DGAP --sep_noisy_node 1 --target_property U --atom_type_pred 1 --branch_layers_num 8 --use_prop_pred 1 --atom_type4prop_pred 1 --atom_connect_fn cat --start_epoch 20 --resume outputs/atom_type_cat4_prop_pred > logs/debug_atom_type_cat_prop_pred.log 2>&1 &

python -u main_qm9.py --n_epochs 3000 --exp_name atom_type_cat4_prop_pred --n_stability_samples 1000 --diffusion_noise_schedule polynomial_2 --diffusion_noise_precision 1e-5 --diffusion_steps 1000 --diffusion_loss_type l2 --batch_size 64 --nf 256 --n_layers 9 --lr 1e-4 --normalize_factors [1,4,10] --test_epochs 20 --ema_decay 0.9999 --property_pred 1 --prediction_threshold_t 10 --model DGAP --sep_noisy_node 1 --target_property U --atom_type_pred 1 --branch_layers_num 8 --use_prop_pred 1 --start_epoch 0 > logs/debug_normal.log 2>&1 &

#add atom type to the property prediction task: sum
srun --gres=gpu:a100:2 --time 8-12:00:00 python -u main_qm9.py --n_epochs 3000 --exp_name atom_type_sum4_prop_pred --n_stability_samples 1000 --diffusion_noise_schedule polynomial_2 --diffusion_noise_precision 1e-5 --diffusion_steps 1000 --diffusion_loss_type l2 --batch_size 64 --nf 256 --n_layers 9 --lr 1e-4 --normalize_factors [1,4,10] --test_epochs 20 --ema_decay 0.9999 --property_pred 1 --prediction_threshold_t 10 --model DGAP --sep_noisy_node 1 --target_property U --atom_type_pred 1 --branch_layers_num 8 --use_prop_pred 1 --atom_type4prop_pred 1 --atom_connect_fn sum > logs/atom_type_sum_prop_pred.log 2>&1 &


# train train_prop_pred_4condition_only: lumo
srun --gres=gpu:a100:2 --time 8-12:00:00 python -u main_qm9.py --n_epochs 3000 --exp_name train_prop_pred4condition_only --n_stability_samples 1000 --diffusion_noise_schedule polynomial_2 --diffusion_noise_precision 1e-5 --diffusion_steps 1000 --diffusion_loss_type l2 --batch_size 64 --nf 256 --n_layers 9 --lr 1e-4 --normalize_factors [1,4,10] --test_epochs 20 --ema_decay 0.9999 --property_pred 1 --prediction_threshold_t 10 --model DGAP --sep_noisy_node 1 --target_property lumo --atom_type_pred 1 --branch_layers_num 8 --use_prop_pred 1 --train_prop_pred_4condition_only 1 --pretrained_model outputs/split_k8_t10_with_atom_type_prop_pred/generative_model_ema.npy > logs/train_prop_pred_4condition_only.log 2>&1 &

# train train_prop_pred_4condition_only: alpha
srun --gres=gpu:a100:2 --time 8-12:00:00 python -u main_qm9.py --n_epochs 3000 --exp_name alpha_train_prop_pred4condition_only --n_stability_samples 1000 --diffusion_noise_schedule polynomial_2 --diffusion_noise_precision 1e-5 --diffusion_steps 1000 --diffusion_loss_type l2 --batch_size 64 --nf 256 --n_layers 9 --lr 1e-4 --normalize_factors [1,4,10] --test_epochs 20 --ema_decay 0.9999 --property_pred 1 --prediction_threshold_t 10 --model DGAP --sep_noisy_node 1 --target_property alpha --atom_type_pred 1 --branch_layers_num 8 --use_prop_pred 1 --train_prop_pred_4condition_only 1 --pretrained_model "condGen/1-head model/alpha/generative_model_ema_2980.npy" > logs/condGen/train_prop_pred_4condition_only_alpha.log 2>&1 &

# train train_prop_pred_4condition_only: gap
srun --gres=gpu:a100:2 --time 8-12:00:00 python -u main_qm9.py --n_epochs 3000 --exp_name gap_train_prop_pred4condition_only --n_stability_samples 1000 --diffusion_noise_schedule polynomial_2 --diffusion_noise_precision 1e-5 --diffusion_steps 1000 --diffusion_loss_type l2 --batch_size 64 --nf 256 --n_layers 9 --lr 1e-4 --normalize_factors [1,4,10] --test_epochs 20 --ema_decay 0.9999 --property_pred 1 --prediction_threshold_t 10 --model DGAP --sep_noisy_node 1 --target_property gap --atom_type_pred 1 --branch_layers_num 8 --use_prop_pred 1 --train_prop_pred_4condition_only 1 --pretrained_model "condGen/1-head model/gap/generative_model_ema_2980.npy" > logs/condGen/train_prop_pred_4condition_only_gap.log 2>&1 &


# train train_prop_pred_4condition_only: homo
srun --gres=gpu:a100:2 --time 8-12:00:00 python -u main_qm9.py --n_epochs 3000 --exp_name homo_train_prop_pred4condition_only --n_stability_samples 1000 --diffusion_noise_schedule polynomial_2 --diffusion_noise_precision 1e-5 --diffusion_steps 1000 --diffusion_loss_type l2 --batch_size 64 --nf 256 --n_layers 9 --lr 1e-4 --normalize_factors [1,4,10] --test_epochs 20 --ema_decay 0.9999 --property_pred 1 --prediction_threshold_t 10 --model DGAP --sep_noisy_node 1 --target_property homo --atom_type_pred 1 --branch_layers_num 8 --use_prop_pred 1 --train_prop_pred_4condition_only 1 --pretrained_model "condGen/1-head model/homo/generative_model_ema_2980.npy" > logs/condGen/train_prop_pred_4condition_only_homo.log 2>&1 &

# train train_prop_pred_4condition_only: mu
srun --gres=gpu:a100:2 --time 8-12:00:00 python -u main_qm9.py --n_epochs 3000 --exp_name mu_train_prop_pred4condition_only --n_stability_samples 1000 --diffusion_noise_schedule polynomial_2 --diffusion_noise_precision 1e-5 --diffusion_steps 1000 --diffusion_loss_type l2 --batch_size 64 --nf 256 --n_layers 9 --lr 1e-4 --normalize_factors [1,4,10] --test_epochs 20 --ema_decay 0.9999 --property_pred 1 --prediction_threshold_t 10 --model DGAP --sep_noisy_node 1 --target_property mu --atom_type_pred 1 --branch_layers_num 8 --use_prop_pred 1 --train_prop_pred_4condition_only 1 --pretrained_model "condGen/1-head model/mu/generative_model_ema_2980.npy" > logs/condGen/train_prop_pred_4condition_only_mu.log 2>&1 &

# train train_prop_pred_4condition_only: mu
srun --gres=gpu:a100:2 --time 8-12:00:00 python -u main_qm9.py --n_epochs 3000 --exp_name mu_train_prop_pred4condition_only --n_stability_samples 1000 --diffusion_noise_schedule polynomial_2 --diffusion_noise_precision 1e-5 --diffusion_steps 1000 --diffusion_loss_type l2 --batch_size 64 --nf 256 --n_layers 9 --lr 1e-4 --normalize_factors [1,4,10] --test_epochs 20 --ema_decay 0.9999 --property_pred 1 --prediction_threshold_t 10 --model DGAP --sep_noisy_node 1 --target_property mu --atom_type_pred 1 --branch_layers_num 8 --use_prop_pred 1 --train_prop_pred_4condition_only 1 --pretrained_model "condGen/1-head model/mu/generative_model_ema_2980.npy" > logs/condGen/train_prop_pred_4condition_only_mu.log 2>&1 &

#evaluate train_prop_pred_4condition_only on lumo for property prediction
cd qm9/property_prediction
srun --gres=gpu:a100:2 --time 3-12:00:00 python -u eval_prop_pred.py --num_workers 2 --lr 5e-4 --property lumo --model_name DGAP --generators_path /home/AI4Science/luy2402/e3_diffusion_for_molecules/outputs/train_prop_pred4condition_only --model_path generative_model_ema.npy > ../../logs/eval/eval_train_prop_pred_4condition_only_lumo.log 2>&1 &


#pretraining eval on lumo for property prediction
cd qm9/property_prediction
srun --gres=gpu:a100:2 --time 3-12:00:00 python -u eval_prop_pred.py --num_workers 2 --lr 5e-4 --property lumo --model_name DGAP --generators_path /home/AI4Science/luy2402/e3_diffusion_for_molecules/outputs/split_k8_t10_with_atom_type_prop_pred --model_path generative_model_ema.npy > ../../logs/eval/eval_pretrain_prop_pred_4condition_only_lumo.log 2>&1 &

# conditional generation guidance task: step-1 training use selves head to predict property, decrease loss(pred, pseudo_context)
CUDA_VISIBLE_DEVICES=4 python -u eval_conditional_qm9.py --generators_path outputs/train_prop_pred4condition_only --classifiers_path qm9/property_prediction/outputs/exp_class_lumo --property lumo --iterations 2500 --batch_size 4 --task edm --test_epoch 900 > logs/eval/eval_train_prop_pred_4condition_only_lumo_step1_30_5.log 2>&1 &

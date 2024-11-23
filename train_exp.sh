CUDA_VISIBLE_DEVICES=1 python -u main_geom_drugs.py --n_epochs 3000 --exp_name edm_egnn_geom_drugs --n_stability_samples 500 --diffusion_noise_schedule polynomial_2 --diffusion_steps 1000 --diffusion_noise_precision 1e-5 --diffusion_loss_type l2 --batch_size 32 --nf 256 --n_layers 4 --lr 1e-4 --normalize_factors [1,4,10] --test_epochs 1 --ema_decay 0.9999 --normalization_factor 1 --model egnn_dynamics --visualize_every_batch 10000 > edm_egnn_geom_drugs.log 2>&1 &


CUDA_VISIBLE_DEVICES=2 python -u main_pcqm.py --n_epochs 3000 --exp_name edm_egnn_pcq --n_stability_samples 500 --diffusion_noise_schedule polynomial_2 --diffusion_steps 1000 --diffusion_noise_precision 1e-5 --diffusion_loss_type l2 --batch_size 64 --nf 256 --n_layers 4 --lr 1e-4 --normalize_factors [1,4,10] --test_epochs 1 --ema_decay 0.9999 --normalization_factor 1 --model egnn_dynamics --visualize_every_batch 10000 > edm_egnn_pcq.log 2>&1 &


CUDA_VISIBLE_DEVICES=3 python -u main_pcqm.py --n_epochs 3000 --exp_name edm_pcq_torchmdnet --n_stability_samples 500 --diffusion_noise_schedule polynomial_2 --diffusion_steps 1000 --diffusion_noise_precision 1e-5 --diffusion_loss_type l2 --batch_size 64 --nf 256 --n_layers 4 --lr 1e-4 --normalize_factors [1,4,10] --test_epochs 1 --ema_decay 0.9999 --normalization_factor 1 --model torchmdnet --visualize_every_batch 10000 > edm_pcq_torchmdnet.log 2>&1 &



# training only split

CUDA_VISIBLE_DEVICES=0 python -u main_qm9.py --n_epochs 3000 --exp_name DGAP_normal_time_sampling_no_loss --n_stability_samples 1000 --diffusion_noise_schedule polynomial_2 --diffusion_noise_precision 1e-5 --diffusion_steps 1000 --diffusion_loss_type l2 --batch_size 64 --nf 256 --n_layers 9 --lr 1e-4 --normalize_factors \[1,4,10\] --test_epochs 20 --ema_decay 0.9999 --property_pred 1 --prediction_threshold_t 10 --model DGAP --sep_noisy_node 1 --target_property lumo --atom_type_pred 0 --branch_layers_num 4 --use_prop_pred 0 
# training split with atom type prediction

CUDA_VISIBLE_DEVICES=0 python -u main_qm9.py --n_epochs 3000 --exp_name DGAP_normal_time_sampling_no_loss --n_stability_samples 1000 --diffusion_noise_schedule polynomial_2 --diffusion_noise_precision 1e-5 --diffusion_steps 1000 --diffusion_loss_type l2 --batch_size 64 --nf 256 --n_layers 9 --lr 1e-4 --normalize_factors \[1,4,10\] --test_epochs 20 --ema_decay 0.9999 --property_pred 1 --prediction_threshold_t 10 --model DGAP --sep_noisy_node 1 --target_property lumo --atom_type_pred 1 --branch_layers_num 4 --use_prop_pred 0

# training split with atom type prediction and property prediction

CUDA_VISIBLE_DEVICES=0 python -u main_qm9.py --n_epochs 3000 --exp_name DGAP_normal_time_sampling_no_loss --n_stability_samples 1000 --diffusion_noise_schedule polynomial_2 --diffusion_noise_precision 1e-5 --diffusion_steps 1000 --diffusion_loss_type l2 --batch_size 64 --nf 256 --n_layers 9 --lr 1e-4 --normalize_factors \[1,4,10\] --test_epochs 20 --ema_decay 0.9999 --property_pred 1 --prediction_threshold_t 10 --model DGAP --sep_noisy_node 1 --target_property lumo --atom_type_pred 1 --branch_layers_num 4 --use_prop_pred 1



CUDA_VISIBLE_DEVICES=7 python -u main_qm9.py --n_epochs 3000 --exp_name split_k8_t10_with_atom_type_prop_pred_U_add_atom_type_cat --n_stability_samples 1000 --diffusion_noise_schedule polynomial_2 --diffusion_noise_precision 1e-5 --diffusion_steps 1000 --diffusion_loss_type l2 --batch_size 64 --nf 256 --n_layers 9 --lr 1e-4 --normalize_factors \[1,4,10\] --test_epochs 20 --ema_decay 0.9999 --property_pred 1 --prediction_threshold_t 10 --model DGAP --sep_noisy_node 1 --target_property U --atom_type_pred 1 --branch_layers_num 8 --use_prop_pred 1 --atom_type4prop_pred 1 --atom_connect_fn cat  --resume /data/protein/SKData/e3_diffusion_for_molecules/outputs/split_k8_t10_with_atom_type_prop_pred_U_add_atom_type_cat --start_epoch 900 > split_k8_t10_with_atom_type_prop_pred_U_add_atom_type_cat.log  2>&1 &



CUDA_VISIBLE_DEVICES=6 python -u main_qm9.py --n_epochs 3000 --exp_name split_k8_t10_with_atom_type_prop_pred_U_add_atom_type_cat2 --n_stability_samples 1000 --diffusion_noise_schedule polynomial_2 --diffusion_noise_precision 1e-5 --diffusion_steps 1000 --diffusion_loss_type l2 --batch_size 64 --nf 256 --n_layers 9 --lr 1e-4 --normalize_factors \[1,4,10\] --test_epochs 20 --ema_decay 0.9999 --property_pred 1 --prediction_threshold_t 10 --model DGAP --sep_noisy_node 1 --target_property U --atom_type_pred 1 --branch_layers_num 8 --use_prop_pred 1 --atom_type4prop_pred 1 --atom_connect_fn cat  > split_k8_t10_with_atom_type_prop_pred_U_add_atom_type_cat2.log  2>&1 &


CUDA_VISIBLE_DEVICES=5 python main_qm9_prop.py --num_workers 2 --lr 5e-4 --property G --exp_name exp_class_G --model_name egnn --dataset qm9 --atom_disturb 1 > exp_class_G_disturb.log 2>&1 &

CUDA_VISIBLE_DEVICES=1 python main_qm9_prop.py --num_workers 2 --lr 5e-4 --property G --exp_name exp_class_G_disturb_later_fusion_2 --model_name egnn --dataset qm9 --atom_disturb 1 --later_fusion_h 1 > exp_class_G_disturb_later_fusion_2.log 2>&1 &


CUDA_VISIBLE_DEVICES=2 python main_qm9_prop.py --num_workers 2 --lr 5e-4 --property G --exp_name exp_class_G_disturb_later_fusion_2_use_ref --model_name egnn --dataset qm9 --atom_disturb 1 --later_fusion_h 1 --use_ref 1 > exp_class_G_disturb_later_fusion_2_use_ref.log 2>&1 &

CUDA_VISIBLE_DEVICES=3 python main_qm9_prop.py --num_workers 2 --lr 5e-4 --property G --exp_name exp_class_G_disturb_use_ref --model_name egnn --dataset qm9 --atom_disturb 1 --use_ref 1 > exp_class_G_disturb_use_ref.log 2>&1 &



# use ref,
CUDA_VISIBLE_DEVICES=7 python -u main_qm9.py --n_epochs 3000 --exp_name split_k8_t10_with_atom_type_prop_pred_U_add_atom_type_use_ref --n_stability_samples 1000 --diffusion_noise_schedule polynomial_2 --diffusion_noise_precision 1e-5 --diffusion_steps 1000 --diffusion_loss_type l2 --batch_size 64 --nf 256 --n_layers 9 --lr 1e-4 --normalize_factors \[1,4,10\] --test_epochs 20 --ema_decay 0.9999 --property_pred 1 --prediction_threshold_t 10 --model DGAP --sep_noisy_node 1 --target_property U --atom_type_pred 1 --branch_layers_num 8 --use_prop_pred 1 --atom_type4prop_pred 1 --use_ref 1  > split_k8_t10_with_atom_type_prop_pred_U_add_atom_type_use_ref.log  2>&1 &


CUDA_VISIBLE_DEVICES=6 python -u main_qm9.py --n_epochs 3000 --exp_name split_k8_t10_with_atom_type_prop_pred_U0_add_atom_type_use_ref --n_stability_samples 1000 --diffusion_noise_schedule polynomial_2 --diffusion_noise_precision 1e-5 --diffusion_steps 1000 --diffusion_loss_type l2 --batch_size 64 --nf 256 --n_layers 9 --lr 1e-4 --normalize_factors \[1,4,10\] --test_epochs 20 --ema_decay 0.9999 --property_pred 1 --prediction_threshold_t 10 --model DGAP --sep_noisy_node 1 --target_property U0 --atom_type_pred 1 --branch_layers_num 8 --use_prop_pred 1 --atom_type4prop_pred 1 --use_ref 1  > split_k8_t10_with_atom_type_prop_pred_U0_add_atom_type_use_ref.log  2>&1 &

CUDA_VISIBLE_DEVICES=5 python -u main_qm9.py --n_epochs 3000 --exp_name split_k8_t10_with_atom_type_prop_pred_G_add_atom_type_use_ref --n_stability_samples 1000 --diffusion_noise_schedule polynomial_2 --diffusion_noise_precision 1e-5 --diffusion_steps 1000 --diffusion_loss_type l2 --batch_size 64 --nf 256 --n_layers 9 --lr 1e-4 --normalize_factors \[1,4,10\] --test_epochs 20 --ema_decay 0.9999 --property_pred 1 --prediction_threshold_t 10 --model DGAP --sep_noisy_node 1 --target_property G --atom_type_pred 1 --branch_layers_num 8 --use_prop_pred 1 --atom_type4prop_pred 1 --use_ref 1  > split_k8_t10_with_atom_type_prop_pred_G_add_atom_type_use_ref.log  2>&1 &

CUDA_VISIBLE_DEVICES=4 python -u main_qm9.py --n_epochs 3000 --exp_name split_k8_t10_with_atom_type_prop_pred_H_add_atom_type_use_ref --n_stability_samples 1000 --diffusion_noise_schedule polynomial_2 --diffusion_noise_precision 1e-5 --diffusion_steps 1000 --diffusion_loss_type l2 --batch_size 64 --nf 256 --n_layers 9 --lr 1e-4 --normalize_factors \[1,4,10\] --test_epochs 20 --ema_decay 0.9999 --property_pred 1 --prediction_threshold_t 10 --model DGAP --sep_noisy_node 1 --target_property H --atom_type_pred 1 --branch_layers_num 8 --use_prop_pred 1 --atom_type4prop_pred 1 --use_ref 1  > split_k8_t10_with_atom_type_prop_pred_H_add_atom_type_use_ref.log  2>&1 &

CUDA_VISIBLE_DEVICES=7 python -u eval_prop_pred.py --num_workers 2 --lr 5e-4 --property U0 --model_name egnn --generators_path /mnt/nfs-ssd/data/fengshikun/e3_bfn_schedule/e3_diffusion_for_molecules/outputs/split_k8_t10_with_atom_type_prop_pred_U0_add_atom_type_use_ref --model_path generative_model_ema_1180.npy --dataset qm9 > split_k8_t10_with_atom_type_prop_pred_U0_use_ref.log 2>&1 &
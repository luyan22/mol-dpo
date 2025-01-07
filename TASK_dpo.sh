# TODO: TRAIN UNI-GEM CONDITIONAL GENERATION
# ref model training for lumo: context embedding in h
export CUDA_VISIBLE_DEVICES=4
export LD_LIBRARY_PATH=/home/admin01/SKData/miniconda3/envs/MOLuni/lib
# debug : lumo
python -u main_qm9.py --n_epochs 3000 --exp_name context_embedding_cond_gen --n_stability_samples 1000 --diffusion_noise_schedule polynomial_2 --diffusion_noise_precision 1e-5 --diffusion_steps 1000 --diffusion_loss_type l2 --batch_size 64 --nf 256 --n_layers 9 --lr 1e-4 --normalize_factors [1,4,10] --test_epochs 20 --ema_decay 0.9999 --property_pred 1 --prediction_threshold_t 10 --model DGAP --sep_noisy_node 1 --target_property lumo --atom_type_pred 1 --branch_layers_num 8 --use_prop_pred 1 --conditioning lumo --no_wandb

# final : lumo doing on 4, tmux dpo_lumo
python -u main_qm9.py --n_epochs 3000 --exp_name context_embedding_cond_gen_lumo --n_stability_samples 1000 --diffusion_noise_schedule polynomial_2 --diffusion_noise_precision 1e-5 --diffusion_steps 1000 --diffusion_loss_type l2 --batch_size 64 --nf 256 --n_layers 9 --lr 1e-4 --normalize_factors [1,4,10] --test_epochs 20 --ema_decay 0.9999 --property_pred 1 --prediction_threshold_t 10 --model DGAP --sep_noisy_node 1 --target_property lumo --atom_type_pred 1 --branch_layers_num 8 --use_prop_pred 1 --conditioning lumo > logs/context_embed_cond_gen_train/lumo.log 2>&1 &

# TODO: TRAIN EDM CONDITIONAL GENERATION
# debug: lumo
export CUDA_VISIBLE_DEVICES=3
export LD_LIBRARY_PATH=/home/admin01/SKData/miniconda3/envs/MOLuni/lib
python main_qm9.py --exp_name edm_cond_lumo  --model egnn_dynamics --lr 1e-4  --nf 192 --n_layers 9 --save_model True --diffusion_steps 1000 --sin_embedding False --n_epochs 3000 --n_stability_samples 500 --diffusion_noise_schedule polynomial_2 --diffusion_noise_precision 1e-5 --dequantization deterministic --include_charges False --diffusion_loss_type l2 --batch_size 64 --normalize_factors [1,8,1] --conditioning lumo --dataset qm9_second_half --no_wandb --start_epoch 9

# final: lumo  doing on 3, tmux edm_condGen_lumo failed
export CUDA_VISIBLE_DEVICES=3
export LD_LIBRARY_PATH=/home/admin01/SKData/miniconda3/envs/MOLuni/lib
python main_qm9.py --exp_name edm_cond_lumo  --model egnn_dynamics --lr 1e-4  --nf 192 --n_layers 9 --save_model True --diffusion_steps 1000 --sin_embedding False --n_epochs 3000 --n_stability_samples 500 --diffusion_noise_schedule polynomial_2 --diffusion_noise_precision 1e-5 --dequantization deterministic --include_charges False --diffusion_loss_type l2 --batch_size 64 --normalize_factors [1,8,1] --conditioning lumo --dataset qm9_second_half > logs/edm_cond_gen_train/lumo.log 2>&1 &

# final: alpha doing on 3, tmux dpo_alpha
export CUDA_VISIBLE_DEVICES=3
export LD_LIBRARY_PATH=/home/admin01/SKData/miniconda3/envs/MOLuni/lib
python main_qm9.py --exp_name edm_cond_alpha  --model egnn_dynamics --lr 1e-4  --nf 192 --n_layers 9 --save_model True --diffusion_steps 1000 --sin_embedding False --n_epochs 3000 --n_stability_samples 500 --diffusion_noise_schedule polynomial_2 --diffusion_noise_precision 1e-5 --dequantization deterministic --include_charges False --diffusion_loss_type l2 --batch_size 64 --normalize_factors [1,8,1] --conditioning alpha --dataset qm9_second_half > logs/edm_cond_gen_train/alpha.log 2>&1 &

# TODO: FINETUNE DPO 
# TODO:DPO training for lumo | REF: EDM | REWARD: uni_gem

    # TODO debug
export CUDA_VISIBLE_DEVICES=2
export LD_LIBRARY_PATH=/home/admin01/SKData/miniconda3/envs/MOLuni/lib
python -m DPO.main_dpo --ref_fold outputs/edm_cond_lumo --ref_model_path outputs/edm_cond_lumo/generative_model_ema_2990.npy --model_type edm --finetune_fold DPO/finetune/lumo --eval_interval 1 --exp_name DPO_lumo_train --reward_network_type uni_gem --reward_model_path "condGen/1-head model/lumo/generative_model_ema_3000.npy" --reward_fold "condGen/1-head model/lumo"  --beta 1 --lr 0.00001 --debug_mode 1 --n_samples 100 > logs/dpo_train/lumo_minus_1_1e-5.log 2>&1 &

export CUDA_VISIBLE_DEVICES=1 
export LD_LIBRARY_PATH=/home/admin01/SKData/miniconda3/envs/MOLuni/lib
python -m DPO.main_dpo --ref_fold outputs/edm_cond_lumo --ref_model_path outputs/edm_cond_lumo/generative_model_ema_2990.npy --model_type edm --finetune_fold DPO/finetune/lumo --eval_interval 1 --exp_name DPO_lumo_train --reward_network_type uni_gem --reward_model_path "condGen/1-head model/lumo/generative_model_ema_3000.npy" --reward_fold "condGen/1-head model/lumo"  --beta 0.1 --lr 0.00001 --debug_mode 0 --n_samples 100 > logs/dpo_train/lumo_minus_1e-1_1e-5.log 2>&1 &

export CUDA_VISIBLE_DEVICES=2
export LD_LIBRARY_PATH=/home/admin01/SKData/miniconda3/envs/MOLuni/lib
python -m DPO.main_dpo --ref_fold outputs/edm_cond_lumo --ref_model_path outputs/edm_cond_lumo/generative_model_ema_2990.npy --model_type edm --finetune_fold DPO/finetune/lumo --eval_interval 1 --exp_name DPO_lumo_train --reward_network_type uni_gem --reward_model_path "condGen/1-head model/lumo/generative_model_ema_3000.npy" --reward_fold "condGen/1-head model/lumo"  --beta 0.1 --lr 0.00001 --debug_mode 1 --n_samples 100 > logs/dpo_train/lumo_minus_1e-1_1e-5.log 2>&1 &

export CUDA_VISIBLE_DEVICES=2
export LD_LIBRARY_PATH=/home/admin01/SKData/miniconda3/envs/MOLuni/lib
python -m DPO.main_dpo --ref_fold outputs/edm_cond_lumo --ref_model_path outputs/edm_cond_lumo/generative_model_ema_2990.npy --model_type edm --finetune_fold DPO/finetune/lumo --eval_interval 1 --exp_name DPO_lumo_train --reward_network_type uni_gem --reward_model_path "condGen/1-head model/lumo/generative_model_ema_3000.npy" --reward_fold "condGen/1-head model/lumo"  --beta 0.01 --lr 0.00001 --debug_mode 1 --n_samples 100 --reward_func minus > logs/dpo_train/lumo_minus_1e-2_1e-5.log 2>&1 &
 
export CUDA_VISIBLE_DEVICES=2 # on tmux dpo_lumo
export LD_LIBRARY_PATH=/home/admin01/SKData/miniconda3/envs/MOLuni/lib
python -m DPO.main_dpo --ref_fold outputs/edm_cond_lumo --ref_model_path outputs/edm_cond_lumo/generative_model_ema_2990.npy --model_type edm --finetune_fold DPO/finetune/lumo --eval_interval 1 --exp_name DPO_lumo_train --reward_network_type uni_gem --reward_model_path "condGen/1-head model/lumo/generative_model_ema_3000.npy" --reward_fold "condGen/1-head model/lumo"  --beta 1 --lr 0.00001 --debug_mode 0 --n_samples 100 --reward_func exp > logs/dpo_train/lumo_exp_1_1e-5.log 2>&1 &

export CUDA_VISIBLE_DEVICES=4
export LD_LIBRARY_PATH=/home/admin01/SKData/miniconda3/envs/MOLuni/lib
python -m DPO.main_dpo --ref_fold outputs/edm_cond_lumo --ref_model_path outputs/edm_cond_lumo/generative_model_ema_2990.npy --model_type edm --finetune_fold DPO/finetune/lumo --eval_interval 1 --exp_name DPO_lumo_train --reward_network_type uni_gem --reward_model_path "condGen/1-head model/lumo/generative_model_ema_3000.npy" --reward_fold "condGen/1-head model/lumo"  --beta 10 --lr 0.0001 --debug_mode 0 --n_samples 100 --reward_func exp > logs/dpo_train/lumo_exp_10_1e-4.log 2>&1 &

export CUDA_VISIBLE_DEVICES=4
export LD_LIBRARY_PATH=/home/admin01/SKData/miniconda3/envs/MOLuni/lib
python -m DPO.main_dpo --ref_fold outputs/edm_cond_lumo --ref_model_path outputs/edm_cond_lumo/generative_model_ema_2990.npy --model_type edm --finetune_fold DPO/finetune/lumo --eval_interval 1 --exp_name DPO_lumo_train --reward_network_type uni_gem --reward_model_path "condGen/1-head model/lumo/generative_model_ema_3000.npy" --reward_fold "condGen/1-head model/lumo"  --beta 100 --lr 0.0001 --debug_mode 1 --n_samples 100 --reward_func inverse > logs/dpo_train/lumo_inverse_100_1e-4.log 2>&1 &
    # TODO final


# TODO: FINETUNE DPO 
# TODO:DPO training for lumo | REF: EDM | REWARD: egnn
    # TODO dubug
export CUDA_VISIBLE_DEVICES=0
export LD_LIBRARY_PATH=/home/admin01/SKData/miniconda3/envs/MOLuni/lib
python -m DPO.main_dpo --ref_fold outputs/edm_cond_lumo --ref_model_path outputs/edm_cond_lumo/generative_model_ema_2990.npy --model_type edm --finetune_fold DPO/finetune/lumo --eval_interval 1 --exp_name DPO_lumo_train --reward_network_type egnn --reward_model_path "qm9/property_prediction/outputs/exp_class_lumo/best_checkpoint.npy" --reward_fold "qm9/property_prediction/outputs/exp_class_lumo"  --beta 1 --lr 0.00001 --debug_mode 1 --n_samples 100 > logs/dpo_train/lumo_minus_1_1e-5_egnn.log 2>&1 &

export CUDA_VISIBLE_DEVICES=0 # doing on tmux dpo_lumo
export LD_LIBRARY_PATH=/home/admin01/SKData/miniconda3/envs/MOLuni/lib
python -m DPO.main_dpo --ref_fold outputs/edm_cond_lumo --ref_model_path outputs/edm_cond_lumo/generative_model_ema_2990.npy --model_type edm --finetune_fold DPO/finetune/lumo --eval_interval 1 --exp_name DPO_lumo_train --reward_network_type egnn --reward_model_path "qm9/property_prediction/outputs/exp_class_lumo/best_checkpoint.npy" --reward_fold "qm9/property_prediction/outputs/exp_class_lumo"  --beta 0.1 --lr 0.00001 --debug_mode 0 --n_samples 100 > logs/dpo_train/lumo_minus_0.1_1e-5_egnn.log 2>&1 &

# TODO:DPO training for alpha | REF: EDM | REWARD: egnn
    # TODO dubug
export CUDA_VISIBLE_DEVICES=0
export LD_LIBRARY_PATH=/home/admin01/SKData/miniconda3/envs/MOLuni/lib
python -m DPO.main_dpo --ref_fold outputs/edm_cond_alpha --ref_model_path outputs/edm_cond_alpha/generative_model_ema_2990.npy --model_type edm --finetune_fold DPO/finetune/alpha --eval_interval 1 --exp_name DPO_alpha_train --reward_network_type egnn --reward_model_path "qm9/property_prediction/outputs/exp_class_alpha/best_checkpoint.npy" --reward_fold "qm9/property_prediction/outputs/exp_class_alpha"  --beta 1 --lr 0.00001 --debug_mode 1 --n_samples 100 > logs/dpo_train/alpha_minus_1_1e-5_egnn.log 2>&1 &

export CUDA_VISIBLE_DEVICES=3 # doing on tmux dpo_alpha
export LD_LIBRARY_PATH=/home/admin01/SKData/miniconda3/envs/MOLuni/lib
python -m DPO.main_dpo --ref_fold outputs/edm_cond_alpha --ref_model_path outputs/edm_cond_alpha/generative_model_ema_2990.npy --model_type edm --finetune_fold DPO/finetune/alpha --eval_interval 1 --exp_name DPO_alpha_train --reward_network_type egnn --reward_model_path "qm9/property_prediction/outputs/exp_class_alpha/best_checkpoint.npy" --reward_fold "qm9/property_prediction/outputs/exp_class_alpha"  --beta 0.1 --lr 0.00001 --debug_mode 1 --n_samples 100 > logs/dpo_train/alpha_minus_0.1_1e-5_egnn.log 2>&1 &



# TODO: DPO training for lumo | REF: uni_gem | REWARD: uni_gem
# debug: lumo
export CUDA_VISIBLE_DEVICES=5
export LD_LIBRARY_PATH=/home/admin01/SKData/miniconda3/envs/MOLuni/lib
srun --gres=gpu:a100:2 --time 8-12:00:00 python -m DPO.main_dpo --ref_fold outputs/context_embedding_cond_gen --ref_model_path outputs/context_embedding_cond_gen/generative_model_ema_100.npy --model_type uni_gem --finetune_fold DPO/finetune/lumo --exp_name DPO_lumo_train --reward_network_type uni_gem --reward_model_path "condGen/1-head model/lumo/generative_model_ema_3000.npy" --reward_fold "condGen/1-head model/lumo"





# TODO: EVAL EDM BASELINE
# classifier egnn
export CUDA_VISIBLE_DEVICES=4
export LD_LIBRARY_PATH=/home/admin01/SKData/miniconda3/envs/MOLuni/lib
python eval_conditional_qm9.py --generators_path outputs/edm_cond_lumo --classifiers_path qm9/property_prediction/outputs/exp_class_lumo --property lumo  --iterations 100  --batch_size 100 --task edm --test_epoch 2990 --classifier_type egnn > logs/dpo_eval/lumo_edm_baseline.log 2>&1 &

# classifier uni-gem
export CUDA_VISIBLE_DEVICES=4
export LD_LIBRARY_PATH=/home/admin01/SKData/miniconda3/envs/MOLuni/lib
python eval_conditional_qm9.py --generators_path outputs/edm_cond_lumo --classifiers_path "condGen/1-head model/lumo"  --property lumo  --iterations 100  --batch_size 100 --task edm --test_epoch 2990 --classifier_type uni-gem > logs/dpo_eval/lumo_edm_baseline.log 2>&1 &

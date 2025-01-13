# train classifier
cd qm9/property_prediction
srun --gres=gpu:a100:2 --time 8-12:00:00  python main_qm9_prop.py --num_workers 2 --lr 5e-4 --property alpha --exp_name exp_class_alpha_01 --model_name egnn

srun --gres=gpu:a100:2 --time 8-12:00:00  python main_qm9_prop.py --num_workers 2 --lr 5e-4 --property gap --exp_name exp_class_gap --model_name egnn > logs/exp_class_gap.log 2>&1 &


# guidance for alpha
python -u eval_conditional_qm9.py --generators_path outputs/alpha_train_prop_pred4condition_only --classifiers_path qm9/property_prediction/outputs/exp_class_alpha_01 --property alpha --iterations 100 --batch_size 100 --task edm --test_epoch 360 --condGenConfig condGen_config/condGen_config_01.json > logs/condGen/final_test_alpha.log 2>&1 &















# check whether the random sample is crucial for the guidance task
python -u eval_conditional_qm9.py --generators_path outputs/train_prop_pred4condition_only --classifiers_path qm9/property_prediction/outputs/exp_class_lumo --property lumo --iterations 2500 --batch_size 4 --task edm --test_epoch 900 --condGenConfig condGen_config/condGen_config_06.json > logs/condGen/cond_gen_lumo_guide_from_1000.log 2>&1 &

python -u eval_conditional_qm9.py --generators_path outputs/train_prop_pred4condition_only --classifiers_path qm9/property_prediction/outputs/exp_class_lumo --property lumo --iterations 2500 --batch_size 4 --task edm --test_epoch 900 --condGenConfig condGen_config/condGen_config_07.json > logs/condGen/all_scale_1_guide_from_800.log 2>&1 &

python -u eval_conditional_qm9.py --generators_path outputs/train_prop_pred4condition_only --classifiers_path qm9/property_prediction/outputs/exp_class_lumo --property lumo --iterations 2500 --batch_size 4 --task edm --test_epoch 900 --condGenConfig condGen_config/condGen_config_07.json > logs/condGen/all_scale_1_guide_from_800_only_guidance_4_nucleation.log 2>&1 &


# sample z under 1.1 prop pred loss for guidance task
python -u eval_conditional_qm9.py --generators_path outputs/train_prop_pred4condition_only --classifiers_path qm9/property_prediction/outputs/exp_class_lumo --property lumo --iterations 2500 --batch_size 4 --task edm --test_epoch 900 --condGenConfig condGen_config/condGen_config_07.json > logs/condGen/control_sample_z_guide_between_150_800.log 2>&1 &

python -u eval_conditional_qm9.py --generators_path outputs/train_prop_pred4condition_only --classifiers_path qm9/property_prediction/outputs/exp_class_lumo --property lumo --iterations 2500 --batch_size 4 --task edm --test_epoch 900 --condGenConfig condGen_config/condGen_config_07.json > logs/condGen/control_sample_z_guide_between_150_800_loss_decr.log 2>&1 &


python -u eval_conditional_qm9.py --generators_path outputs/train_prop_pred4condition_only --classifiers_path qm9/property_prediction/outputs/exp_class_lumo --property lumo --iterations 2500 --batch_size 4 --task edm --test_epoch 900 --condGenConfig condGen_config/condGen_config_07.json > logs/condGen/800_150_10_0_diff_strategy_guide.log 2>&1 &

python -u eval_conditional_qm9.py --generators_path outputs/train_prop_pred4condition_only --classifiers_path qm9/property_prediction/outputs/exp_class_lumo --property lumo --iterations 2500 --batch_size 4 --task edm --test_epoch 900 --condGenConfig condGen_config/condGen_config_07.json > logs/condGen/under_1.0_loss_800_150_10_0_diff_strategy_guide.log 2>&1 &

python -u eval_conditional_qm9.py --generators_path outputs/train_prop_pred4condition_only --classifiers_path qm9/property_prediction/outputs/exp_class_lumo --property lumo --iterations 2500 --batch_size 4 --task edm --test_epoch 900 --condGenConfig condGen_config/condGen_config_07.json > logs/condGen/under_1.0_loss_800_150_10_0_diff_strategy_guide_0.1.log 2>&1 &

python -u eval_conditional_qm9.py --generators_path outputs/train_prop_pred4condition_only --classifiers_path qm9/property_prediction/outputs/exp_class_lumo --property lumo --iterations 2500 --batch_size 4 --task edm --test_epoch 900 --condGenConfig condGen_config/condGen_config_07.json > logs/condGen/under_1.0_loss_800_150_10_0_diff_strategy_guide_scale_to_0.log 2>&1 &

python -u eval_conditional_qm9.py --generators_path outputs/train_prop_pred4condition_only --classifiers_path qm9/property_prediction/outputs/exp_class_lumo --property lumo --iterations 2500 --batch_size 4 --task edm --test_epoch 900 --condGenConfig condGen_config/condGen_config_07.json > logs/condGen/800_150_10_0_diff_strategy_guide_scale_to_0.log 2>&1 &

python -u eval_conditional_qm9.py --generators_path outputs/train_prop_pred4condition_only --classifiers_path qm9/property_prediction/outputs/exp_class_lumo --property lumo --iterations 2500 --batch_size 4 --task edm --test_epoch 900 --condGenConfig condGen_config/condGen_config_01.json > logs/condGen/under_1.0_loss_800_250_10_0_diff_strategy_guide.log 2>&1 &

python -u eval_conditional_qm9.py --generators_path outputs/train_prop_pred4condition_only --classifiers_path qm9/property_prediction/outputs/exp_class_lumo --property lumo --iterations 2500 --batch_size 4 --task edm --test_epoch 900 --condGenConfig condGen_config/condGen_config_01.json > logs/condGen/debug.log 2>&1 &

python -u eval_conditional_qm9.py --generators_path outputs/train_prop_pred4condition_only --classifiers_path qm9/property_prediction/outputs/exp_class_lumo --property lumo --iterations 2500 --batch_size 4 --task edm --test_epoch 900 --condGenConfig condGen_config/condGen_config_01.json > logs/condGen/debug_vis.log 2>&1 &

# final test
python -u eval_conditional_qm9.py --generators_path outputs/train_prop_pred4condition_only --classifiers_path qm9/property_prediction/outputs/exp_class_lumo --property lumo --iterations 100 --batch_size 100 --task edm --test_epoch 900 --condGenConfig condGen_config/condGen_config_01.json > logs/condGen/final_test_lumo.log 2>&1 &

python -u eval_conditional_qm9.py --generators_path outputs/train_prop_pred4condition_only --classifiers_path qm9/property_prediction/outputs/exp_class_lumo --property lumo --iterations 100 --batch_size 100 --task edm --test_epoch 900 --condGenConfig condGen_config/condGen_config_01.json > logs/condGen/final_test_lumo_2.log 2>&1 &

python -u eval_conditional_qm9.py --generators_path outputs/train_prop_pred4condition_only --classifiers_path qm9/property_prediction/outputs/exp_class_lumo --property lumo --iterations 100 --batch_size 100 --task edm --test_epoch 900 --condGenConfig condGen_config/condGen_config_01.json > logs/condGen/final_test_lumo_3.log 2>&1 &

python -u eval_conditional_qm9.py --generators_path outputs/train_prop_pred4condition_only --classifiers_path qm9/property_prediction/outputs/exp_class_lumo --property lumo --iterations 100 --batch_size 4 --task edm --test_epoch 360 --condGenConfig condGen_config/condGen_config_01.json > logs/condGen/lumo_debug.log 2>&1 &


# validity test lumo debug
python -u eval_conditional_qm9.py --generators_path outputs/train_prop_pred4condition_only --classifiers_path qm9/property_prediction/outputs/exp_class_lumo --property lumo --iterations 4 --batch_size 4 --task edm --test_epoch 940 --condGenConfig condGen_config/condGen_config_01.json --check_stability 1 > logs/condGen/lumo_stablilty_debug.log 2>&1 &


# validity test lumo debug
python -u eval_conditional_qm9.py --generators_path outputs/train_prop_pred4condition_only --classifiers_path qm9/property_prediction/outputs/exp_class_lumo --property lumo --iterations 4 --batch_size 4 --task edm --test_epoch 940 --condGenConfig condGen_config/condGen_config_02.json --check_stability 1 > logs/condGen/lumo_stablilty_debug_2.log 2>&1 &


# validity test lumo no guidance baseline
python -u eval_conditional_qm9.py --generators_path outputs/train_prop_pred4condition_only --classifiers_path qm9/property_prediction/outputs/exp_class_lumo --property lumo --iterations 4 --batch_size 4 --task edm --test_epoch 940 --condGenConfig condGen_config/condGen_config_07.json --check_stability 1 > logs/condGen/lumo_stablilty_debug_base.log 2>&1 &

# validity test lumo final test
python -u eval_conditional_qm9.py --generators_path outputs/train_prop_pred4condition_only --classifiers_path qm9/property_prediction/outputs/exp_class_lumo --property lumo --iterations 100 --batch_size 100 --task edm --test_epoch 940 --condGenConfig condGen_config/condGen_config_01.json --check_stability 1 > logs/condGen/lumo_stablilty_final.log 2>&1 &
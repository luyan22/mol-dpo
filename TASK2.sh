#train the property classifier for evaluation
#重定向输出到logs/TASK2.log文件中
cd qm9/property_prediction
# srun --gres=gpu:a100:2 --time 3-12:00:00 python main_qm9_prop.py --num_workers 2 --lr 5e-4 --property lumo --exp_name exp_class_lumo_22_one_hot --model_name egnn --finetune 1  > /home/AI4Science/luy2402/e3_diffusion_for_molecules/logs/task2.log 2>&1
#

# python main_qm9_prop.py --num_workers 2 --lr 5e-4 --property alpha --exp_name exp_class_alpha_22_one_hot_test --model_name egnn --finetune 1  
import argparse
import torch
from qm9.models import get_optim, get_model
import wandb
import utils
import os
import time
import pickle
from qm9 import dataset
from qm9.models import get_optim, get_model
from configs.datasets_config import get_dataset_info
from DPO.model import DPO
from qm9.property_prediction import main_qm9_prop

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--debug_mode", type=int, default=0, help="Debug mode: no wandb")
    # settings related
    parser.add_argument('--ref_fold', type=str, help="Fole for the ref model, containing the model, optim and the args.")
    parser.add_argument("--ref_args", type=dict, default=None)
    parser.add_argument("--ref_model_path", type=str, default=None, help="Path to the model to be finetuned.")
    parser.add_argument("--finetune_fold", type=str, help="Folder for the finetuned model, containing the model, optim and the args.")
    parser.add_argument("--device", type=str, default="cpu", help="Device to run the code on.")
    parser.add_argument("--eval_interval", type=int, default=20, help="Interval for evaluating the model.")

    # ref model type
    parser.add_argument("--model_type", choices=["edm", "uni_gem"], default="edm")

    # reward related
    parser.add_argument("--reward_network_type", choices=["uni_gem", "egnn"], default="uni_gem", help="Type of reward network to use.")
    parser.add_argument("--reward_model_path", type=str, default=None, help="Path to the model to be used for reward calculation.")
    parser.add_argument("--reward_fold", type=str, default=None, help="Folder for the reward model, containing the model, optim and the args.")
    parser.add_argument("--reward_func", choices=['exp', "minus", "inverse"], default="minus", help="")

    # wandb arguments
    parser.add_argument("--wandb_usr", type=str, default="luyan22", help="Wandb username for logging.") # TODO change to your wandb username
    parser.add_argument("--exp_name", type=str, default="Finetune_lumo", help="Wandb project name for logging.")
    # model arguments
    parser.add_argument("--beta", type=float, default=0.01, help="Beta for the DPO loss.")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate for the finetuning.")
    parser.add_argument("--lr_scheduler", type=str, default="importance_sampling", choices=["importance_sampling", "linear_decay"], help="Learning rate scheduler for the finetuning.")
    
    parser.add_argument("--num_epochs", type=int, default=1000, help="Number of epochs to finetune the model.")
    parser.add_argument("--n_samples", type=int, default=4, help="Number of samples to use for training the DPO model.")

    parser.add_argument("--training_scheduler", type=str, default="increase_t", choices=["random", "increase_t", "decrease_t"], help="Sequence of selecting timestamps")

    args = parser.parse_args()
    return args

def get_args(model_fold): 
    ref_args_path = os.path.join(model_fold, "args.pickle")
    with open(ref_args_path, "rb") as f:
        ref_args = pickle.load(f)
    return ref_args

def print_device_info(device):
    total_memory = torch.cuda.get_device_properties(device).total_memory
    reserved_memory = torch.cuda.memory_reserved(device)
    allocated_memory = torch.cuda.memory_allocated(device)
    free_memory = reserved_memory - allocated_memory
    
    print(f"CUDA 是否可用: {torch.cuda.is_available()}")
    print(f"当前设备索引: {torch.cuda.current_device()}")
    print(f"当前设备名称: {torch.cuda.get_device_name(torch.cuda.current_device())}")

    # 输出显存信息（转换为 MB 或 GB）
    print(f"总显存: {total_memory / 1024**2:.2f} MB")
    print(f"已分配显存: {allocated_memory / 1024**2:.2f} MB")
    print(f"已保留显存: {reserved_memory / 1024**2:.2f} MB")
    print(f"可用显存: {free_memory / 1024**2:.2f} MB")

def main():
    args = get_parser()
    print(f"Args: {args}")
    ref_args = get_args(args.ref_fold)
    args.ref_args = ref_args # log the ref_args for in wandb
    print(f"Ref args: {ref_args}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)
    dtype = torch.float32
    if args.wandb_usr is not None and args.debug_mode == 0:
        print(f"Using wandb for logging with username {args.wandb_usr}.")
        args.wandb_usr = utils.get_wandb_username(args.wandb_usr)
        exp_name = args.exp_name + "_" + args.reward_func + "_" + str(args.beta) + "_" + str(args.lr) + time.strftime("_%Y%m%d_%H%M%S")
        kwargs = {"entity": args.wandb_usr, "project": "DPO", "name": exp_name, "config": args}
        wandb.init(**kwargs)
        wandb.save("*.txt")
    dataset_info = get_dataset_info(ref_args.dataset, ref_args.remove_h, finetune=ref_args.finetune)
    print(f"Dataset info: {dataset_info}")
    # dataloaders to get mean and mad for property normalization
    dataloaders, charge_scale = dataset.retrieve_dataloaders(ref_args)

    if args.model_type == "uni_gem":
        ref_args.conditioning = [ref_args.target_property]
    
    # load the property prediction model for reward calculation(uni_gem)
    reward_args = get_args(args.reward_fold)
    if args.reward_network_type == "uni_gem":
        reward_model, _, _ = get_model(reward_args, device, dataset_info, dataloaders['train'], uni_diffusion=0, use_basis=reward_args.use_basis if args.reward_network_type == "uni_gem" else 0, 
        decoupling=reward_args.decoupling if args.reward_network_type == "uni_gem" else 0, 
        finetune=reward_args.finetune if args.reward_network_type == "uni_gem" else 0)
    elif args.reward_network_type == "egnn":
        reward_model = main_qm9_prop.get_model(reward_args)
    print(f"reward_args: {reward_args}")
    
    reward_model_path = args.reward_model_path if args.reward_model_path is not None else os.path.join(args.reward_fold, "generative_model_ema.npy")
    reward_model.load_state_dict(torch.load(reward_model_path, map_location=device), strict=True)
    reward_model.to(device)
    reward_model.eval()
    print(f"Loaded reward model from {reward_model_path}")

    print_device_info(device)

    # prop_dist.set_normalizer在DPO模型的初始化中完成

    model, nodes_dist, prop_dist = get_model(ref_args, device, dataset_info, dataloaders['train'], uni_diffusion=ref_args.uni_diffusion, use_basis=ref_args.use_basis, decoupling=ref_args.decoupling, finetune=ref_args.finetune)

    ref_model_path = args.ref_model_path if args.ref_model_path is not None else os.path.join(args.ref_fold, "generative_model_ema.npy")
    flow_state_dict = torch.load(ref_model_path, map_location=device)
    # 打印失配的参数
    model.load_state_dict(flow_state_dict)
    print(f"Loaded ref model from {ref_model_path}")
    # 检查哪些参数没有找到
    missing_keys = []
    unexpected_keys = []
    for key in flow_state_dict.keys():
        if key not in model.state_dict():
            missing_keys.append(key)

    # 检查哪些参数不应该存在
    for key in model.state_dict().keys():
        if key not in flow_state_dict:
            unexpected_keys.append(key)

    # 输出结果
    if missing_keys:
        print("Missing keys: ", missing_keys)

    if unexpected_keys:
        print("Unexpected keys: ", unexpected_keys)


    # optim = get_optim(ref_args, model)
    # optim_state_dict = torch.load(os.path.join(args.ref_fold, 'optim.npy'))
    # optim.load_state_dict(optim_state_dict)
    # optim = torch.optim.Adam(model.parameters(), lr=args.lr)
    optim = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9)
    model.to(device)


    dpo_model = DPO(model_ref=model, ref_args=ref_args, model_reward=reward_model, beta=args.beta, lr=args.lr, optim=optim, num_epochs=args.num_epochs, nodes_dist=nodes_dist, dataset_info=dataset_info, device=device, dataloaders=dataloaders, prop_dist=prop_dist, ref_model_type=args.model_type, reward_func=args.reward_func, reward_network_type=args.reward_network_type, lr_scheduler=args.lr_scheduler)

    dpo_model.train(n_samples=args.n_samples, wandb=wandb if args.wandb_usr is not None and args.debug_mode == 0 else None, 
                    eval_interval=args.eval_interval, training_scheduler=args.training_scheduler)
    
if __name__ == '__main__':
    main()
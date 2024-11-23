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

def get_parser():
    parser = argparse.ArgumentParser()
    # settings related
    parser.add_argument('--ref_fold', type=str, help="Fole for the ref model, containing the model, optim and the args.")
    parser.add_argument("--model_path", type=str, default=None, help="Path to the model to be finetuned.")
    parser.add_argument("--finetune_fold", type=str, help="Folder for the finetuned model, containing the model, optim and the args.")
    parser.add_argument("--device", type=str, default="cpu", help="Device to run the code on.")
    parser.add_argument("--ref_args", type=dict, default=None)
    # wandb arguments
    parser.add_argument("--wandb_usr", type=str, default=None, help="Wandb username for logging.")
    parser.add_argument("--exp_name", type=str, default=None, help="Wandb project name for logging.")
    # model arguments
    parser.add_argument("--beta", type=float, default=0.01, help="Beta for the DPO loss.")
    parser.add_argument("--num_epochs", type=int, default=1000, help="Number of epochs to finetune the model.")
    parser.add_argument("--n_samples", type=int, default=4, help="Number of samples to use for training the DPO model.")

    args = parser.parse_args()
    return args

def get_ref_args(model_fold): 
    ref_args_path = os.path.join(model_fold, "args.pickle")
    with open(ref_args_path, "rb") as f:
        ref_args = pickle.load(f)
    return ref_args

def main():
    args = get_parser()
    ref_args = get_ref_args(args.ref_fold)
    args.ref_args = ref_args # log the ref_args for in wandb
    print(f"Ref args: {ref_args}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)
    dtype = torch.float32
    if args.wandb_usr is not None:
        print(f"Using wandb for logging with username {args.wandb_usr}.")
        args.wandb_usr = utils.get_wandb_username(args.wandb_usr)
        exp_name = args.exp_name + time.strftime("%Y%m%d_%H%M%S")
        kwargs = {"entity": args.wandb_usr, "project": "DPO", "name": exp_name, "config": args}
        wandb.init(**kwargs)
        wandb.save("*.txt")
    dataset_info = get_dataset_info(ref_args.dataset, ref_args.remove_h, finetune=ref_args.finetune)
    print(f"Dataset info: {dataset_info}")
    # dataloaders to get mean and mad for property normalization
    dataloaders, charge_scale = dataset.retrieve_dataloaders(ref_args)

    ref_args.conditioning = [ref_args.target_property]
    model, nodes_dist, prop_dist = get_model(ref_args, device, dataset_info, dataloaders['train'], uni_diffusion=ref_args.uni_diffusion, use_basis=ref_args.use_basis, decoupling=ref_args.decoupling, finetune=ref_args.finetune)

    # prop_dist.set_normalizer在DPO模型的初始化中完成


    model_path = args.model_path if args.model_path is not None else os.path.join(args.ref_fold, "generative_model_ema.npy")
    flow_state_dict = torch.load(model_path, map_location=device)
    # 打印失配的参数
    model.load_state_dict(flow_state_dict)
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


    optim = get_optim(ref_args, model)
    optim_state_dict = torch.load(os.path.join(args.ref_fold, 'optim.npy'))
    optim.load_state_dict(optim_state_dict)
    model.to(device)


    dpo_model = DPO(model_ref=model, ref_args=ref_args, beta=args.beta, optim=optim, num_epochs=args.num_epochs, nodes_dist=nodes_dist, dataset_info=dataset_info, device=device, dataloaders=dataloaders, prop_dist=prop_dist)


    dpo_model.train(n_samples=args.n_samples)
    
if __name__ == '__main__':
    main()
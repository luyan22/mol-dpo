# Rdkit import should be first, do not move it
try:
    from rdkit import Chem
except ModuleNotFoundError:
    pass
import build_geom_dataset
from configs.datasets_config import pcq_with_h
import copy
import utils
import argparse
import wandb
from os.path import join
from qm9.models import get_optim, get_model
from equivariant_diffusion import en_diffusion

from equivariant_diffusion import utils as diffusion_utils
import torch
import time
import pickle
from torch.utils.data import Subset

from qm9.utils import prepare_context, compute_mean_mad
import train_test
from train_test import evaluate_properties

parser = argparse.ArgumentParser(description='e3_diffusion')
parser.add_argument('--exp_name', type=str, default='debug_10')
parser.add_argument('--model', type=str, default='egnn_dynamics',
                    help='our_dynamics | schnet | simple_dynamics | '
                         'kernel_dynamics | egnn_dynamics |gnn_dynamics')
parser.add_argument('--probabilistic_model', type=str, default='diffusion',
                    help='diffusion')

# Training complexity is O(1) (unaffected), but sampling complexity O(steps).
parser.add_argument('--diffusion_steps', type=int, default=500)
parser.add_argument('--diffusion_noise_schedule', type=str, default='polynomial_2',
                    help='learned, cosine')
parser.add_argument('--diffusion_loss_type', type=str, default='l2',
                    help='vlb, l2')
parser.add_argument('--diffusion_noise_precision', type=float, default=1e-5)

parser.add_argument('--n_epochs', type=int, default=10000)
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--lr', type=float, default=5e-5)
parser.add_argument('--break_train_epoch', type=eval, default=False,
                    help='True | False')
parser.add_argument('--dp', type=eval, default=True,
                    help='True | False')
parser.add_argument('--condition_time', type=eval, default=True,
                    help='True | False')
parser.add_argument('--clip_grad', type=eval, default=True,
                    help='True | False')
parser.add_argument('--trace', type=str, default='hutch',
                    help='hutch | exact')
# EGNN args -->
parser.add_argument('--n_layers', type=int, default=6,
                    help='number of layers')
parser.add_argument('--inv_sublayers', type=int, default=1,
                    help='number of layers')
parser.add_argument('--nf', type=int, default=192,
                    help='number of layers')
parser.add_argument('--tanh', type=eval, default=True,
                    help='use tanh in the coord_mlp')
parser.add_argument('--attention', type=eval, default=True,
                    help='use attention in the EGNN')
parser.add_argument('--norm_constant', type=float, default=1,
                    help='diff/(|diff| + norm_constant)')
parser.add_argument('--sin_embedding', type=eval, default=False,
                    help='whether using or not the sin embedding')
# <-- EGNN args
parser.add_argument('--ode_regularization', type=float, default=1e-3)
parser.add_argument('--dataset', type=str, default='geom',
                    help='dataset name')
parser.add_argument('--filter_n_atoms', type=int, default=None,
                    help='When set to an integer value, QM9 will only contain molecules of that amount of atoms')
parser.add_argument('--dequantization', type=str, default='argmax_variational',
                    help='uniform | variational | argmax_variational | deterministic')
parser.add_argument('--n_report_steps', type=int, default=50)
parser.add_argument('--wandb_usr', type=str)
parser.add_argument('--no_wandb', action='store_true', help='Disable wandb')
parser.add_argument('--online', type=bool, default=True, help='True = wandb online -- False = wandb offline')
parser.add_argument('--no-cuda', action='store_true', default=False, help='disable CUDA training')
parser.add_argument('--save_model', type=eval, default=True, help='save model')
parser.add_argument('--generate_epochs', type=int, default=1)
parser.add_argument('--num_workers', type=int, default=0,
                    help='Number of worker for the dataloader')
parser.add_argument('--test_epochs', type=int, default=1)
parser.add_argument('--data_augmentation', type=eval, default=False,
                    help='use attention in the EGNN')
parser.add_argument("--conditioning", nargs='+', default=[],
                    help='multiple arguments can be passed, '
                         'including: homo | onehot | lumo | num_atoms | etc. '
                         'usage: "--conditioning H_thermo homo onehot H_thermo"')
parser.add_argument('--resume', type=str, default=None,
                    help='')
parser.add_argument('--start_epoch', type=int, default=0,
                    help='')
parser.add_argument('--ema_decay', type=float, default=0,           # TODO
                    help='Amount of EMA decay, 0 means off. A reasonable value'
                         ' is 0.999.')
parser.add_argument('--augment_noise', type=float, default=0)
parser.add_argument('--n_stability_samples', type=int, default=20,
                    help='Number of samples to compute the stability')
parser.add_argument('--normalize_factors', type=eval, default=[1, 4, 10],
                    help='normalize factors for [x, categorical, integer]')
parser.add_argument('--remove_h', action='store_true')
parser.add_argument('--include_charges', type=eval, default=False, help='include atom charge or not')
parser.add_argument('--visualize_every_batch', type=int, default=5000)
parser.add_argument('--normalization_factor', type=float,
                    default=100, help="Normalize the sum aggregation of EGNN")
parser.add_argument('--aggregation_method', type=str, default='sum',
                    help='"sum" or "mean" aggregation for the graph network')
parser.add_argument('--filter_molecule_size', type=int, default=None,
                    help="Only use molecules below this size.")
parser.add_argument('--sequential', action='store_true',
                    help='Organize data by size to reduce average memory usage.')

parser.add_argument('--uni_diffusion', type=int, default=0,
                    help='whether use uni diffusion of the diffusion steps')


parser.add_argument('--expand_diff', type=int, default=0,
                    help='whether expand the diffusion steps')

parser.add_argument('--only_sample', type=int, default=0, help='only sample the diffusion steps')


parser.add_argument('--denoise_pretrain', type=int, default=0, help='pretrain the model only using the denoise')
args = parser.parse_args()

data_file = './data/geom/geom_drugs_30.npy'
data_file = '/data/protein/SKData/Denoise_Data/GEOM-DRUGS/geom_drugs_30.npy'

if args.remove_h:
    raise NotImplementedError()
else:
    dataset_info = pcq_with_h

args.cuda = not args.no_cuda and torch.cuda.is_available()
device = torch.device("cuda" if args.cuda else "cpu")
dtype = torch.float32


transform = build_geom_dataset.GeomDrugsTransform(dataset_info, args.include_charges, torch.device("cpu"), args.sequential)



# build pcq dataset
pcq_data = build_geom_dataset.PCQM4MV2_Dihedral2(
    root='/nfs/SKData/DenoisingData/pcq', sdf_path='mol_iter_all.pickle', dihedral_angle_noise_scale=2, position_noise_scale=0.04, composition=True, decay=False, decay_coe=0.2, pre_transform=None, addh=True, new_transform=transform,
)


pcq_length = len(pcq_data)
# Set random seed for reproducibility
torch.manual_seed(42)

# Define the size of each dataset split
total_size = len(pcq_data)
train_size = int(0.99 * total_size)
val_size = test_size = (total_size - train_size) // 2

# Create indices for each split
indices = torch.randperm(total_size)
train_indices = indices[:train_size]
val_indices = indices[train_size:train_size + val_size]
test_indices = indices[train_size + val_size:]

# Define a function to split dataset into subdatasets
def split_dataset(dataset, indices):
    return torch.utils.data.Subset(dataset, indices)

# Create subsets for train, validation, and test sets
train_dataset = split_dataset(pcq_data, train_indices)
val_dataset = split_dataset(pcq_data, val_indices)
test_dataset = split_dataset(pcq_data, test_indices)



# test code
print(pcq_data[torch.tensor(0)])

# split_data = build_geom_dataset.load_split_data(data_file, val_proportion=0.1, test_proportion=0.1, filter_size=args.filter_molecule_size)
dataloaders = {}
split_data = [train_dataset, val_dataset, test_dataset]

# for key, data_list in zip(['train', 'val', 'test'], split_data):
for key, dataset in zip(['train', 'val', 'test'], split_data):
    # dataset = build_geom_dataset.GeomDrugsDataset(data_list, transform=transform)
    shuffle = (key == 'train') and not args.sequential

    # Sequential dataloading disabled for now.
    dataloaders[key] = build_geom_dataset.GeomDrugsDataLoader(
        sequential=args.sequential, dataset=dataset, batch_size=args.batch_size,
        shuffle=shuffle, number_workers=args.num_workers)
del split_data

atom_encoder = dataset_info['atom_encoder']
atom_decoder = dataset_info['atom_decoder']

# args, unparsed_args = parser.parse_known_args()
args.wandb_usr = utils.get_wandb_username(args.wandb_usr)

if args.resume is not None:
    exp_name = args.exp_name + '_resume'
    start_epoch = args.start_epoch
    resume = args.resume
    wandb_usr = args.wandb_usr

    only_sample = args.only_sample
    with open(join(args.resume, 'args.pickle'), 'rb') as f:
        args = pickle.load(f)
    
    args.only_sample = only_sample
    args.resume = resume
    args.break_train_epoch = False
    args.exp_name = exp_name
    args.start_epoch = start_epoch
    args.wandb_usr = wandb_usr
    print(args)

utils.create_folders(args)
print(args)

# Wandb config
if args.no_wandb:
    mode = 'disabled'
else:
    mode = 'online' if args.online else 'offline'
kwargs = {'entity': args.wandb_usr, 'name': args.exp_name, 'project': 'e3_diffusion_geom', 'config': args,
          'settings': wandb.Settings(_disable_stats=True), 'reinit': True, 'mode': mode}
wandb.init(**kwargs)
wandb.save('*.txt')

data_dummy = next(iter(dataloaders['train']))


if len(args.conditioning) > 0:
    print(f'Conditioning on {args.conditioning}')
    property_norms = compute_mean_mad(dataloaders, args.conditioning)
    context_dummy = prepare_context(args.conditioning, data_dummy, property_norms)
    context_node_nf = context_dummy.size(2)
else:
    context_node_nf = 0
    property_norms = None

args.context_node_nf = context_node_nf


# Create EGNN flow
model, nodes_dist, prop_dist = get_model(args, device, dataset_info, dataloader_train=dataloaders['train'], uni_diffusion=args.uni_diffusion, pretrain=True)
model = model.to(device)
optim = get_optim(args, model)
# print(model)


gradnorm_queue = utils.Queue()
gradnorm_queue.add(3000)  # Add large value that will be flushed.


def main():
    if args.resume is not None:
        # flow_state_dict = torch.load(join(args.resume, 'flow.npy'))
        # dequantizer_state_dict = torch.load(join(args.resume, 'dequantizer.npy'))
        flow_state_dict = torch.load(join(args.resume, f'generative_model_{args.start_epoch}.npy'))
        optim_state_dict = torch.load(join(args.resume, f'optim_{args.start_epoch}.npy'))
        model.load_state_dict(flow_state_dict)
        optim.load_state_dict(optim_state_dict)

    # Initialize dataparallel if enabled and possible.
    if args.dp and torch.cuda.device_count() > 1 and args.cuda:
        print(f'Training using {torch.cuda.device_count()} GPUs')
        model_dp = torch.nn.DataParallel(model.cpu())
        model_dp = model_dp.cuda()
    else:
        model_dp = model

    # Initialize model copy for exponential moving average of params.
    if args.ema_decay > 0:
        model_ema = copy.deepcopy(model)
        ema = diffusion_utils.EMA(args.ema_decay)

        if args.dp and torch.cuda.device_count() > 1:
            model_ema_dp = torch.nn.DataParallel(model_ema)
        else:
            model_ema_dp = model_ema
    else:
        ema = None
        model_ema = model
        model_ema_dp = model_dp

    best_nll_val = 1e8
    best_nll_test = 1e8
    
    
    if args.only_sample:
        epoch = 0
        train_test.only_sample_chains(args, dataloaders['train'], epoch, model, model_dp, model_ema, ema, device, dtype,
                               property_norms, optim, nodes_dist, gradnorm_queue, dataset_info,
                               prop_dist, uni_diffusion=args.uni_diffusion)
        return
    
    
    for epoch in range(args.start_epoch, args.n_epochs):
        start_epoch = time.time()
        
        # for test
        # evaluate_properties(args=args, loader=dataloaders['val'], epoch=epoch, eval_model=model_ema_dp,partition='Val', device=device, dtype=dtype, nodes_dist=nodes_dist, property_norms=property_norms, wandb=wandb)
        # train_test.analyze_and_save(epoch, model_ema, nodes_dist, args, device,
        #                                     dataset_info, prop_dist, n_samples=args.n_stability_samples)
        # nll_val = train_test.test(args, dataloaders['val'], epoch, model_ema_dp, device, dtype,
        #                               property_norms, nodes_dist, partition='Val', uni_diffusion=args.uni_diffusion)
        train_test.train_epoch(args, dataloaders['train'], epoch, model, model_dp, model_ema, ema, device, dtype,
                               property_norms, optim, nodes_dist, gradnorm_queue, dataset_info,
                               prop_dist, uni_diffusion=args.uni_diffusion)
        print(f"Epoch took {time.time() - start_epoch:.1f} seconds.")

        if epoch % args.test_epochs == 0:
            
            if args.uni_diffusion:
                # evaluate properties
                evaluate_properties(args=args, loader=dataloaders['val'], epoch=epoch, eval_model=model_ema_dp,partition='Val', device=device, dtype=dtype, nodes_dist=nodes_dist, property_norms=property_norms, wandb=wandb)
            
            if isinstance(model, en_diffusion.EnVariationalDiffusion):
                wandb.log(model.log_info(), commit=True)

            if not args.break_train_epoch:
                train_test.analyze_and_save(epoch, model_ema, nodes_dist, args, device,
                                            dataset_info, prop_dist, n_samples=args.n_stability_samples)
            nll_val = train_test.test(args, dataloaders['val'], epoch, model_ema_dp, device, dtype,
                                      property_norms, nodes_dist, partition='Val', uni_diffusion=args.uni_diffusion)
            nll_test = train_test.test(args, dataloaders['test'], epoch, model_ema_dp, device, dtype,
                                       property_norms, nodes_dist, partition='Test', uni_diffusion=args.uni_diffusion)

            if nll_val < best_nll_val:
                best_nll_val = nll_val
                best_nll_test = nll_test
                if args.save_model:
                    args.current_epoch = epoch + 1
                    utils.save_model(optim, 'outputs/%s/optim.npy' % args.exp_name)
                    utils.save_model(model, 'outputs/%s/generative_model.npy' % args.exp_name)
                    if args.ema_decay > 0:
                        utils.save_model(model_ema, 'outputs/%s/generative_model_ema.npy' % args.exp_name)
                    with open('outputs/%s/args.pickle' % args.exp_name, 'wb') as f:
                        pickle.dump(args, f)

            if args.save_model:
                utils.save_model(optim, 'outputs/%s/optim_%d.npy' % (args.exp_name, epoch))
                utils.save_model(model, 'outputs/%s/generative_model_%d.npy' % (args.exp_name, epoch))
                if args.ema_decay > 0:
                    utils.save_model(model_ema, 'outputs/%s/generative_model_ema_%d.npy' % (args.exp_name, epoch))
                with open('outputs/%s/args_%d.pickle' % (args.exp_name, epoch), 'wb') as f:
                    pickle.dump(args, f)
            print('Val loss: %.4f \t Test loss:  %.4f' % (nll_val, nll_test))
            print('Best val loss: %.4f \t Best test loss:  %.4f' % (best_nll_val, best_nll_test))
            wandb.log({"Val loss ": nll_val}, commit=True)
            wandb.log({"Test loss ": nll_test}, commit=True)
            wandb.log({"Best cross-validated test loss ": best_nll_test}, commit=True)


if __name__ == "__main__":
    main()

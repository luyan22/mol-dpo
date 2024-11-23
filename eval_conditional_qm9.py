import argparse
from os.path import join
import torch
import pickle
from qm9.models import get_model
from configs.datasets_config import get_dataset_info
from qm9 import dataset
from qm9.utils import compute_mean_mad
from qm9.sampling import sample
from qm9.property_prediction.main_qm9_prop import test
from qm9.property_prediction import main_qm9_prop
from qm9.sampling import sample_chain, sample, sample_sweep_conditional
import qm9.visualizer as vis
import json
from qm9.analyze import check_stability


def get_classifier(dir_path='', device='cpu'):
    with open(join(dir_path, 'args.pickle'), 'rb') as f:
        args_classifier = pickle.load(f)
    args_classifier.device = device
    args_classifier.model_name = 'egnn'
    classifier = main_qm9_prop.get_model(args_classifier)
    classifier_state_dict = torch.load(join(dir_path, 'best_checkpoint.npy'), map_location=torch.device('cpu'))
    classifier.load_state_dict(classifier_state_dict)

    return classifier


def get_args_gen(dir_path):
    with open(join(dir_path, 'args.pickle'), 'rb') as f:
        args_gen = pickle.load(f)
    # print("args_gen.dataset: ", args_gen.dataset)
    if args_gen.property_pred == 0:
        assert args_gen.dataset == 'qm9_second_half'

    # Add missing args!
    if not hasattr(args_gen, 'normalization_factor'):
        args_gen.normalization_factor = 1
    if not hasattr(args_gen, 'aggregation_method'):
        args_gen.aggregation_method = 'sum'
    return args_gen


def get_generator(dir_path, dataloaders, device, args_gen, property_norms):
    dataset_info = get_dataset_info(args_gen.dataset, args_gen.remove_h)
    model, nodes_dist, prop_dist = get_model(args_gen, device, dataset_info, dataloaders['train'], finetune=args_gen.finetune)
    fn = 'generative_model_ema.npy' if args_gen.ema_decay > 0 else 'generative_model.npy'
    # model_state_dict = torch.load(join(dir_path, fn), map_location='cpu')
    if args.test_epoch > 0:
        load_model_path = join(dir_path, f'generative_model_ema_{args.test_epoch}.npy')
    else:
        load_model_path = join(dir_path, fn)
    
    model_state_dict = torch.load(load_model_path, map_location=device)
    model.load_state_dict(model_state_dict, strict=True)
    print("Generator loaded from: ", load_model_path)
    # print("load_model_parameters: ", model_state_dict.keys())

    # The following function be computes the normalization parameters using the 'valid' partition
    # print("prop_dist probs of 19:", prop_dist.distributions["alpha"][19]["probs"])
    # print("prop_dist params of 19:", prop_dist.distributions["alpha"][19]["params"])
    if prop_dist is not None:
        prop_dist.set_normalizer(property_norms)
    return model.to(device), nodes_dist, prop_dist, dataset_info


def get_dataloader(args_gen):
    dataloaders, charge_scale = dataset.retrieve_dataloaders(args_gen)
    return dataloaders


def analyze_stability_for_genmol(one_hot, x, node_mask, dataset_info):
    atomsxmol = torch.sum(node_mask, dim=1)
    n_samples = len(x)

    molecule_stable = 0
    nr_stable_bonds = 0
    n_atoms = 0

    processed_list = []

    for i in range(n_samples):
        atom_type = one_hot[i].argmax(1).cpu().detach()
        pos = x[i].cpu().detach()

        atom_type = atom_type[0:int(atomsxmol[i])]
        pos = pos[0:int(atomsxmol[i])]
        processed_list.append((pos, atom_type))
    
    mol_stable_lst = []
    for mol in processed_list:
        pos, atom_type = mol
        validity_results = check_stability(pos, atom_type, dataset_info)

        molecule_stable = int(validity_results[0])
        mol_stable_lst.append(molecule_stable)
    
    return mol_stable_lst


class DiffusionDataloader:
    def __init__(self, args_gen, model, nodes_dist, prop_dist, device, unkown_labels=False,
                 batch_size=1, iterations=200, check_stability=False):
        self.args_gen = args_gen
        self.model = model
        self.nodes_dist = nodes_dist
        self.prop_dist = prop_dist
        self.batch_size = batch_size
        self.iterations = iterations
        self.device = device
        self.unkown_labels = unkown_labels
        self.dataset_info = get_dataset_info(self.args_gen.dataset, self.args_gen.remove_h)
        self.i = 0
        self.check_stability = check_stability

    def __iter__(self):
        return self

    def sample(self):
        nodesxsample = self.nodes_dist.sample(self.batch_size)
        if self.args_gen.property_pred == 0:
            context = self.prop_dist.sample_batch(nodesxsample).to(self.device)
        else:
            context = None
        
        mean = self.prop_dist.normalizer[self.args_gen.target_property]['mean']
        mad = self.prop_dist.normalizer[self.args_gen.target_property]['mad']

        if self.args_gen.property_pred and self.args_gen.branch_layers_num > 0:
            pseudo_context = self.prop_dist.sample_batch(nodesxsample).to(self.device)
            # for lumo, need recover to the unnormalized value
            pseudo_context = pseudo_context * self.prop_dist.normalizer[self.args_gen.target_property]['mad'] + self.prop_dist.normalizer[self.args_gen.target_property]['mean']
        else:
            pseudo_context = None
        
        if self.args_gen.property_pred:
            one_hot, charges, x, node_mask, pred = sample(self.args_gen, self.device, self.model,
                                                self.dataset_info, self.prop_dist, nodesxsample=nodesxsample,
                                                context=context, pseudo_context=pseudo_context, mean=mean, mad=mad)
        else:
            one_hot, charges, x, node_mask = sample(self.args_gen, self.device, self.model,
                                                self.dataset_info, self.prop_dist, nodesxsample=nodesxsample,
                                                context=context)
        
        node_mask = node_mask.squeeze(2)
        if self.args_gen.property_pred == 0:
            context = context.squeeze(1)

        # edge_mask
        bs, n_nodes = node_mask.size()
        edge_mask = node_mask.unsqueeze(1) * node_mask.unsqueeze(2)
        diag_mask = ~torch.eye(edge_mask.size(1), dtype=torch.bool).unsqueeze(0)
        diag_mask = diag_mask.to(self.device)
        edge_mask *= diag_mask
        edge_mask = edge_mask.view(bs * n_nodes * n_nodes, 1)
        
        if self.args_gen.property_pred == 0:
            prop_key = self.prop_dist.properties[0]
        else: # in DGAP
            prop_key = "DGAP_unused_key"
        if self.unkown_labels and self.args_gen.property_pred == 0:
            context[:] = self.prop_dist.normalizer[prop_key]['mean']
        elif self.args_gen.property_pred == 0:
            context = context * self.prop_dist.normalizer[prop_key]['mad'] + self.prop_dist.normalizer[prop_key]['mean']
        
        if self.args_gen.property_pred:
            
            data = {
                'positions': x.detach(),
                'atom_mask': node_mask.detach(),
                'edge_mask': edge_mask.detach(),
                'one_hot': one_hot.detach(),
                # prop_key: context.detach(),
                self.args_gen.target_property: pseudo_context,
            }
            # print pesude_context and pred
            pseudo_context = pseudo_context.squeeze(1)
            print("pseudo_context:", pseudo_context)
            print("pred:", pred)
            mae = torch.mean(torch.abs(pseudo_context - pred))
            print("mae:", mae)
        else:
            data = {
                'positions': x.detach(),
                'atom_mask': node_mask.detach(),
                'edge_mask': edge_mask.detach(),
                'one_hot': one_hot.detach(),
                prop_key: context.detach() if not self.args_gen.property_pred else None,
                self.args_gen.target_property: pred if self.args_gen.property_pred else None,
            }

        mol_stable_lst = analyze_stability_for_genmol(one_hot, x, node_mask, self.dataset_info)

        if self.check_stability == 1:
            data["stability"] = mol_stable_lst
        # print("data in diffusion dataloader: ", data.keys())
        # print the size for items in data
        # for k, v in data.items():
        #     if v is not None:
        #         print(k, v.size())
        
        return data

    def __next__(self):
        if self.i <= self.iterations:
            self.i += 1
            return self.sample()
        else:
            self.i = 0
            raise StopIteration

    def __len__(self):
        return self.iterations


def main_quantitative(args):
    # Get classifier
    #if args.task == "numnodes":
    #    class_dir = args.classifiers_path[:-6] + "numnodes_%s" % args.property
    #else:
    class_dir = args.classifiers_path
    classifier = get_classifier(class_dir).to(args.device)

    # Get generator and dataloader used to train the generator and evalute the classifier
    args_gen = get_args_gen(args.generators_path)
    print("args_gen:", args_gen)

    # Careful with this -->
    if not hasattr(args_gen, 'diffusion_noise_precision'):
        args_gen.normalization_factor = 1e-4
    if not hasattr(args_gen, 'normalization_factor'):
        args_gen.normalization_factor = 1
    if not hasattr(args_gen, 'aggregation_method'):
        args_gen.aggregation_method = 'sum'

    dataloaders = get_dataloader(args_gen)
    
    if not len(args_gen.conditioning):
        args_gen.conditioning = [args.property]
    
    print("conditioning: ", args_gen.conditioning)
    property_norms = compute_mean_mad(dataloaders, args_gen.conditioning, args_gen.dataset)
    model, nodes_dist, prop_dist, _ = get_generator(args.generators_path, dataloaders,
                                                    args.device, args_gen, property_norms)
    if args.condGenConfig is not None: # 加入guidance的配置
        condGen_config = json.load(open(args.condGenConfig, "r"))
        print(f"Load condGen_config from file {args.condGenConfig}")
        print("condGen_config: ", condGen_config)
        model.condGen_config = condGen_config
    
    if not "dataset_info" in model.__dict__:
        model.dataset_info = get_dataset_info(args_gen.dataset, args_gen.remove_h)

    if not "classifier" in model.__dict__:
        model.classifier = classifier

    model.condGenConfig = condGen_config

    # Create a dataloader with the generator
    if args_gen.property_pred == 0:
        mean, mad = property_norms[args.property]['mean'], property_norms[args.property]['mad']
    else:
        mean, mad = property_norms[args.property]['mean'], property_norms[args.property]['mad']
    print("mean: ", mean)
    print("mad: ", mad)
    if args.task == 'edm':
        diffusion_dataloader = DiffusionDataloader(args_gen, model, nodes_dist, prop_dist,
                                                   args.device, batch_size=args.batch_size, iterations=args.iterations, check_stability=args.check_stability)
        print("EDM: We evaluate the classifier on our generated samples")
        loss = test(args_gen, classifier, 0, diffusion_dataloader, mean, mad, args.property, args.device, 1, args.debug_break)
        print("Loss classifier on Generated samples: %.4f" % loss)
    elif args.task == 'qm9_second_half':
        print("qm9_second_half: We evaluate the classifier on QM9")
        loss = test(args_gen, classifier, 0, dataloaders['train'], mean, mad, args.property, args.device, args.log_interval,
                    args.debug_break)
        print("Loss classifier on qm9_second_half: %.4f" % loss)
    elif args.task == 'naive':
        print("Naive: We evaluate the classifier on QM9")
        length = dataloaders['train'].dataset.data[args.property].size(0)
        idxs = torch.randperm(length)
        dataloaders['train'].dataset.data[args.property] = dataloaders['train'].dataset.data[args.property][idxs]
        loss = test(args_gen, classifier, 0, dataloaders['train'], mean, mad, args.property, args.device, args.log_interval,
                    args.debug_break)
        print("Loss classifier on naive: %.4f" % loss)
    #elif args.task == 'numnodes':
    #    print("Numnodes: We evaluate the numnodes classifier on EDM samples")
    #    diffusion_dataloader = DiffusionDataloader(args_gen, model, nodes_dist, prop_dist, device,
    #                                               batch_size=args.batch_size, iterations=args.iterations)
    #    loss = test(classifier, 0, diffusion_dataloader, mean, mad, args.property, args.device, 1, args.debug_break)
    #    print("Loss numnodes classifier on EDM generated samples: %.4f" % loss)


def save_and_sample_conditional(args, device, model, prop_dist, dataset_info, epoch=0, id_from=0):
    one_hot, charges, x, node_mask = sample_sweep_conditional(args, device, model, dataset_info, prop_dist)

    vis.save_xyz_file(
        'outputs/%s/analysis/run%s/' % (args.exp_name, epoch), one_hot, charges, x, dataset_info,
        id_from, name='conditional', node_mask=node_mask)

    vis.visualize_chain("outputs/%s/analysis/run%s/" % (args.exp_name, epoch), dataset_info,
                        wandb=None, mode='conditional', spheres_3d=True)

    return one_hot, charges, x


def main_qualitative(args):
    args_gen = get_args_gen(args.generators_path)
    print("args_gen:", args_gen)
    dataloaders = get_dataloader(args_gen)
    
    if not len(args_gen.conditioning):
        args_gen.conditioning = [args.property]
    
    property_norms = compute_mean_mad(dataloaders, args_gen.conditioning, args_gen.dataset)
    model, nodes_dist, prop_dist, dataset_info = get_generator(args.generators_path,
                                                               dataloaders, args.device, args_gen,
                                                               property_norms)

    for i in range(args.n_sweeps):
        print("Sampling sweep %d/%d" % (i+1, args.n_sweeps))
        save_and_sample_conditional(args_gen, device, model, prop_dist, dataset_info, epoch=i, id_from=0)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_name', type=str, default='debug_alpha')
    parser.add_argument('--generators_path', type=str, default='outputs/exp_cond_alpha_pretrained')
    parser.add_argument('--classifiers_path', type=str, default='qm9/property_prediction/outputs/exp_class_alpha_pretrained')
    parser.add_argument('--property', type=str, default='alpha',
                        help="'alpha', 'homo', 'lumo', 'gap', 'mu', 'Cv'")
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='enables CUDA training')
    parser.add_argument('--debug_break', type=eval, default=False,
                        help='break point or not')
    parser.add_argument('--log_interval', type=int, default=5,
                        help='break point or not')
    parser.add_argument('--batch_size', type=int, default=1,
                        help='break point or not')
    parser.add_argument('--iterations', type=int, default=20,
                        help='break point or not')
    parser.add_argument('--task', type=str, default='qualitative',
                        help='naive, edm, qm9_second_half, qualitative')
    parser.add_argument('--n_sweeps', type=int, default=10,
                        help='number of sweeps for the qualitative conditional experiment')
    parser.add_argument("--finetune", type=int, default=0,)
    parser.add_argument("--expand_diff", type=int, default=0,)
    
    #for ours' conditional generation
    parser.add_argument("--condGenConfig", type=str, default=None,)
    
    #unused args
    parser.add_argument("--property_pred", type=int, default=0,)
    parser.add_argument("--prediction_threshold_t", type=int, default=10,)
    
    parser.add_argument("--test_epoch", type=int, default=0,)

    # check the stability of the generated samples
    parser.add_argument("--check_stability", type=int, default=0,)
    
    args = parser.parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if args.cuda else "cpu")
    args.device = device

    if args.task == 'qualitative':
        main_qualitative(args)
    else:
        main_quantitative(args)

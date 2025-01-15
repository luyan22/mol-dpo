import torch
from equivariant_diffusion.en_diffusion import EnVariationalDiffusion
from qm9.utils import compute_mean_mad
import copy
from eval_conditional_qm9 import analyze_stability_for_genmol
from qm9.models import DistributionProperty
import time
from qm9.property_prediction import prop_utils
import random
import utils
import pickle, os
from equivariant_diffusion import utils as flow_utils
from equivariant_diffusion.utils import assert_mean_zero_with_mask, remove_mean_with_mask,\
    assert_correctly_masked
class DPO(torch.nn.Module):
    def __init__(self, ref_args, args, exp_name, model_ref:EnVariationalDiffusion, model_reward:EnVariationalDiffusion, beta, lr, optim, num_epochs, nodes_dist, dataset_info, device, dataloaders, prop_dist:DistributionProperty, ref_model_type, reward_func, reward_network_type, lr_scheduler, save_model):
        super(DPO, self).__init__()

        # prepare finetune and ref models
        self.model_ref = copy.deepcopy(model_ref)
        self.model_ref.to(device)
        self.exp_name = exp_name
        self.model_finetune = model_ref  
        self.model_finetune.to(device)

        # prepare ema model
        self.model_ema = copy.deepcopy(model_ref)
        self.ema_decay = args.ema_decay
        self.ema:flow_utils.EMA = flow_utils.EMA(args.ema_decay)

        self.T = self.model_ref.T
        self.model_reward = model_reward
        self.beta = beta
        self.lr = lr
        self.ref_args = ref_args
        self.args = args
        self.optim=optim
        self.num_epochs=num_epochs
        self.nodes_dist = nodes_dist
        self.dataset_info = dataset_info
        self.device = device
        self.dataloaders = dataloaders
        self.prop_dist = prop_dist

        self.reward_func=reward_func
        self.reward_network_type=reward_network_type

        self.lr_scheduler = lr_scheduler

        self.save_model = save_model

        self.max_n_nodes = self.dataset_info["max_n_nodes"]

        # get target property from ref_args
        if ref_model_type == "uni_gem":
            property_norms = compute_mean_mad(dataloaders, [self.target_property], ref_args.dataset)
            self.target_property = ref_args.target_property
        elif ref_model_type == "edm":
            property_norms = compute_mean_mad(dataloaders, ref_args.conditioning, ref_args.dataset)
            self.target_property = ref_args.conditioning[0]
        else:
            raise ValueError(f"ref_model_type {ref_model_type} not supported")
        self.mean, self.mad = property_norms[self.target_property]['mean'], property_norms[self.target_property]['mad']
        print(f"Target property: {self.target_property}, mean: {self.mean}, mad: {self.mad}")
        self.prop_dist.set_normalizer(property_norms) 
        # freeze ref parameters, unfreeze finetune parameters
        for param in self.model_ref.parameters():
            param.requires_grad = False
        for param in self.model_finetune.parameters():
            param.requires_grad = True
        for param in self.model_reward.parameters():
            param.requires_grad = False

    def assert_one_hot(self, nodes, node_mask, max_nodes):
        n_samples = int(nodes.shape[0] / max_nodes)
        assert nodes.shape[0] % max_nodes == 0, f"nodes.shape[0] should be a multiple of max_nodes, but got {nodes.shape[0]} % {max_nodes}"
        assert len(nodes.shape) == 2 and nodes.shape[1] == 5, f"nodes shape {nodes.shape} should be (n_samples * max_nodes, 5)"
        # nodes.shape = (n_samples * max_nodes, 5)
        # node_mask.shape = (n_samples, max_nodes, 1)
        for sample_index in range(n_samples):
            for node_index in range(max_nodes):
                atom_index = sample_index * max_nodes + node_index
                if node_mask[sample_index][node_index] == 0:
                    continue
                assert torch.sum(torch.abs(nodes[atom_index])) == 1, f"nodes[{atom_index}] should be one-hot encoded"
                assert torch.max(nodes[atom_index]) == 1, f"nodes[{atom_index}] should be one-hot encoded"

    def gamma_pred(self, z, context, node_mask, edge_mask, max_nodes):
        nodes = z[:, :, 3:8]     
        # print("context: ", context)
        batch_size = nodes.shape[0]
        context = context.squeeze(dim=2).squeeze(dim=1)
        context = context.view(batch_size, -1)
        context = context[:, 0]
        # print("context: ", context)
        if self.reward_network_type == "uni_gem":
            assert False, "Currently use edm, not supported"
            gamma, prop_loss = self.model_reward.dpo_reward(z=z, context=context, node_mask=node_mask, edge_mask=edge_mask, dpo_beta=self.beta, mean=self.mean, mad=self.mad, reward_func=self.reward_func)
        elif self.reward_network_type == "egnn":
            n_samples = batch_size
            mad = self.mad
            mean = self.mean
            # print("n_nodes:", max_nodes)
            nodes = nodes.view(batch_size * max_nodes, -1)
            self.assert_one_hot(nodes, node_mask, max_nodes)

            atom_positions = z[:, :, 0:3].view(batch_size * max_nodes, -1)
            device = nodes.device
            node_mask = node_mask.view(batch_size * max_nodes, -1)
            edges = prop_utils.get_adj_matrix(max_nodes, batch_size, device)
            pred = self.model_reward(h0=nodes, x=atom_positions, edges=edges, edge_attr=None, node_mask=node_mask, edge_mask=edge_mask,
                     n_nodes=max_nodes)
            
            # pred (n_samples, )
            assert len(pred.shape) == 1 and pred.shape[0] == n_samples, f"pred shape {pred.shape} should be ({n_samples})"

            reward_func = self.reward_func
            dpo_beta = self.beta
            loss_fn = torch.nn.L1Loss(reduction='none')
            assert context.shape[0] == n_samples and len(context.shape) == 1, f"context shape {context.shape} should be ({n_samples})"
            assert context.shape == pred.shape, f"context shape {context.shape} should be equal to pred shape {pred.shape}"
            loss = loss_fn(pred, context)
            loss_reparam = loss_fn(pred * mad + mean, context * mad + mean)
            # print("pred: ", pred * mad + mean)
            # print("context: ", context * mad + mean)
    
     
            if reward_func == "minus":
                gamma = torch.exp((1 / dpo_beta) * (-loss))
                assert gamma.item() >= 0 and gamma.item() <= 1, f"gamma should be between 0 and 1, but got {gamma.value}"
            elif reward_func == "exp":
                gamma = torch.exp((1 / dpo_beta) * torch.exp(-loss))
            elif reward_func == "inverse":
                gamma = torch.exp(1 / (dpo_beta * loss))
            else:
                assert False, f"reward function {reward_func} not supported"
            assert gamma.shape == loss_reparam.shape, f"gamma shape {gamma.shape} should be equal to loss shape {loss.shape}"
            # gamma = torch.zeros_like(gamma)
            
            prop_loss = loss_reparam.mean()
            pass
        else:
            assert False, f"reward_network_type {self.reward_network_type} not supported"
        return gamma, prop_loss
    
    def train_step(self, n_samples, n_nodes, node_mask, edge_mask, context, fix_noise=False, conditional_sampling=False, wandb=None, sample_chain=None):
        # TODO
        # 1. sample a chain of z_t from the reference model
        if self.model_ref.property_pred: # x, h, pred are not used
            assert False, "Currently use edm, not supported"
            _, _, _, ref_zt_chain, ref_eps_t_chain = self.model_ref.sample_dpo_chain(max_n_nodes=self.max_n_nodes, n_samples=n_samples, node_mask=node_mask, edge_mask=edge_mask, context=context, fix_noise=fix_noise, conditional_sampling=conditional_sampling)
        else:
            loss_hist = None
            if sample_chain is None:
                print("Sampling new chain")
                # print("context: ", context.mean())
                x, h, ref_zt_chain, ref_eps_t_chain = self.model_ref.sample_dpo_chain(max_n_nodes=self.max_n_nodes, n_samples=n_samples, node_mask=node_mask, edge_mask=edge_mask, context=context, fix_noise=fix_noise, conditional_sampling=conditional_sampling)
            else:
                x, h, ref_zt_chain, ref_eps_t_chain = sample_chain["x"], sample_chain["h"], sample_chain["zt_chain"], sample_chain["eps_t_chain"]
                loss_hist = sample_chain["loss_hist"]
        ref_zt_chain = [z.detach() for z in ref_zt_chain]
        ref_eps_t_chain = [eps.detach() for eps in ref_eps_t_chain]

        assert_correctly_masked(x, node_mask)
        assert_mean_zero_with_mask(x, node_mask)

        one_hot = h['categorical']
        assert_correctly_masked(one_hot.float(), node_mask)

        # normalize x, h
        x_norm, h_norm, _ = self.model_ref.normalize(x, h, node_mask)

        # 2. calc stability and gamma for sampled results
        xh = torch.cat([x, h["categorical"]], dim=-1)
        xh_norm = torch.cat([x_norm, h_norm["categorical"]], dim=-1)
        if self.ref_args.include_charges:
            assert False, "Currently no charge"
            xh = torch.cat([xh, h["int"]], dim=-1) 
        assert xh.shape[0] == n_samples and xh.shape[1] == self.max_n_nodes and xh.shape[2] == 8, f"xh shape {xh.shape} should be ({n_samples}, {self.max_n_nodes}, 8)"
        assert xh_norm.shape == xh.shape, f"xh_norm shape {xh_norm.shape} should be equal to xh shape {xh.shape}"
        stability_lst = analyze_stability_for_genmol(one_hot=xh[:,:,3:8].detach(), x=xh[:,:,0:3].detach(), node_mask=node_mask, dataset_info=self.dataset_info) # eval the stability of samples
        print("stability_lst: ", stability_lst)
        print("sampled chain mol stability: ", stability_lst.count(True)/len(stability_lst))
        stability_mask = torch.tensor(stability_lst, dtype=torch.float32, device=self.device)

        print("finish chain sampling")

        gamma, prop_loss = self.gamma_pred(z=xh, context=context, node_mask=node_mask, edge_mask=edge_mask, max_nodes=self.max_n_nodes)

        gamma = gamma.detach()
        prop_loss = prop_loss.detach()
        print("gamma: ", gamma)
        print("prop_loss: ", prop_loss)
        if wandb is not None:
            wandb.log({"prop_loss": prop_loss})

        # 3. finetune the model with the sampled results and the reference chain
        loss_all, loss_hist = self.model_finetune.dpo_finetune_step(z=xh_norm, ref_zt_chain=ref_zt_chain, ref_eps_t_chain=ref_eps_t_chain, n_samples=n_samples, gamma=gamma, node_mask=node_mask, edge_mask=edge_mask, context=context, fix_noise=fix_noise, conditional_sampling=conditional_sampling, max_n_nodes=self.max_n_nodes, optim=self.optim, wandb=wandb, stability_mask=stability_mask, lr_dict=self.lr_dict(None, loss_hist=loss_hist), training_scheduler=self.training_scheduler)

        if wandb is not None:
            wandb.log({"loss_all": loss_all})

        # 4. return the sampled chain
        sample_chain = {"x": x, "h": h, "zt_chain": ref_zt_chain, "eps_t_chain": ref_eps_t_chain, "node_mask": node_mask, "edge_mask": edge_mask, "context": context, "gamma": gamma, "loss_hist": loss_hist}
        return sample_chain

    def prepare_masks(self, n_samples):
         # prepare data for sampling, mimic from qm9/sampling.py sample & eval_conditional_qm9 sample
        nodesxsample = self.nodes_dist.sample(n_samples) 
        max_n_nodes = self.dataset_info["max_n_nodes"]
        node_mask = torch.zeros(n_samples, max_n_nodes, dtype=torch.bool)
        for i in range(n_samples):
            node_mask[i, :nodesxsample[i]] = 1
        edge_mask = node_mask.unsqueeze(1) * node_mask.unsqueeze(2)
        diag_mask = ~torch.eye(edge_mask.size(1), dtype=torch.bool).unsqueeze(0)
        edge_mask *= diag_mask
        edge_mask = edge_mask.view(n_samples * max_n_nodes * max_n_nodes, 1).to(self.device)

        node_mask = node_mask.unsqueeze(2).to(self.device)

        node_mask = node_mask.type(torch.float32)
        edge_mask = edge_mask.type(torch.float32)

        return nodesxsample, node_mask, edge_mask
    
    def prepare_pseudo_context(self, n_samples, nodesxsample, node_mask):
        # prepare pseudo context for conditional sampling, mimic from eval_conditional_qm9 DiffusionDataloader.sample
        pseudo_context = self.prop_dist.sample_batch(nodesxsample).to(self.device)
        # pseudo_context = pseudo_context * self.prop_dist.normalizer[self.target_property]['mad'] + self.prop_dist.normalizer[self.target_property]['mean']
        assert self.mean == self.prop_dist.normalizer[self.target_property]['mean'], f"mean of target property is different from the mean of the dataset, self.mean: {self.mean} vs  prop_dist.normalizer[self.args.target_property]['mean']: {self.prop_dist.normalizer[self.target_property]['mean']}"
        assert self.mad == self.prop_dist.normalizer[self.target_property]['mad'], f"mad of target property is different from the mad of the dataset, self.mad: {self.mad} vs  prop_dist.normalizer[self.args.target_property]['mad']: {self.prop_dist.normalizer[self.target_property]['mad']}"

        # pseudo_context = pseudo_context * self.mad + self.mean

        # pseudo_context.shape = (n_samples, 1)
        assert self.max_n_nodes == 29, "max_n_nodes should be 29 for QM9"
        context = pseudo_context.view(n_samples, 1, 1).repeat(1, self.max_n_nodes, 1)
        assert context.shape == (n_samples, self.max_n_nodes, 1), f"context.shape: {context.shape}"

        assert context.shape == node_mask.shape, f"context.shape: {context.shape} should be equal to node_mask.shape: {node_mask.shape}"
        context = context * node_mask
        # context.shape = (n_samples, max_nodes, 1)
        return context

    def lr_dict(self, mu_prop_loss_chain=None, loss_hist=None):
        '''
        TODO
        construct a dictionary of learning rate for each time step
        choices=["importance_sampling", "linear_decay", "constant"]
        '''
        lr_dict = {}
        if self.lr_scheduler == "importance_sampling":
            if loss_hist is None:
                print("No loss history provided, use constant lr")
                for t_int in range(self.T):
                    lr_dict[t_int/self.T] = self.lr / self.T
            else:
                sum_loss = sum(loss_hist)
                for t_int in range(self.T):
                    t = float(t_int) / self.T
                    lr_dict[t] = self.lr * (loss_hist[t] / sum_loss)
        elif self.lr_scheduler == "linear_decay": 
            for t_int in range(self.T):
                t = float(t_int) / self.T
                lr_dict[t] = self.lr * t
        elif self.lr_scheduler == "constant":
            for t_int in range(self.T):
                lr_dict[t_int/self.T] = self.lr
        else:
            assert False, f"lr_scheduler {self.lr_scheduler} not supported"
        return lr_dict

    def train(self, n_samples, wandb=None, eval_interval=0, training_scheduler="increase_t"):
        self.training_scheduler = training_scheduler

        for epoch in range(self.num_epochs):
            epoch_start_time = time.time()
            print("Epoch: ", epoch)
            if wandb is not None:
                wandb.log({"Epoch": epoch})
            nodesxsample, node_mask, edge_mask = self.prepare_masks(n_samples)
            assert node_mask.shape == ( n_samples, self.max_n_nodes, 1), f"node_mask.shape: {node_mask.shape}"

            context = self.prepare_pseudo_context(n_samples, nodesxsample, node_mask) # different only in the dimensionality of the context

            self.train_step(n_samples=n_samples, n_nodes=nodesxsample, node_mask=node_mask, edge_mask=edge_mask, context=context, fix_noise=False, conditional_sampling=False, wandb=wandb) 

            if self.ema_decay > 0:
                self.ema.update_model_average(self.model_ema, self.model_finetune)

            epoch_end_time = time.time()
            print("Epoch training time: ", epoch_end_time - epoch_start_time)
            if wandb is not None:
                wandb.log({"Epoch_training_time": epoch_end_time - epoch_start_time})
            torch.cuda.empty_cache()
            if eval_interval > 0 and epoch % eval_interval == 0:
                eval_start_time = time.time()
                self.eval(n_samples=n_samples, stability_eval=True, fix_noise=False, conditional_sampling=False, wandb=wandb, ref_finetune_dist=True)
                eval_end_time = time.time()
                print("Epoch evaluation time: ", eval_end_time - eval_start_time)
                if wandb is not None:
                    wandb.log({"Epoch_evaluation_time": eval_end_time - eval_start_time})
            
            if self.save_model:
                dir = f"outputs/{self.exp_name}_{self.args.reward_network_type}_{self.beta}_{self.lr}"
                if not os.path.exists(dir):
                    os.makedirs(dir)
                utils.save_model(self.optim, f'{dir}/optim_{epoch}.npy')
                utils.save_model(self.model_finetune, '%s/generative_model_%d.npy' % (dir, epoch))
                if self.ema_decay > 0:
                    utils.save_model(self.model_ema, '%s/generative_model_ema_%d.npy' % (dir, epoch))
                with open('%s/args.pickle' % (dir), 'wb') as f:
                    pickle.dump(self.ref_args, f)
                pass
        pass

    def eval(self, n_samples, stability_eval=True, fix_noise=False, conditional_sampling=False, wandb=None, ref_finetune_dist=False): # TODO
        print(f"Evaluating model on {n_samples} samples")
        nodesxsample, node_mask, edge_mask = self.prepare_masks(n_samples)
        context = self.prepare_pseudo_context(n_samples, nodesxsample, node_mask) # different only in the dimensionality of the context
        # self.model_finetune.sample_chain(n_samples=n_samples, n_nodes=self.max_n_nodes, node_mask=node_mask, edge_mask=edge_mask, context=context)
        if self.model_finetune.property_pred: # x, h, pred are not used
            assert False, "DGAP conditional generation not supported"
            _, _, _, zt_chain, eps_t_chain = self.model_finetune.sample_dpo_chain(max_n_nodes=self.max_n_nodes, n_samples=n_samples, node_mask=node_mask, edge_mask=edge_mask, context=context, fix_noise=fix_noise, conditional_sampling=conditional_sampling)
        else:
            x, h, _, _ = self.model_finetune.sample_dpo_chain(max_n_nodes=self.max_n_nodes, n_samples=n_samples, node_mask=node_mask, edge_mask=edge_mask, context=context, fix_noise=fix_noise, conditional_sampling=conditional_sampling)
            if self.ema_decay > 0:
                x_ema, h_ema, _, _ = self.model_ema.sample_dpo_chain(max_n_nodes=self.max_n_nodes, n_samples=n_samples, node_mask=node_mask, edge_mask=edge_mask, context=context, fix_noise=fix_noise, conditional_sampling=conditional_sampling)
        

        xh = torch.cat([x, h["categorical"]], dim=-1)
        if self.ref_args.include_charges:
            xh = torch.cat([xh, h["int"]], dim=-1)
            assert False, "Currently no charge"
        assert xh.shape[0] == n_samples and xh.shape[1] == self.max_n_nodes and xh.shape[2] == 8, f"xh shape {xh.shape} should be ({n_samples}, {self.max_n_nodes}, 8)"

        gamma, prop_loss = self.gamma_pred(z=xh, context=context, node_mask=node_mask, edge_mask=edge_mask, max_nodes=self.max_n_nodes)
        # gamma, prop_loss = self.model_reward.dpo_reward(z0=xh, context=pseudo_context, node_mask=node_mask, edge_mask=edge_mask, dpo_beta=self.beta, mean=self.mean, mad=self.mad, reward_func=self.reward_func)
        gamma = gamma.detach()
        prop_loss = prop_loss.detach()

        # print("eval gamma: ", gamma)
        print("eval prop_loss: ", prop_loss)
        #TODO : add more eval metrics, such as mol-stable, novelty, etc.
        if stability_eval:
            z_final = xh
            mol_stable_lst = analyze_stability_for_genmol(one_hot=z_final[:,:,3:8].detach(), x=z_final[:,:,:3].detach(), node_mask=node_mask, dataset_info=self.dataset_info)
            print("mol_stable_lst: ", mol_stable_lst)
            print("mol stability: ", sum(mol_stable_lst) / len(mol_stable_lst))
            if wandb is not None:
                wandb.log({"eval_mol_stable": sum(mol_stable_lst) / len(mol_stable_lst)})
                wandb.log({"eval_prop_loss": prop_loss})
            
            if self.ema_decay > 0:
                xh_ema = torch.cat([x_ema, h_ema["categorical"]], dim=-1)
                if self.ref_args.include_charges:
                    assert False, "Currently no charge"
                    xh_ema = torch.cat([xh_ema, h_ema["int"]], dim=-1)
                assert xh_ema.shape[0] == n_samples and xh_ema.shape[1] == self.max_n_nodes and xh_ema.shape[2] == 8, f"xh_ema shape {xh_ema.shape} should be ({n_samples}, {self.max_n_nodes}, 8)"
                gamma_ema, prop_loss_ema = self.gamma_pred(z=xh_ema, context=context, node_mask=node_mask, edge_mask=edge_mask, max_nodes=self.max_n_nodes)
                gamma_ema = gamma_ema.detach()
                prop_loss_ema = prop_loss_ema.detach()
                # print("eval gamma_ema: ", gamma_ema)
                print("eval prop_loss_ema: ", prop_loss_ema)
                z_final_ema = xh_ema
                mol_stable_lst_ema = analyze_stability_for_genmol(one_hot=z_final_ema[:,:,3:8].detach(), x=z_final_ema[:,:,:3].detach(), node_mask=node_mask, dataset_info=self.dataset_info)
                print("mol_stable_lst_ema: ", mol_stable_lst_ema)
                print("mol stability ema: ", sum(mol_stable_lst_ema) / len(mol_stable_lst_ema))
                if wandb is not None:
                    wandb.log({"eval_mol_stable_ema": sum(mol_stable_lst_ema) / len(mol_stable_lst_ema)})
                    wandb.log({"eval_prop_loss_ema": prop_loss_ema})
        
        if ref_finetune_dist:
            dist, ema_dist = self.ref_finetune_dist()
            print("ref_finetune_dist: ", dist)
            if wandb is not None:
                wandb.log({"ref_finetune_dist": dist})
                if ema_dist is not None:
                    wandb.log({"ref_finetune_dist_ema": ema_dist})
        pass

    def ref_finetune_dist(self):
        # calculate the difference in parameters between the reference model and the finetuned model
        # dist = \sum {|ref_param - finetuned_param| / |ref_param|}
        ref_params = list(self.model_ref.parameters())
        finetuned_params = list(self.model_finetune.parameters())
        assert len(ref_params) == len(finetuned_params), f"ref_params: {len(ref_params)}, finetuned_params: {len(finetuned_params)}"
        diff_params = []
        for i in range(len(ref_params)):
            diff_params.append(torch.mean(torch.abs(ref_params[i] - finetuned_params[i]) / torch.abs(ref_params[i])))
        dist = sum(diff_params) / len(diff_params)

        if self.ema_decay > 0:
            ema_params = list(self.model_ema.parameters())
            diff_params_ema = []
            for i in range(len(ref_params)):
                diff_params_ema.append(torch.mean(torch.abs(ref_params[i] - ema_params[i]) / torch.abs(ref_params[i])))
            dist_ema = sum(diff_params_ema) / len(diff_params_ema)
        return dist, dist_ema if self.ema_decay > 0 else None
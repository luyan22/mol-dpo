import torch
from equivariant_diffusion.en_diffusion import EnVariationalDiffusion
from qm9.utils import compute_mean_mad
import copy
from eval_conditional_qm9 import analyze_stability_for_genmol
from qm9.models import DistributionProperty
import time
from qm9.property_prediction import prop_utils
import random

class DPO(torch.nn.Module):
    def __init__(self, ref_args, model_ref:EnVariationalDiffusion, model_reward:EnVariationalDiffusion, beta, lr, optim, num_epochs, nodes_dist, dataset_info, device, dataloaders, prop_dist:DistributionProperty, ref_model_type, reward_func, reward_network_type, lr_scheduler):
        super(DPO, self).__init__()
        self.model_ref = copy.deepcopy(model_ref)
        self.model_ref.to(device)
        self.model_finetune = model_ref  
        self.model_finetune.to(device)
        self.T = self.model_ref.T
        self.model_reward = model_reward
        self.beta = beta
        self.lr = lr
        self.args = ref_args
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

    def gamma_pred(self, z, context, node_mask, edge_mask, max_nodes):
        nodes = z[:, :, 3:8]     
        batch_size = nodes.shape[0]
        context = context.squeeze(dim=2).squeeze(dim=1)
        context = context.view(batch_size, -1)
        context = context.mean(dim=1)
        if self.reward_network_type == "uni_gem":
            gamma, prop_loss = self.model_reward.dpo_reward(z=z, context=context, node_mask=node_mask, edge_mask=edge_mask, dpo_beta=self.beta, mean=self.mean, mad=self.mad, reward_func=self.reward_func)
        elif self.reward_network_type == "egnn":
            n_samples = batch_size
            mad = self.mad
            mean = self.mean
            # print("n_nodes:", max_nodes)
            nodes = nodes.view(batch_size * max_nodes, -1)
            atom_positions = z[:, :, 0:3].view(batch_size * max_nodes, -1)
            device = nodes.device
            node_mask = node_mask.view(batch_size * max_nodes, -1)
            edges = prop_utils.get_adj_matrix(max_nodes, batch_size, device)
            pred = self.model_reward(h0=nodes, x=atom_positions, edges=edges, edge_attr=None, node_mask=node_mask, edge_mask=edge_mask,
                     n_nodes=max_nodes)
            reward_func = self.reward_func
            dpo_beta = self.beta
            loss_fn = torch.nn.L1Loss(reduction='none')
            assert context.shape[0] == n_samples and len(context.shape) == 1, f"context shape {context.shape} should be ({n_samples})"
            assert context.shape == pred.shape, f"context shape {context.shape} should be equal to pred shape {pred.shape}"
            loss = loss_fn(pred, context).sum() / pred.shape[0] 
            loss_reparam = loss_fn(pred * mad + mean, context * mad + mean).sum() / pred.shape[0]
            # assert loss_reparam.mean() < loss.mean(), f"loss_reparam should be smaller than loss, but got {loss_reparam.mean()} and {loss.mean()}"
            # print("dpo reward loss: ", loss)
            # print("dpo reward loss_reparam: ", loss_reparam)
     
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
            
            prop_loss = loss_reparam
            pass
        else:
            assert False, f"reward_network_type {self.reward_network_type} not supported"
        return gamma, prop_loss

    def mu_chain_reward_calc(self, mu_chain, n_samples, max_n_nodes, node_mask, edge_mask, context, wandb=None):
        mu_prop_loss = []
        for t, mu in enumerate(mu_chain):
            assert len(mu.shape) == 3 and mu.shape[0] == n_samples and mu.shape[1] == self.max_n_nodes and mu.shape[2] == 8, f"mu shape {mu.shape} should be ({n_samples}, {self.max_n_nodes}, 8)"
            gamma, prop_loss = self.gamma_pred(z=mu, context=context, node_mask=node_mask, edge_mask=edge_mask, max_nodes=self.max_n_nodes)
            gamma = gamma.detach()
            prop_loss = prop_loss.detach()
            # print("t: ", t, "prop_loss: ", prop_loss)
            mu_prop_loss.append(prop_loss)
            if wandb is not None:
                wandb.log({"prop_loss_t": prop_loss})
        return mu_prop_loss
    
    def train_step(self, n_samples, n_nodes, node_mask, edge_mask, context, fix_noise=False, conditional_sampling=False, wandb=None, sample_chain=None):
        # TODO
        if self.model_ref.property_pred: # x, h, pred are not used
            assert False, "Currently use edm, not supported"
            _, _, _, ref_zt_chain, ref_eps_t_chain = self.model_ref.sample_dpo_chain(max_n_nodes=self.max_n_nodes, n_samples=n_samples, node_mask=node_mask, edge_mask=edge_mask, context=context, fix_noise=fix_noise, conditional_sampling=conditional_sampling)
        else:
            if sample_chain is None:
                x, h, ref_zt_chain, ref_eps_t_chain, mu_chain = self.model_ref.sample_dpo_chain(max_n_nodes=self.max_n_nodes, n_samples=n_samples, node_mask=node_mask, edge_mask=edge_mask, context=context, fix_noise=fix_noise, conditional_sampling=conditional_sampling)
                mu_prop_loss_chain = self.mu_chain_reward_calc(mu_chain, n_samples, self.max_n_nodes, node_mask, edge_mask, context, wandb=wandb)
            else:
                x, h, ref_zt_chain, ref_eps_t_chain, mu_chain, mu_prop_loss_chain = sample_chain["x"], sample_chain["h"], sample_chain["zt_chain"], sample_chain["eps_t_chain"], sample_chain["mu_chain"], sample_chain["mu_prop_loss_chain"]
        ref_zt_chain = [z.detach() for z in ref_zt_chain]
        ref_eps_t_chain = [eps.detach() for eps in ref_eps_t_chain]
        mu_chain = [mu.detach() for mu in mu_chain]

        xh = torch.cat([x, h["categorical"]], dim=-1)
        if self.args.include_charges:
            xh = torch.cat([xh, h["int"]], dim=-1)
            assert False, "Currently no charge"
        assert xh.shape[0] == n_samples and xh.shape[1] == self.max_n_nodes and xh.shape[2] == 8, f"xh shape {xh.shape} should be ({n_samples}, {self.max_n_nodes}, 8)"
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

        loss_all = self.model_finetune.dpo_finetune_step(z=xh, ref_zt_chain=ref_zt_chain, ref_eps_t_chain=ref_eps_t_chain, mu_chain=mu_chain, n_samples=n_samples, gamma=gamma, node_mask=node_mask, edge_mask=edge_mask, context=context, fix_noise=fix_noise, conditional_sampling=conditional_sampling, max_n_nodes=self.max_n_nodes, optim=self.optim, wandb=wandb, stability_mask=stability_mask, lr_dict=self.lr_dict(mu_prop_loss_chain), training_scheduler=self.training_scheduler)

        if wandb is not None:
            wandb.log({"loss_all": loss_all})

        sample_chain = {"x": x, "h": h, "zt_chain": ref_zt_chain, "eps_t_chain": ref_eps_t_chain, "mu_chain":mu_chain, "mu_prop_loss_chain": mu_prop_loss_chain, "node_mask": node_mask, "edge_mask": edge_mask, "context": context, "gamma": gamma}
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
        choices=["importance_sampling", "linear_decay"]
        '''
        lr_dict = {}
        if self.lr_scheduler == "importance_sampling":
            if loss_hist is None:
                print("No loss history provided, use constant lr")
                for t_int in range(self.T):
                    lr_dict[t_int/self.T] = self.lr
            else:
                sum_loss = sum(loss_hist)
                for t_int in range(self.T):
                    t = float(t_int) / self.T
                    lr_dict[t] = self.lr * (loss_hist[t_int] / sum_loss)
        elif self.lr_scheduler == "linear_decay": 
            for t_int in range(self.T):
                t = float(t_int) / self.T
                lr_dict[t] = self.lr * t
        elif self.lr_scheduler == "constant":
            for t_int in range(self.T):
                lr_dict[t_int/self.T] = self.lr
        elif self.lr_scheduler == "mu_sampling":
            sum_prop_loss = sum(mu_prop_loss_chain)
            for t_int in range(self.T):
                lr_dict[t_int/self.T] = self.lr * (mu_prop_loss_chain[t_int] / sum_prop_loss)
            pass
        else:
            assert False, f"lr_scheduler {self.lr_scheduler} not supported"
        return lr_dict

    def train(self, n_samples, wandb=None, eval_interval=0, training_scheduler="increase_t"):
        self.training_scheduler = training_scheduler
        dpo_chain_lst = []
        for epoch in range(self.num_epochs):
            epoch_start_time = time.time()
            print("Epoch: ", epoch)
            if wandb is not None:
                wandb.log({"Epoch": epoch})
            nodesxsample, node_mask, edge_mask = self.prepare_masks(n_samples)
            assert node_mask.shape == ( n_samples, self.max_n_nodes, 1), f"node_mask.shape: {node_mask.shape}"

            context = self.prepare_pseudo_context(n_samples, nodesxsample, node_mask) # different only in the dimensionality of the context

            if len(dpo_chain_lst):
                # TODO use previous sample_chain as input for next sample_chain
                new_sample = random.random()
                if new_sample < 0.5:
                    node_mask = dpo_chain_lst[-1]["node_mask"]
                    edge_mask = dpo_chain_lst[-1]["edge_mask"]
                    context = dpo_chain_lst[-1]["context"]
                    sample_chain = self.train_step(n_samples=n_samples, n_nodes=nodesxsample, node_mask=node_mask, edge_mask=edge_mask, context=context, fix_noise=False, conditional_sampling=False, wandb=wandb, sample_chain=dpo_chain_lst[-1]) # TODO more data needed for conditional sampling
                else:
                    sample_chain = self.train_step(n_samples=n_samples, n_nodes=nodesxsample, node_mask=node_mask, edge_mask=edge_mask, context=context, fix_noise=False, conditional_sampling=False, wandb=wandb) # TODO more data needed for conditional sampling
                dpo_chain_lst.append(sample_chain)
            else:
                sample_chain = self.train_step(n_samples=n_samples, n_nodes=nodesxsample, node_mask=node_mask, edge_mask=edge_mask, context=context, fix_noise=False, conditional_sampling=False, wandb=wandb) # TODO more data needed for conditional sampling
                dpo_chain_lst.append(sample_chain)


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
        pass

    def eval(self, n_samples, stability_eval=True, fix_noise=False, conditional_sampling=False, wandb=None, ref_finetune_dist=False): # TODO
        print(f"Evaluating model on {n_samples} samples")
        nodesxsample, node_mask, edge_mask = self.prepare_masks(n_samples)
        context = self.prepare_pseudo_context(n_samples, nodesxsample, node_mask) # different only in the dimensionality of the context
        self.model_finetune.sample_chain(n_samples=n_samples, n_nodes=self.max_n_nodes, node_mask=node_mask, edge_mask=edge_mask, context=context)
        if self.model_finetune.property_pred: # x, h, pred are not used
            assert False, "DGAP conditional generation not supported"
            _, _, _, zt_chain, eps_t_chain = self.model_finetune.sample_dpo_chain(max_n_nodes=self.max_n_nodes, n_samples=n_samples, node_mask=node_mask, edge_mask=edge_mask, context=context, fix_noise=fix_noise, conditional_sampling=conditional_sampling)
        else:
            x, h, zt_chain, eps_t_chain, mu_chain = self.model_finetune.sample_dpo_chain(max_n_nodes=self.max_n_nodes, n_samples=n_samples, node_mask=node_mask, edge_mask=edge_mask, context=context, fix_noise=fix_noise, conditional_sampling=conditional_sampling)
            # _, _, zt_chain, eps_t_chain = self.model_ref.sample_dpo_chain(max_n_nodes=self.max_n_nodes, n_samples=n_samples, node_mask=node_mask, edge_mask=edge_mask, context=context, fix_noise=fix_noise, conditional_sampling=conditional_sampling)
        
        zt_chain = [z.detach() for z in zt_chain]
        eps_t_chain = [eps.detach() for eps in eps_t_chain]

        xh = torch.cat([x, h["categorical"]], dim=-1)
        if self.args.include_charges:
            xh = torch.cat([xh, h["int"]], dim=-1)
            assert False, "Currently no charge"
        assert xh.shape[0] == n_samples and xh.shape[1] == self.max_n_nodes and xh.shape[2] == 8, f"xh shape {xh.shape} should be ({n_samples}, {self.max_n_nodes}, 8)"

        gamma, prop_loss = self.gamma_pred(z=xh, context=context, node_mask=node_mask, edge_mask=edge_mask, max_nodes=self.max_n_nodes)
        # gamma, prop_loss = self.model_reward.dpo_reward(z0=xh, context=pseudo_context, node_mask=node_mask, edge_mask=edge_mask, dpo_beta=self.beta, mean=self.mean, mad=self.mad, reward_func=self.reward_func)
        gamma = gamma.detach()
        prop_loss = prop_loss.detach()

        print("eval gamma: ", gamma)
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
        
        if ref_finetune_dist:
            dist = self.ref_finetune_dist()
            print("ref_finetune_dist: ", dist)
            if wandb is not None:
                wandb.log({"ref_finetune_dist": dist})
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
        return dist
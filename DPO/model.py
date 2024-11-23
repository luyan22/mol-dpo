import torch
from equivariant_diffusion.en_diffusion import EnVariationalDiffusion
from qm9.utils import compute_mean_mad

class DPO(torch.nn.Module):
    def __init__(self, ref_args, model_ref:EnVariationalDiffusion, beta, optim, num_epochs, nodes_dist, dataset_info, device, dataloaders, prop_dist):
        super(DPO, self).__init__()
        self.model_ref = model_ref  
        self.model_finetune = model_ref
        self.beta = beta
        self.args = ref_args
        self.optim=optim
        self.num_epochs=num_epochs
        self.nodes_dist = nodes_dist
        self.dataset_info = dataset_info
        self.device = device
        self.dataloaders = dataloaders
        self.prop_dist = prop_dist

        self.max_n_nodes = self.dataset_info["max_n_nodes"]

        # get target property from ref_args
        self.target_property = ref_args.target_property
        print("target_property: ", self.target_property)
        property_norms = compute_mean_mad(dataloaders, [self.target_property], ref_args.dataset)
        self.mean, self.mad = property_norms[self.target_property]['mean'], property_norms[self.target_property]['mad']
        # self.property_norm
        self.prop_dist.set_normalizer(property_norms) 
        # freeze ref parameters, unfreeze finetune parameters
        for param in self.model_ref.parameters():
            param.requires_grad = False
        for param in self.model_finetune.parameters():
            param.requires_grad = True
    
    def train_step(self, n_samples, n_nodes, node_mask, edge_mask, context, fix_noise=False, conditional_sampling=False, pseudo_context=None):
        # TODO
        ref_sample_dpo_chain = self.model_ref.sample_dpo_chain(max_n_nodes=self.max_n_nodes, n_samples=n_samples, node_mask=node_mask, edge_mask=edge_mask, context=context, fix_noise=fix_noise, conditional_sampling=conditional_sampling)
        ref_zt_chain = ref_sample_dpo_chain["zt"]
        ref_eps_t_chain = ref_sample_dpo_chain["eps_t"]
        print("DPO t: ", self.args.T)

        sample_dpo_chain = self.model_finetune.sample_dpo_chain(max_n_nodes=self.max_n_nodes, n_samples=n_samples, node_mask=node_mask, edge_mask=edge_mask, context=context, fix_noise=fix_noise, conditional_sampling=conditional_sampling)
        zt_chain = sample_dpo_chain["zt"]
        eps_t_chain = sample_dpo_chain["eps_t"]

        z0 = ref_zt_chain[0]

        loss_l2 = torch.nn.MSELoss(reduction='None')

        for t in reversed(range(self.args.T)):
            zt = ref_zt_chain[t]

            t_array = torch.full((n_samples, 1), fill_value=t, device=self.device)
            t_array = t_array / self.args.T
            gamma_t = self.model_ref.inflate_batch_array(self.model_ref.gamma(t_array), zt)
            alpha_t = self.model_ref.alpha(gamma_t, zt)
            sigma_t = self.model_ref.sigma(t_array, zt)
            epsilon_t = (zt - alpha_t * z0) / sigma_t


            phi_star = eps_t_chain[t]
            gamma = self.model_ref.dpo_reward(z0=zt_chain[0], context=pseudo_context, node_mask=node_mask, edge_mask=edge_mask, dpo_beta=self.beta)
            # alpha_t = self.model_ref.alpha(gamma, t)

            RHS = phi_star
            LHS = gamma * epsilon_t + (1 - gamma) * ref_eps_t_chain[t]
            print("LHS: ", LHS.shape)
            print("RHS: ", RHS.shape)
            loss_t = loss_l2(LHS, RHS)
            print("loss_t: ", loss_t.shape)
            loss_t = loss_t.mean()
            print("loss_t: ", loss_t)
            self.optim.zero_grad()
            loss_t.backward()
            self.optim.step()


            pass
        pass

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
    
    def prepare_pseudo_context(self, n_samples, nodesxsample):
        # prepare pseudo context for conditional sampling, mimic from eval_conditional_qm9 DiffusionDataloader.sample
        pseudo_context = self.prop_dist.sample_batch(nodesxsample).to(self.device)
        pseudo_context = pseudo_context * self.prop_dist.normalizer[self.args.target_property]['mad'] + self.prop_dist.normalizer[self.args.target_property]['mean']
        assert self.mean == self.prop_dist.normalizer[self.args.target_property]['mean'], f"mean of target property is different from the mean of the dataset, self.mean: {self.mean} vs  prop_dist.normalizer[self.args.target_property]['mean']: {self.prop_dist.normalizer[self.args.target_property]['mean']}"
        return pseudo_context

    def train(self, n_samples):
        for epoch in range(self.num_epochs):
            print("Epoch: ", epoch)
            nodesxsample, node_mask, edge_mask = self.prepare_masks(n_samples)

            pseudo_context = self.prepare_pseudo_context(n_samples, nodesxsample)

            self.train_step(n_samples=n_samples, n_nodes=nodesxsample, node_mask=node_mask, edge_mask=edge_mask, context=None, fix_noise=False, conditional_sampling=False, pseudo_context=pseudo_context) # TODO more data needed for conditional sampling
        pass
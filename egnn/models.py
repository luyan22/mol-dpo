import torch
import torch.nn as nn
from egnn.egnn_new import EGNN, GNN
from equivariant_diffusion.utils import remove_mean, remove_mean_with_mask
import numpy as np
from egnn.torchmd_et import TorchMD_ET, AccumulatedNormalization
from egnn.output_modules import EquivariantVectorOutput, EquivariantScalarMol2
from torch_scatter import scatter_add, scatter_mean

from egnn.utils import GaussianSmearing
from egnn.egnn_new import EquivariantBlock, coord2diff
from egnn.priors import Atomref
import torch.nn.functional as F

class EGNN_dynamics_QM9(nn.Module):
    def __init__(self, in_node_nf, context_node_nf,
                 n_dims, hidden_nf=64, device='cpu',
                 act_fn=torch.nn.SiLU(), n_layers=4, attention=False,
                 condition_time=True, tanh=False, mode='egnn_dynamics', norm_constant=0,
                 inv_sublayers=2, sin_embedding=False, normalization_factor=100, aggregation_method='sum', condition_decoupling=False,
                 uni_diffusion=False, use_basis=False, decoupling=False, 
                 freeze_gradient=False, # if freeze_gradient, the gradient of property prediciton don't influence the generation model
                 atom_type_pred = False, # if atom_type_pred, the model will predict the atom type, even under the DGAP mode
                 branch_layers_num = 0, # the number of layers for the second branch
                 atom_type4prop_pred = 0, # use atom type for property prediction
                 use_ref = False, # use atomtyperef
                 train_prop_pred_4condition_only = False, # if train_prop_pred_4condition_only, only train the property prediction head for the condition
                 **kwargs):
        super().__init__()
        self.mode = mode
        
        self.uni_diffusion = uni_diffusion
        self.use_basis = use_basis
        
        self.decoupling = decoupling
        
        self.hidden_nf = hidden_nf
        
        self.property_pred = kwargs.get('property_pred', False)
        self.prediction_threshold_t=kwargs.get('prediction_threshold_t', 10)
        
        self.pretraining = kwargs.get('pretraining', False)
        
        # Get the finetune option
        self.finetune = kwargs.get('finetune', False)
        
        self.atom_type4prop_pred = atom_type4prop_pred
        self.use_ref = use_ref
        self.train_prop_pred_4condition_only = train_prop_pred_4condition_only
        
        if mode == 'egnn_dynamics':
            
            if self.uni_diffusion: # convert the property into the latent space
                # context_node_nf = 64 + 1 # laten space + time
                context_node_nf = 2 # context + t2
                if self.use_basis:
                    context_basis_dim = 64
                else:
                    context_basis_dim = 0
            else:
                context_basis_dim = 0
                
            self.egnn = EGNN(
                in_node_nf=in_node_nf + context_node_nf, in_edge_nf=1,
                hidden_nf=hidden_nf, device=device, act_fn=act_fn,
                n_layers=n_layers, attention=attention, tanh=tanh, norm_constant=norm_constant,
                inv_sublayers=inv_sublayers, sin_embedding=sin_embedding,
                normalization_factor=normalization_factor,
                aggregation_method=aggregation_method, condition_decoupling=condition_decoupling, context_basis_dim=context_basis_dim)

            if self.decoupling:
                self.egnn2 = EGNN(
                    in_node_nf=in_node_nf + context_node_nf, in_edge_nf=1,
                    hidden_nf=hidden_nf, device=device, act_fn=act_fn,
                    n_layers=n_layers, attention=attention, tanh=tanh, norm_constant=norm_constant,
                    inv_sublayers=inv_sublayers, sin_embedding=sin_embedding,
                    normalization_factor=normalization_factor,
                    aggregation_method=aggregation_method, condition_decoupling=condition_decoupling, context_basis_dim=context_basis_dim)
            
            
            self.in_node_nf = in_node_nf
        elif mode == 'gnn_dynamics':
            self.gnn = GNN(
                in_node_nf=in_node_nf + context_node_nf + 3, in_edge_nf=0,
                hidden_nf=hidden_nf, out_node_nf=3 + in_node_nf, device=device,
                act_fn=act_fn, n_layers=n_layers, attention=attention,
                normalization_factor=normalization_factor, aggregation_method=aggregation_method)
        elif mode == 'torchmdnet':
            print("uni_diffusion: ", self.uni_diffusion)
            print("self.finetune: ", self.finetune)
            print("in_node_nf: ", in_node_nf)
            print("context_node_nf: ", context_node_nf)
            if self.uni_diffusion:
                context_node_nf = 53 + 1 # 53 is the property numbers, 1 is the time
                self.mask_embedding = nn.Embedding(1, context_node_nf) # for masking the property
            
            # if self.finetune: #TODO: check if this is correct
            context_node_nf = 53 + 1
            self.property_fc = nn.Linear(1, context_node_nf - 1) # fc emb + property
            
            shared_args = {'hidden_channels': 256, 'num_layers': 8, 'num_rbf': 64, 'rbf_type': 'expnorm', 'trainable_rbf': False, 'activation': 'silu', 'neighbor_embedding': True, 'cutoff_lower': 0.0, 'cutoff_upper': 5.0, 'max_z': in_node_nf + context_node_nf, 'max_num_neighbors': 32}
            self.gnn = TorchMD_ET(
                attn_activation="silu",
                num_heads=8,
                distance_influence="both",
                layernorm_on_vec='whitened',
                md17=False,
                seperate_noise=False,
                **shared_args,
            )
            self.node_out = nn.Sequential(
                nn.Linear(shared_args['hidden_channels'], shared_args['hidden_channels']),
                nn.SiLU(),
                nn.Linear(shared_args['hidden_channels'], in_node_nf + context_node_nf),
            )
            self.noise_out = EquivariantVectorOutput(shared_args['hidden_channels'], shared_args['activation'])

            if self.pretraining:
                # for denoise
                self.noise_out2 = EquivariantVectorOutput(shared_args['hidden_channels'], shared_args['activation'])
                # for nomalizer
                # normalized_pos_target = self.model.pos_normalizer(batch.pos_target)
                self.pos_normalizer = AccumulatedNormalization(accumulator_shape=(3,))
                
                
            # self.noise_head = 
            # self.node_head = 
        elif mode == "DGAP":
            if self.uni_diffusion: # convert the property into the latent space
                # context_node_nf = 64 + 1 # laten space + time
                context_node_nf = 2 # context + t2
                if self.use_basis:
                    context_basis_dim = 64
                else:
                    context_basis_dim = 0
            else:
                context_basis_dim = 0
            self.in_node_nf = in_node_nf # here the in_node_nf contains time, didn't contain the conditional dim
            self.atom_type_pred = atom_type_pred
            # if self.atom_type_pred:
            #     assert branch_layers_num > 0, "branch_layers_num should be larger than 0 when the atom_type_pred is True"
            self.egnn = EGNN(
                in_node_nf=in_node_nf + context_node_nf, in_edge_nf=1,
                hidden_nf=hidden_nf, device=device, act_fn=act_fn,
                n_layers=n_layers, attention=attention, tanh=tanh, norm_constant=norm_constant,
                inv_sublayers=inv_sublayers, sin_embedding=sin_embedding,
                normalization_factor=normalization_factor,
                aggregation_method=aggregation_method, condition_decoupling=condition_decoupling, context_basis_dim=context_basis_dim, branch_layers_num=branch_layers_num, condition=context_node_nf, later_fusion=self.atom_type4prop_pred)
            
            # self.condition_prop_pred_head = nn.Sequential( #TODO MLP for property prediction in conditional generation guidance. t ~ (10, 1000)
            #     nn.Linear(context_node_nf, hidden_nf),
            #     act_fn,
            #     nn.Linear(hidden_nf, hidden_nf),
            #     act_fn,
            #     nn.Linear(hidden_nf, hidden_nf),
            # )
            
            self.freeze_gradient = freeze_gradient
            if self.freeze_gradient:
                self.property_layers = nn.ModuleList()
                p_layers = 2
                for _ in range(p_layers):
                    self.property_layers.append(EquivariantBlock(hidden_nf, edge_feat_nf=2, device=device,
                                                               act_fn=act_fn, n_layers=inv_sublayers,
                                                               attention=attention, norm_diff=True, tanh=tanh,
                                                               coords_range=15, norm_constant=norm_constant,
                                                               sin_embedding=None,
                                                               normalization_factor=1,
                                                               aggregation_method='sum'))
            
            if self.atom_type4prop_pred:
                self.node_dec = nn.Sequential(
                    nn.Linear(hidden_nf * 2, hidden_nf),
                    act_fn,
                    nn.Linear(hidden_nf,  hidden_nf),
                )
                if self.use_ref:
                    self.atomref = Atomref(max_z=100)
            else:
                self.node_dec = nn.Sequential(
                    nn.Linear(hidden_nf, hidden_nf),
                    act_fn,
                    nn.Linear(hidden_nf,  hidden_nf),
                )
            
            if self.atom_type4prop_pred:
                self.atom_type4prop_pred_decoder = nn.Sequential(
                    nn.Linear(in_node_nf - 2, hidden_nf), # exclude time and conditional dim
                    act_fn,
                    nn.Linear(hidden_nf, hidden_nf), # in_node_nf = 6 contains time; (or 7 include charge) context_node_nf: conditional generation
                ) # for atom prediction 
            
            if self.atom_type_pred:
                self.node_dec2 = nn.Sequential(
                    nn.Linear(hidden_nf, hidden_nf),
                    act_fn,
                    nn.Linear(hidden_nf,  in_node_nf + context_node_nf), # in_node_nf = 6 contains time; context_node_nf: conditional generation
                ) # for atom prediction 
            
            basic_prob = kwargs.get('basic_prob', 0)
            if basic_prob:
                pred_dim = 53
            else:
                pred_dim = 1
            
            self.graph_dec = nn.Sequential(
                nn.Linear(hidden_nf, hidden_nf),
                act_fn,
                nn.Linear(hidden_nf,  pred_dim),
            )
            if train_prop_pred_4condition_only:
                print("init train_prop_pred_4condition_only")
                self.blur_node_decode = nn.Sequential( #TODO MLP for property prediction in conditional generation guidance. t ~ (10, 1000)
                    nn.Linear(hidden_nf, hidden_nf),
                    act_fn,
                    nn.Linear(hidden_nf, hidden_nf),
                )
                self.blur_graph_decode = nn.Sequential(
                    nn.Linear(hidden_nf, hidden_nf),
                    act_fn,
                    nn.Linear(hidden_nf, pred_dim),
                )
            
        elif mode == "PAT":
            '''
            PAT = DGAP - graph_dec
            input_dim = 3
            change the dim of the node_dec to h_dims
            '''
            if self.uni_diffusion: # convert the property into the latent space
                # context_node_nf = 64 + 1 # laten space + time
                context_node_nf = 2 # context + t2
                if self.use_basis:
                    context_basis_dim = 64
                else:
                    context_basis_dim = 0
            else:
                context_basis_dim = 0
            self.in_node_nf = in_node_nf
            self.egnn = EGNN(
                in_node_nf=in_node_nf + context_node_nf, in_edge_nf=1,
                hidden_nf=hidden_nf, device=device, act_fn=act_fn,
                n_layers=n_layers, attention=attention, tanh=tanh, norm_constant=norm_constant,
                inv_sublayers=inv_sublayers, sin_embedding=sin_embedding,
                normalization_factor=normalization_factor,
                aggregation_method=aggregation_method, condition_decoupling=condition_decoupling, context_basis_dim=context_basis_dim)
            #7 0 3
            #in_node_nf: 7 = 5 + 1 + 1(one-hot + charges + time)
            self.node_dec = nn.Sequential(
                nn.Linear(hidden_nf, hidden_nf),
                act_fn,
                nn.Linear(hidden_nf, in_node_nf),
            )
        self.context_node_nf = context_node_nf
        self.device = device
        self.n_dims = n_dims
        self._edges_dict = {}
        self.condition_time = condition_time
        
        
        
        
        if self.uni_diffusion:
            # init the guassian layer for the property
            # get start, end, num_gaussians, trainable from kwargs
            start = kwargs.get('start', -5)
            end = kwargs.get('end', 5)
            num_gaussians = kwargs.get('num_gaussians', 100)
            trainable = kwargs.get('trainable', True)
            property_latent = kwargs.get('property_latent', 64)
            
            self.gaussian_layer = GaussianSmearing(start, end, num_gaussians, trainable)
            
            self.property_emb = nn.Sequential(
                nn.Linear(num_gaussians, property_latent),
                nn.SiLU(),
                nn.Linear(property_latent, property_latent),
            )
            
            # define the property output layer
            
            if mode == 'torchmdnet':
               self.property_out = nn.Sequential(
                        nn.Linear(hidden_nf, hidden_nf),
                        nn.SiLU(),
                        nn.Linear(hidden_nf, context_node_nf - 1), # substract time
                    )
            else:
                self.property_out = nn.Sequential(
                    nn.Linear(hidden_nf, hidden_nf),
                    nn.SiLU(),
                    nn.Linear(hidden_nf, 1),
                )
        
        if self.finetune and mode == 'torchmdnet':
            self.property_out = nn.Sequential(
                    nn.Linear(hidden_nf, hidden_nf),
                    nn.SiLU(),
                    nn.Linear(hidden_nf, 1),
                )
            
            

    def forward(self, t, xh, node_mask, edge_mask, context=None):
        raise NotImplementedError

    def wrap_forward(self, node_mask, edge_mask, context):
        def fwd(time, state):
            return self._forward(time, state, node_mask, edge_mask, context)
        return fwd

    def unwrap_forward(self):
        return self._forward
    
    @torch.no_grad()
    def DGAP_prop_pred(self, t, xh, node_mask, edge_mask, atom_type_pred=False):
        '''
        use node decoder and graph decoder to predict the property
        '''
        assert self.mode == "DGAP", "DGAP mode should be used"
        assert (t == 0).all(), "t should be 0"
        bs, n_nodes, dims = xh.shape
        h_dims = dims - self.n_dims
        edges = self.get_adj_matrix(n_nodes, bs, self.device)
        edges = [x.to(self.device) for x in edges]
        node_mask = node_mask.view(bs*n_nodes, 1)
        edge_mask = edge_mask.view(bs*n_nodes*n_nodes, 1)
        xh = xh.view(bs*n_nodes, -1).clone() * node_mask
        x = xh[:, 0:self.n_dims].clone()
        assert h_dims == 6, "h_dims should be 6"
        h = xh[:, self.n_dims:].clone()
        if self.condition_time:
            if np.prod(t.size()) == 1:
                # t is the same for all elements in batch.
                h_time = torch.empty_like(h[:, 0:1]).fill_(t.item())
            else:
                # t is different over the batch dimension.
                h_time = t.view(bs, 1).repeat(1, n_nodes)
                h_time = h_time.view(bs * n_nodes, 1)
            
            if atom_type_pred:
                h = torch.ones_like(h) * node_mask # if atom prediction loss, h should be all in 1
            
            h = torch.cat([h, h_time], dim=1)

        # print("h_shape: ", h.shape)
        
        
        if self.use_basis:
            h_final, x_final, org_h = self.egnn(h, x, edges, node_mask=node_mask, edge_mask=edge_mask, context_basis=context_basis)
        else:
            # print("h_shape: ", h.shape)
            h_final, x_final, org_h = self.egnn(h, x, edges, node_mask=node_mask, edge_mask=edge_mask, batch_size=bs, n_nodes=n_nodes)
                
        if self.decoupling:
            _, _, org_h = self.egnn2(h, x, edges, node_mask=node_mask, edge_mask=edge_mask)
                
                
        vel = (x_final - x) * node_mask  # This masking operation is redundant but just in case
        #TODO check the output dim
        
        if self.atom_type4prop_pred:
            # no_noise_xh = no_noise_xh.view(bs*n_nodes, -1)
            # print("no_noise_xh: ", no_noise_xh.shape, no_noise_xh[0])
            no_noise_h = xh[:, self.n_dims:-1].clone()
            assert no_noise_h.shape[1] == self.in_node_nf - 2, "no_noise_h should have the same size as h. no_noise_h.shape: %s, self.in_node_nf - 2: %s" % (str(no_noise_h.shape), str(self.in_node_nf - 2))
            # atom_type_decode = self.atom_type4prop_pred_decoder(no_noise_h)
            atom_type_decode = self.egnn.embedding2(no_noise_h)
            distances, _ = coord2diff(x, edges)
            no_noise_h_emb, _ = self.egnn.atomtype_add_layer(atom_type_decode, x, edges, node_mask=node_mask, edge_mask=edge_mask, edge_attr=distances)
            
            node_dec = self.node_dec(torch.cat([org_h, no_noise_h_emb], dim=1))
            
            # print("before node_dec: ", node_dec.shape)
            node_dec = node_dec.view(bs, n_nodes, self.hidden_nf)
            # print("node_dec: ", node_dec.shape)
            node_dec = torch.sum(node_dec, dim=1)
            # print("after node_dec: ", node_dec.shape)
            pred = self.graph_dec(node_dec)
            if self.use_ref:
                # return 1 index in the org_h0
                atom_types = torch.argmax(no_noise_h, dim=1)
                atoms_pred = self.atomref(atom_types)
                atoms_pred = atoms_pred * node_mask
                atoms_pred = atoms_pred.view(-1, n_nodes)
                atoms_pred = torch.sum(atoms_pred, dim=1)
                pred = pred + atoms_pred.unsqueeze(1)
            # print("before pred: ", pred.shape)
            pred = pred.squeeze(1) 
        else:
            node_dec = self.node_dec(org_h)
            node_dec = node_dec.view(bs, n_nodes, self.hidden_nf)
            node_dec = torch.sum(node_dec, dim=1)
            pred = self.graph_dec(node_dec)
            pred = pred.squeeze(1)
        
        return pred
    
    def PAT_forward(self, t, xh, node_mask, edge_mask, context, t2=None, mask_y=None):
        '''
        input x: (bs, n_nodes, 3), no h
        '''
        bs, n_nodes, dims = xh.shape
        # print("xh_shape: ", xh.shape)
        h_dims = dims - self.n_dims
        edges = self.get_adj_matrix(n_nodes, bs, self.device)
        edges = [x.to(self.device) for x in edges]
        node_mask = node_mask.view(bs*n_nodes, 1)
        edge_mask = edge_mask.view(bs*n_nodes*n_nodes, 1)
        xh = xh.view(bs*n_nodes, -1).clone() * node_mask
        x = xh[:, 0:self.n_dims].clone()
        
        if h_dims == 0:
            h = torch.ones(bs*n_nodes, 6).to(self.device)
            h = h * node_mask
            xh = torch.cat([x, h], dim=1)
        else:
            h = xh[:, self.n_dims:].clone()
        #assert h is all in 1
        assert (h == (torch.ones_like(h).detach() * node_mask)).all(), "h should be all in 1"

        if self.condition_time:
            if np.prod(t.size()) == 1:
                # t is the same for all elements in batch.
                h_time = torch.empty_like(h[:, 0:1]).fill_(t.item())
            else:
                # t is different over the batch dimension.
                h_time = t.view(bs, 1).repeat(1, n_nodes)
                h_time = h_time.view(bs * n_nodes, 1)
            h = torch.cat([h, h_time], dim=1)
        # print("h_shape after time in PAT: ", h.shape)
        if context is not None:
            # We're conditioning, awesome!
            #一行完成assert not true
            assert not (self.mode == "DGAP" or self.mode == "PAT"), "unconditional generation not supported have context"
            
        assert self.mode == "PAT", "PAT mode should be used"
        h_final, x_final, org_h = self.egnn(h, x, edges, node_mask=node_mask, edge_mask=edge_mask)
                
                
                
        vel = (x_final - x) * node_mask  # This masking operation is redundant but just in case
        #TODO check the output dim
        node_dec = self.node_dec(org_h)
        #将node_dec作为h的预测
        # print("node_dec_dim: ", node_dec.size())
        # print("bs, n_nodes: ", bs, n_nodes)
        h_final = node_dec.view(bs*n_nodes, -1)
        
        if context is not None and self.context_node_nf > 0:
            # Slice off context size:
            h_final = h_final[:, :-self.context_node_nf]

        if self.condition_time:
            # Slice off last dimension which represented time.
            h_final = h_final[:, :-1]

        vel = vel.view(bs, n_nodes, -1)

        if torch.any(torch.isnan(vel)):
            print('Warning: detected nan, resetting EGNN output to zero.')
            vel = torch.zeros_like(vel)

        if node_mask is None:
            vel = remove_mean(vel)
        else:
            vel = remove_mean_with_mask(vel, node_mask.view(bs, n_nodes, 1))
        # print(f"h_dims: {h_dims}, self.uni_diffusion: {self.uni_diffusion}")
        h_final = h_final.view(bs, n_nodes, -1)
        return torch.cat([vel, h_final], dim=2)

    def _bfnforward(self, t, xh, node_mask, edge_mask, context, t2=None, mask_y=None):
        bs, n_nodes, dims = xh.shape
        h_dims = dims - self.n_dims
        edges = self.get_adj_matrix(n_nodes, bs, self.device)
        edges = [x.to(self.device) for x in edges]
        node_mask = node_mask.view(bs*n_nodes, 1)
        edge_mask = edge_mask.view(bs*n_nodes*n_nodes, 1)
        xh = xh.view(bs*n_nodes, -1).clone() * node_mask
        x = xh[:, 0:self.n_dims].clone()
        
        if h_dims == 0:
            h = torch.ones(bs*n_nodes, 1).to(self.device)
        else:
            h = xh[:, self.n_dims:].clone()
        
        if self.atom_type_pred:
            assert (h == (torch.ones_like(h).detach() * node_mask)).all(), "h should not be all in 1"
        
        # h means charge
        mu_charge_t = h
        h = self.gaussian_basis(mu_charge_t) # bfn shape: X x 16
        
        
        
        
        h_time = t.view(bs, 1).repeat(1, n_nodes)
        h_time = h_time.view(bs * n_nodes, 1)
        h = torch.cat([h, h_time], dim=1)
        
        if context is not None and self.context_node_nf > 0:
            context = context.view(bs*n_nodes, self.context_node_nf)
            h = torch.cat([h, context], dim=1)
        
        h_final, x_final, org_h = self.egnn(h, x, edges, node_mask=node_mask, edge_mask=edge_mask)
        
        
        pass

    def gaussian_basis(self, x): # Form bfn
        """x: [batch_size, ...]"""
        # x = torch.unsqueeze(x, dim=-1)  # [batch_size, ..., 1]
        out = (x - self.centers) / self.width
        ret = torch.exp(-0.5 * out**2)

        return F.normalize(ret, dim=-1, p=1) * 2 - 1  # 


    def _forward(self, t, xh, node_mask, edge_mask, context, t2=None, mask_y=None, no_noise_xh=None):
        '''
        xh_shape: (bs, n_nodes, 9)
        '''
        # print("xh_shape: ", xh.shape)
        # print("xh[0]", xh[0])
        if self.mode == 'PAT':
            return self.PAT_forward(t, xh, node_mask, edge_mask, context, t2, mask_y)
        bs, n_nodes, dims = xh.shape
        h_dims = dims - self.n_dims
        edges = self.get_adj_matrix(n_nodes, bs, self.device)
        edges = [x.to(self.device) for x in edges]
        node_mask = node_mask.view(bs*n_nodes, 1)
        edge_mask = edge_mask.view(bs*n_nodes*n_nodes, 1)
        xh = xh.view(bs*n_nodes, -1).clone() * node_mask
        x = xh[:, 0:self.n_dims].clone()
        # if self.mode == "DGAP":
        #     assert h_dims == 6, "h_dims should be 6"
        if h_dims == 0:
            if self.atom_type_pred:
                h = torch.ones(bs*n_nodes, self.in_node_nf - 1).to(self.device) # erase the time, self in_node_nf contains the time
                h = h * node_mask
                xh = torch.cat([x, h], dim=1)
            else:
                h = torch.ones(bs*n_nodes, 1).to(self.device)
        else:
            h = xh[:, self.n_dims:].clone()
        if self.atom_type_pred:
            assert (h == (torch.ones_like(h).detach() * node_mask)).all(), "h should not be all in 1"
        else:
            assert not (h == (torch.ones_like(h).detach() * node_mask)).all(), "h should not be all in 1"
        if self.condition_time:
            if np.prod(t.size()) == 1:
                # t is the same for all elements in batch.
                h_time = torch.empty_like(h[:, 0:1]).fill_(t.item())
            else:
                # t is different over the batch dimension.
                h_time = t.view(bs, 1).repeat(1, n_nodes)
                h_time = h_time.view(bs * n_nodes, 1)
            h = torch.cat([h, h_time], dim=1)

        if context is not None:
            # We're conditioning, awesome!
            #一行完成assert not true
            # assert not (self.mode == "DGAP" or self.mode == "PAT"), "unconditional generation not supported have context"
            
            if self.uni_diffusion:
                # context = context.view(bs * n_nodes, 1)
                # context = self.gaussian_layer(context.squeeze())
                # # convert it to the 64 dim
                # context = self.property_emb(context)
                
                context = context.view(bs*n_nodes, -1)
                
                if self.use_basis:
                    context_basis = self.gaussian_layer(context.squeeze())
                    context_basis = self.property_emb(context_basis)
                    # context = torch.cat([context, context_basis], dim=1)
                
                
                # concat t2:
                c_time = t2.view(bs, 1).repeat(1, n_nodes)
                c_time = c_time.view(bs * n_nodes, 1)
                
                context_with_t = torch.cat([context, c_time], dim=1)
                if mask_y is not None and mask_y: # masking the property
                    context_with_t = self.mask_embedding(torch.zeros(1).long().to(t.device)).repeat(bs*n_nodes, 1)
                
                h = torch.cat([h, context_with_t], dim=1)
                
            
            elif self.finetune: # 
                context_emb = self.property_fc(context)
                context = context.view(bs*n_nodes, -1)
                context_emb = context_emb.view(bs*n_nodes, -1)
                h = torch.cat([h, context_emb, context], dim=1)
            
            elif self.context_node_nf > 0: # TODO eval
                context = context.view(bs*n_nodes, self.context_node_nf)
                print("context_node_nf: ", self.context_node_nf)
                h = torch.cat([h, context], dim=1)

            else:
                print(f"ERROR: cannot enbed context {context.shape} in {self.mode} mode")
        # print("h_shape: ", h.shape)
        if self.mode == 'egnn_dynamics':
            if self.use_basis:
                h_final, x_final, org_h = self.egnn(h, x, edges, node_mask=node_mask, edge_mask=edge_mask, context_basis=context_basis)
            else:
                h_final, x_final, org_h = self.egnn(h, x, edges, node_mask=node_mask, edge_mask=edge_mask)
                
            if self.decoupling:
                _, _, org_h = self.egnn2(h, x, edges, node_mask=node_mask, edge_mask=edge_mask)
                
                
            vel = (x_final - x) * node_mask  # This masking operation is redundant but just in case
            if self.uni_diffusion:
                # construct batch for scatter
                node_mask_b = node_mask.reshape(bs, n_nodes)
                atom_num_lst = node_mask_b.sum(dim=1)
                batch_lst = []
                for i, atom_num in enumerate(atom_num_lst):
                    current_lst = torch.full([int(atom_num.item())], i)
                    batch_lst.append(current_lst)
                batch = torch.cat(batch_lst).to(h_final.device)
                h_new_final = org_h[node_mask.squeeze().to(torch.bool)]
                pred = self.property_out(h_new_final)
                pred = scatter_mean(pred, batch, dim=0) # batch_size * embedding_size
                
                
            
        elif self.mode == 'gnn_dynamics':
            xh = torch.cat([x, h], dim=1)
            output = self.gnn(xh, edges, node_mask=node_mask)
            vel = output[:, 0:3] * node_mask
            h_final = output[:, 3:]
        elif self.mode == 'torchmdnet':
            # print('torchmdnet forward')
            node_mask_b = node_mask.reshape(bs, n_nodes)
            atom_num_lst = node_mask_b.sum(dim=1)
            
            vidx = node_mask.squeeze().to(torch.bool)
            pos = x[vidx]
            z = h[vidx]
            # generate the batch
            batch_lst = []
            half_batch_num = 0
            for i, atom_num in enumerate(atom_num_lst):
                current_lst = torch.full([int(atom_num.item())], i)
                if i < bs // 2:
                    half_batch_num += int(atom_num.item())
                
                batch_lst.append(current_lst)
            batch = torch.cat(batch_lst).to(pos.device)
            xo, vo, z, pos, batch = self.gnn(z, pos, batch=batch)
            noise_pred = self.noise_out.pre_reduce(xo, vo, z, pos, batch)
            if mask_y is not None and mask_y:
                noise_pred2 = torch.zeros_like(noise_pred)
                noise_pred2[:half_batch_num, :] = noise_pred[:half_batch_num, :]
                denoise_pred = self.noise_out2.pre_reduce(xo, vo, z, pos, batch)
                noise_pred2[half_batch_num:, :] = denoise_pred[half_batch_num:, :]
                noise_pred = noise_pred2
            
            
            h_pred = self.node_out(xo)
            
            if self.uni_diffusion:
            # predict the property
                pred = self.property_out(xo)
                pred = scatter_mean(pred, batch, dim=0)
                
            # h_final and vel
            vel = torch.zeros_like(x)
            h_final = torch.zeros_like(h)
            
            vel[vidx] = noise_pred
            h_final[vidx] = h_pred
            
            # print('batch')
        elif self.mode == "DGAP":
            if self.use_basis:
                h_final, x_final, org_h = self.egnn(h, x, edges, node_mask=node_mask, edge_mask=edge_mask, context_basis=context_basis)
            else:
                h_final, x_final, org_h = self.egnn(h, x, edges, node_mask=node_mask, edge_mask=edge_mask, batch_size=bs, n_nodes=n_nodes)
                
            if self.decoupling:
                _, _, org_h = self.egnn2(h, x, edges, node_mask=node_mask, edge_mask=edge_mask)
            
            if self.freeze_gradient:
                distances, _ = coord2diff(x, edges)
                org_h_pred = org_h
                tmp_x = x_final
                for module in self.property_layers:
                    org_h_pred, tmp_x = module(org_h_pred, tmp_x, edges, node_mask, edge_mask, distances)
            else:
                org_h_pred = org_h
                
            vel = (x_final - x) * node_mask  # This masking operation is redundant but just in case
            #TODO check the output dim
            
            if self.atom_type_pred:
                h_final = self.node_dec2(org_h)
                h_final = h_final.view(bs*n_nodes, -1)
            
            # print("self.atom_type4prop_pred: ", self.atom_type4prop_pred)
            if self.atom_type4prop_pred and no_noise_xh is not None:
                no_noise_xh = no_noise_xh.view(bs*n_nodes, -1)
                # print("no_noise_xh: ", no_noise_xh.shape, no_noise_xh[0])
                no_noise_h = no_noise_xh[:, self.n_dims:-1].clone()
                assert no_noise_h.shape[1] == self.in_node_nf - 2, "no_noise_h should have the same size as h. no_noise_h.shape: %s, self.in_node_nf - 2: %s" % (str(no_noise_h.shape), str(self.in_node_nf - 2))
                # atom_type_decode = self.atom_type4prop_pred_decoder(no_noise_h)
                atom_type_decode = self.egnn.embedding2(no_noise_h)
                distances, _ = coord2diff(x, edges)
                no_noise_h_emb, _ = self.egnn.atomtype_add_layer(atom_type_decode, x, edges, node_mask=node_mask, edge_mask=edge_mask, edge_attr=distances)
                
                node_dec = self.node_dec(torch.cat([org_h_pred, no_noise_h_emb], dim=1))
            elif self.atom_type4prop_pred: # no no_noise_xh, meaning it's in sampling steps
                node_dec = self.node_dec(torch.cat([org_h_pred, torch.zeros_like(org_h_pred)], dim=1))
            else:
                node_dec = self.node_dec(org_h_pred)            
            # print("before node_dec: ", node_dec.shape)
            node_dec = node_dec.view(bs, n_nodes, self.hidden_nf)
            # print("node_dec: ", node_dec.shape)
            node_dec = torch.sum(node_dec, dim=1)
            # print("after node_dec: ", node_dec.shape)
            pred = self.graph_dec(node_dec)
            if self.use_ref and no_noise_xh is not None:
                # return 1 index in the org_h0
                atom_types = torch.argmax(no_noise_h, dim=1)
                atoms_pred = self.atomref(atom_types)
                atoms_pred = atoms_pred * node_mask
                atoms_pred = atoms_pred.view(-1, n_nodes)
                atoms_pred = torch.sum(atoms_pred, dim=1)
                pred = pred + atoms_pred.unsqueeze(1)
            # print("before pred: ", pred.shape)
            pred = pred.squeeze(1) 
            # print("after pred: ", pred.shape)
            # if pred.device == torch.device('cuda:0'):
            #     print("pred before: ", pred) 
            if self.train_prop_pred_4condition_only: # 在t>10的位置上 覆盖原来的pred
                blur_node_dec = self.blur_node_decode(org_h_pred).view(bs, n_nodes, self.hidden_nf)
                blur_node_dec = torch.sum(blur_node_dec, dim=1)
                blur_prop_pred = self.blur_graph_decode(blur_node_dec)
                blur_prop_pred = blur_prop_pred.squeeze(1)
                # if blur_prop_pred.device == torch.device('cuda:0'):
                #     print("blur_prop_pred: ", blur_prop_pred)
                # if t.device == torch.device('cuda:0'):
                #     print("t: ", t.squeeze(1) * 1000)
                assert pred.shape == blur_prop_pred.shape, "pred.shape: %s, blur_prop_pred.shape: %s" % (str(pred.shape), str(blur_prop_pred.shape))
                # when (t * self.T) > threshold, use blur_prop_pred, otherwise use pred
                pred = torch.where((t.squeeze(1) * 1000) > 10, blur_prop_pred, pred)      
                if (t.squeeze(1) * 1000).max() < 100 and (t.squeeze(1) * 1000).min() > 10:
                    # print("blur_prop_pred: ", blur_prop_pred)
                    # print("condition generation t: ",t.squeeze(1)*1000)
                    assert (pred == blur_prop_pred).all(), "pred should be equal to blur_prop_pred"
        else:
            raise Exception("Wrong mode %s" % self.mode)
        
        if context is not None and self.context_node_nf > 0:
            # Slice off context size:
            h_final = h_final[:, :-self.context_node_nf]

        if self.condition_time:
            # Slice off last dimension which represented time.
            h_final = h_final[:, :-1]

        vel = vel.view(bs, n_nodes, -1)

        if torch.any(torch.isnan(vel)):
            print('Warning: detected nan, resetting EGNN output to zero.')
            vel = torch.zeros_like(vel)

        if node_mask is None:
            vel = remove_mean(vel)
        else:
            vel = remove_mean_with_mask(vel, node_mask.view(bs, n_nodes, 1))
        # print(f"h_dims: {h_dims}, self.uni_diffusion: {self.uni_diffusion}")
        if h_dims == 0 and not self.property_pred and not self.atom_type_pred:
            return vel
        else:
            h_final = h_final.view(bs, n_nodes, -1)
            if self.uni_diffusion:
                return torch.cat([vel, h_final], dim=2), pred
            if self.property_pred:
                return (torch.cat([vel, h_final], dim=2), pred)
            return torch.cat([vel, h_final], dim=2)

    def get_adj_matrix(self, n_nodes, batch_size, device):
        if n_nodes in self._edges_dict:
            edges_dic_b = self._edges_dict[n_nodes]
            if batch_size in edges_dic_b:
                return edges_dic_b[batch_size]
            else:
                # get edges for a single sample
                rows, cols = [], []
                for batch_idx in range(batch_size):
                    for i in range(n_nodes):
                        for j in range(n_nodes):
                            rows.append(i + batch_idx * n_nodes)
                            cols.append(j + batch_idx * n_nodes)
                edges = [torch.LongTensor(rows).to(device),
                         torch.LongTensor(cols).to(device)]
                edges_dic_b[batch_size] = edges
                return edges
        else:
            self._edges_dict[n_nodes] = {}
            return self.get_adj_matrix(n_nodes, batch_size, device)
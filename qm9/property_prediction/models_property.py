from .models.gcl import E_GCL, unsorted_segment_sum
import torch
from torch import nn
from priors import Atomref


class E_GCL_mask(E_GCL):
    """Graph Neural Net with global state and fixed number of nodes per graph.
    Args:
          hidden_dim: Number of hidden units.
          num_nodes: Maximum number of nodes (for self-attentive pooling).
          global_agg: Global aggregation function ('attn' or 'sum').
          temp: Softmax temperature.
    """

    def __init__(self, input_nf, output_nf, hidden_nf, edges_in_d=0, nodes_attr_dim=0, act_fn=nn.ReLU(), recurrent=True, coords_weight=1.0, attention=False):
        E_GCL.__init__(self, input_nf, output_nf, hidden_nf, edges_in_d=edges_in_d, nodes_att_dim=nodes_attr_dim, act_fn=act_fn, recurrent=recurrent, coords_weight=coords_weight, attention=attention)

        del self.coord_mlp
        self.act_fn = act_fn

    def coord_model(self, coord, edge_index, coord_diff, edge_feat, edge_mask):
        row, col = edge_index
        trans = coord_diff * self.coord_mlp(edge_feat) * edge_mask
        agg = unsorted_segment_sum(trans, row, num_segments=coord.size(0))
        coord += agg*self.coords_weight
        return coord

    def forward(self, h, edge_index, coord, node_mask, edge_mask, edge_attr=None, node_attr=None, n_nodes=None):
        row, col = edge_index
        radial, coord_diff = self.coord2radial(edge_index, coord)

        edge_feat = self.edge_model(h[row], h[col], radial, edge_attr)

        edge_feat = edge_feat * edge_mask

        # TO DO: edge_feat = edge_feat * edge_mask

        #coord = self.coord_model(coord, edge_index, coord_diff, edge_feat, edge_mask)
        h, agg = self.node_model(h, edge_index, edge_feat, node_attr)

        return h, coord, edge_attr



class EGNN(nn.Module):
    def __init__(self, in_node_nf, in_edge_nf, hidden_nf, device='cpu', act_fn=nn.SiLU(), n_layers=4, coords_weight=1.0, attention=False, node_attr=1):
        super(EGNN, self).__init__()
        self.hidden_nf = hidden_nf
        self.device = device
        self.n_layers = n_layers

        ### Encoder
        self.embedding = nn.Linear(in_node_nf, hidden_nf)
        self.node_attr = node_attr
        if node_attr:
            n_node_attr = in_node_nf
        else:
            n_node_attr = 0
        for i in range(0, n_layers):
            self.add_module("gcl_%d" % i, E_GCL_mask(self.hidden_nf, self.hidden_nf, self.hidden_nf, edges_in_d=in_edge_nf, nodes_attr_dim=n_node_attr, act_fn=act_fn, recurrent=True, coords_weight=coords_weight, attention=attention))

        self.node_dec = nn.Sequential(nn.Linear(self.hidden_nf, self.hidden_nf),
                                      act_fn,
                                      nn.Linear(self.hidden_nf, self.hidden_nf))

        self.graph_dec = nn.Sequential(nn.Linear(self.hidden_nf, self.hidden_nf),
                                       act_fn,
                                       nn.Linear(self.hidden_nf, 1))
        self.to(self.device)

    def forward(self, h0, x, edges, edge_attr, node_mask, edge_mask, n_nodes):
        h = self.embedding(h0)
        for i in range(0, self.n_layers):
            if self.node_attr:
                h, _, _ = self._modules["gcl_%d" % i](h, edges, x, node_mask, edge_mask, edge_attr=edge_attr, node_attr=h0, n_nodes=n_nodes)
            else:
                h, _, _ = self._modules["gcl_%d" % i](h, edges, x, node_mask, edge_mask, edge_attr=edge_attr,
                                                      node_attr=None, n_nodes=n_nodes)

        h = self.node_dec(h)
        h = h * node_mask
        h = h.view(-1, n_nodes, self.hidden_nf)
        h = torch.sum(h, dim=1)
        pred = self.graph_dec(h)
        return pred.squeeze(1)



class EGNN(nn.Module):
    def __init__(self, in_node_nf, in_edge_nf, hidden_nf, device='cpu', act_fn=nn.SiLU(), n_layers=4, coords_weight=1.0, attention=False, node_attr=1, atom_disturb=False, later_fusion_h=False, use_ref=False):
        super(EGNN, self).__init__()
        self.in_node_nf = in_node_nf
        self.hidden_nf = hidden_nf
        # print("embedding structure: ", self.in_node_nf, self.hidden_nf)
        print("device for EGNN: ", device)
        self.device = device
        self.n_layers = n_layers
        
        self.atom_disturb = atom_disturb
        self.later_fusion_h = later_fusion_h

        ### Encoder
        self.embedding = nn.Linear(in_node_nf, hidden_nf)
        self.node_attr = node_attr
        if node_attr:
            n_node_attr = in_node_nf
        else:
            n_node_attr = 0
        for i in range(0, n_layers):
            self.add_module("gcl_%d" % i, E_GCL_mask(self.hidden_nf, self.hidden_nf, self.hidden_nf, edges_in_d=in_edge_nf, nodes_attr_dim=n_node_attr, act_fn=act_fn, recurrent=True, coords_weight=coords_weight, attention=attention))

        self.node_dec = nn.Sequential(nn.Linear(self.hidden_nf, self.hidden_nf),
                                      act_fn,
                                      nn.Linear(self.hidden_nf, self.hidden_nf))

        if self.later_fusion_h: # For cat
            self.embedding2 = nn.Linear(in_node_nf, hidden_nf)
            self.atomtype_add_layer = E_GCL_mask(self.hidden_nf, self.hidden_nf, self.hidden_nf, edges_in_d=in_edge_nf, nodes_attr_dim=n_node_attr, act_fn=act_fn, recurrent=True, coords_weight=coords_weight, attention=attention)           
            self.node_dec = nn.Sequential(
                nn.Linear(hidden_nf * 2, hidden_nf),
                act_fn,
                nn.Linear(hidden_nf,  hidden_nf),
            )

        self.graph_dec = nn.Sequential(nn.Linear(self.hidden_nf, self.hidden_nf),
                                       act_fn,
                                       nn.Linear(self.hidden_nf, 1))

        self.atom_disturb = atom_disturb
        self.later_fusion_h = later_fusion_h
        if self.later_fusion_h:
            self.atom_type4prop_pred_decoder = nn.Sequential(
                nn.Linear(self.in_node_nf, hidden_nf), # exclude time
                act_fn,
                nn.Linear(hidden_nf, hidden_nf), # in_node_nf = 6 contains time; (or 7 include charge) context_node_nf: conditional generation
            ) # for atom prediction
        
        self.use_ref = use_ref
        if self.use_ref:
            self.atomref = Atomref(max_z=100)    
         
        self.to(self.device)

    def forward(self, h0, x, edges, edge_attr, node_mask, edge_mask, n_nodes):
        '''
        h0 [batchsize*n_nodes, 22]
        x [batchsize*n_nodes, 3]
        edges list, len = 2 (from - to)   size = batchsize * n_nodes * n_nodes
        node_mask [batchsize, 1]
        edge_mask [batchsize*n_nodes*n_nodes, 1]
        n_nodes: int
        '''
        # print(f"input type: h0 type: {type(h0)}, x type: {type(x)}, edges type: {type(edges)}, edge_attr type: {type(edge_attr)}, node_mask type: {type(node_mask)}, edge_mask type: {type(edge_mask)}, n_nodes type: {type(n_nodes)}")
        # print(f"input dim: h0 size: {h0.size()}, x size: {x.size()}, edges size: {len(edges)}, edge_attr: {edge_attr}, node_mask size: {node_mask.size()}, edge_mask size{edge_mask.size()}, n_nodes size: {n_nodes}")
        # print("edge size: ", edges[0].size(), edges[1].size())
        if self.atom_disturb:
            org_h0 = h0.clone()
            h0 = torch.ones_like(h0) # simple disturb all atom types
        
        h = self.embedding(h0)
        for i in range(0, self.n_layers):
            if self.node_attr:
                h, _, _ = self._modules["gcl_%d" % i](h, edges, x, node_mask, edge_mask, edge_attr=edge_attr, node_attr=h0, n_nodes=n_nodes)
            else:
                h, _, _ = self._modules["gcl_%d" % i](h, edges, x, node_mask, edge_mask, edge_attr=edge_attr,
                                                      node_attr=None, n_nodes=n_nodes)

        if self.later_fusion_h:
            # org_h = self.atom_type4prop_pred_decoder(org_h0)
            org_h = self.embedding2(org_h0)
            org_h, _, _ = self.atomtype_add_layer(org_h, edges, x, node_mask, edge_mask, edge_attr=edge_attr, node_attr=None, n_nodes=n_nodes)
            
            h = torch.cat([h, org_h], dim=1)
            h = self.node_dec(h)
        else:
            h = self.node_dec(h)
        h = h * node_mask
        h = h.view(-1, n_nodes, self.hidden_nf)
        h = torch.sum(h, dim=1)
        pred = self.graph_dec(h)
        
        if self.use_ref:
            # return 1 index in the org_h0
            atom_types = torch.argmax(org_h0, dim=1)
            atoms_pred = self.atomref(atom_types)
            atoms_pred = atoms_pred * node_mask
            atoms_pred = atoms_pred.view(-1, n_nodes)
            atoms_pred = torch.sum(atoms_pred, dim=1)
            pred = pred + atoms_pred.unsqueeze(1)
            return pred.squeeze(1)
        return pred.squeeze(1)



class Naive(nn.Module):
    def __init__(self, device):
        super(Naive, self).__init__()
        self.device = device
        self.linear = nn.Linear(1, 1)
        self.to(self.device)

    def forward(self, h0, x, edges, edge_attr, node_mask, edge_mask, n_nodes):
        node_mask = node_mask.view(-1, n_nodes)
        bs, n_nodes = node_mask.size()
        x = torch.zeros(bs, 1).to(self.device)
        return self.linear(x).squeeze(1)


class NumNodes(nn.Module):
    def __init__(self, device, nf=128):
        super(NumNodes, self).__init__()
        self.device = device
        self.linear1 = nn.Linear(1, nf)
        self.linear2 = nn.Linear(nf, 1)
        self.act_fn = nn.SiLU()
        self.to(self.device)

    def forward(self, h0, x, edges, edge_attr, node_mask, edge_mask, n_nodes):
        reshaped_mask = node_mask.view(-1, n_nodes)
        nodesxmol = torch.sum(reshaped_mask, dim=1).unsqueeze(1)/29
        x = self.act_fn(self.linear1(nodesxmol))
        return self.linear2(x).squeeze(1)
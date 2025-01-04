# -*- encoding: utf-8 -*-
'''
@File    :   model.py
@Time    :   2023/11/07 19:12:30
@Author  :   Fei Gao
'''
import torch
import torch.nn as nn
from torch_geometric.utils import scatter

class BatchNorm(nn.Module):
    def __init__(self, num_features, momentum=0.1, eps=1e-5):
        super(BatchNorm, self).__init__()
        self.bn = nn.BatchNorm1d(num_features, momentum=momentum, eps=eps)
    
    def forward(self, x):
        # x: [batch_size, num_nodes, num_features]
        # convert to [batch_size, num_features, num_nodes]
        x = x.transpose(1, 2)
        x = self.bn(x)
        return x.transpose(1, 2)
        

class MessagePassing(nn.Module):
    def __init__(self,
                 input_size:int,
                 hidden_size:int,
                 activation:nn.Module,
                 num_molecules:int,
                 num_reactions:int,
                 dropout:float):
        super(MessagePassing, self).__init__()
        self.num_molecules = num_molecules
        self.num_reactions = num_reactions
        self.dropout = dropout
        
        self.reactant2reaction = nn.Linear(input_size, hidden_size)
        self.product2reaction  = nn.Linear(input_size, hidden_size)
        self.reactionAggregation = nn.Sequential(nn.Linear(hidden_size, hidden_size), 
                                                 activation,
                                                 BatchNorm(hidden_size),
                                                 nn.Dropout(self.dropout),
                                                 nn.Linear(hidden_size, hidden_size))
        
        self.reaction2reactant = nn.Linear(hidden_size, hidden_size)
        self.reaction2product  = nn.Linear(hidden_size, hidden_size)
        self.moleculeAggregation = nn.Sequential(nn.Linear(hidden_size, hidden_size),
                                                 activation,
                                                 BatchNorm(hidden_size),
                                                 nn.Dropout(self.dropout),
                                                 nn.Linear(hidden_size, hidden_size))
        
    
    # def forward(self, nodes_input, edges, molecule_embs, reaction_embs):
    def forward(self, nodes_input, edges):
        # reactants2reaction, products2reaction, reactions2reactants, reactions2products = edges
        r2e, p2e, e2r, e2p = edges
        # nodes_input --> reactants_hidden, products_hidden --> reaction_hidden
        reactants_hidden = self.reactant2reaction(nodes_input) 
        products_hidden = self.product2reaction(nodes_input)
        
        reactants_agg = scatter(reactants_hidden[:, r2e[0]], r2e[1], dim=1, reduce='mean',
                                dim_size=self.num_reactions)
        products_agg = scatter(products_hidden[:, p2e[0]], p2e[1], dim=1, reduce='mean',
                               dim_size=self.num_reactions)
        aggregation = reactants_agg + products_agg
        reaction_hidden = self.reactionAggregation(aggregation)
        
        # reaction_hidden --> reactants_hidden, products_hidden -> nodes_output        
        reactants_hidden = self.reaction2reactant(reaction_hidden)
        products_hidden = self.reaction2product(reaction_hidden)
        
        reactants_agg = scatter(reactants_hidden[:, e2r[0]], e2r[1], dim=1, reduce="mean",
                                dim_size=self.num_molecules)
        products_agg = scatter(products_hidden[:, e2p[0]], e2p[1], dim=1, reduce="mean",
                               dim_size=self.num_molecules)
        nodes_output = self.moleculeAggregation(reactants_agg + products_agg)
        
        return nodes_output
    
class HyperMP(nn.Module):
    def __init__(self,
                 in_dim: int,
                 out_dim: int,
                 hid_dim: int,
                 num_message_passing: int,
                 num_molecules: int,
                 num_reactions: int,
                 dropout: float = 0.3):
        super().__init__()
        self.num_message_passing = num_message_passing
        self.dropout = dropout
        self.input_layer = nn.Linear(in_dim, hid_dim)
        self.message_passing_layers = nn.ModuleList([MessagePassing(hid_dim, hid_dim,
                                                                    nn.ReLU(),
                                                                    num_molecules,
                                                                    num_reactions,
                                                                    self.dropout)
                                                     for _ in range(num_message_passing)])
        
        self.output_layer =  nn.Sequential(nn.Linear(hid_dim, hid_dim),
                                           nn.ReLU(),
                                           nn.Linear(hid_dim, out_dim))
        # self.loss_fun = nn.MSELoss()
        self.loss_fun = nn.L1Loss()
        
    
    def forward(self, node_inputs, edges):
        # prepare the node inputs
        node_hidden = self.input_layer(node_inputs)
        node_hidden_init = node_hidden.clone()
        # message passing
        for layer in range(self.num_message_passing):
            mp = self.message_passing_layers[layer]
            new_node_hidden = mp(node_hidden, edges)
            node_hidden = node_hidden + new_node_hidden
        # add the inital node information 
        node_out = node_hidden + node_hidden_init
        pred = self.output_layer(node_out)
        return pred
    

    def loss(self, y, y_hat):
        return self.loss_fun(y_hat, y)
    
    @staticmethod
    def conc_mse(x, y, y_hat, test_dataset, dim=None):
        x, y, y_hat = [t.detach().cpu() for t in [x, y, y_hat]]
        init_conc, final_conc = test_dataset.log_transform_scale_and_relative_change(x, y, inverse=True)
        _, pred_final_conc = test_dataset.log_transform_scale_and_relative_change(x, y_hat, inverse=True)
        
        pred_mse = torch.mean((final_conc - pred_final_conc) ** 2, dim=dim)
        base_mse = torch.mean((final_conc - init_conc) ** 2, dim=dim)
        
        return pred_mse, base_mse
    
    @staticmethod
    def conc_r2(x, y, y_hat, test_dataset, dim=0):
        x, y, y_hat = [t.detach().cpu() for t in [x, y, y_hat]]
        init_conc, final_conc = test_dataset.log_transform_scale_and_relative_change(x, y, inverse=True)
        _, pred_final_conc = test_dataset.log_transform_scale_and_relative_change(x, y_hat, inverse=True)
        
        ss_tot = torch.sum((final_conc - torch.mean(final_conc, dim=dim, keepdim=True)) ** 2, dim=dim)
        
        pred_ss_res = torch.sum((final_conc - pred_final_conc) ** 2, dim=dim)
        pred_r2 = 1 - pred_ss_res / (ss_tot + 1e-30)
        
        base_ss_res = torch.sum((final_conc - init_conc) ** 2, dim=dim)
        base_r2 = 1 - base_ss_res / (ss_tot + 1e-30)
        

        n_ss_tot = torch.sum((x+y - torch.mean(x+y, dim=dim, keepdim=True)) ** 2, dim=dim)
        
        n_pred_ss_res = torch.sum((y - y_hat) ** 2, dim=dim)
        n_pred_r2 = 1 - n_pred_ss_res / (n_ss_tot + 1e-30)
        
        # n_base_ss_res = torch.sum((y - x) ** 2, dim=dim)
        # n_base_ss_res = torch.sum((y - x) ** 2, dim=dim) # not quite good
        n_base_ss_res = torch.sum((y) ** 2, dim=dim)
        n_base_r2 = 1 -n_base_ss_res / (n_ss_tot + 1e-30)
        
        return pred_r2, base_r2, n_pred_r2, n_base_r2
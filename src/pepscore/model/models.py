from typing import Optional, Iterable
import torch
from torch import nn
import torch.nn.functional as F
import torch.nn.init as init

class SelfAttentionPooling(nn.Module):
    """Self attention pooling."""
    def __init__(self,
                 input_dim: int,
                 attention_heads: int = 16,
                 attention_units: Optional[Iterable[int]] = None,
                 output_activation: Optional[torch.nn.Module] = None,
                 hidden_activation: Optional[torch.nn.Module] = None,
                 input_dropout: float = 0.1,
                 attention_dropout: float = 0.1,
                 ):
        """Initialize a self attention pooling layer

        Parameters
        ----------
        input_dim : int
            The input data dim
        attention_heads: int
            the number of attn heads
        attention_units: Iterable[int]
            the list of hidden dimensions of the MLP computing the attn
        input_dropout: float
            dropout applied to the data argument of the forward method.
        attention_dropout: float
            dropout applied to the attention output before applying it
            to the input for reduction. decouples the attn dropout
            from the input dropout
        """
        super().__init__()
        # creating the MLP
        dimensions = [input_dim, *attention_units, attention_heads]
        self.input_dim = input_dim
        self.dropout = nn.Dropout(input_dropout) if input_dropout > 0. else nn.Identity()
        
        self.in_drop = nn.Dropout(input_dropout) if input_dropout > 0. else nn.Identity()
        self.layer_norm = nn.LayerNorm(input_dim)
        layers = []
        for l in range(len(dimensions) - 2):
            layers.append(nn.Linear(dimensions[l], dimensions[l+1], bias=False))
            init.xavier_uniform_(layers[-1].weight)
            layers.append(nn.Tanh() if hidden_activation is None else hidden_activation)
        layers.append(nn.Linear(dimensions[-2], dimensions[-1], bias=False))
        init.xavier_uniform_(layers[-1].weight)
        
        if attention_dropout > 0.:
            layers.append(nn.Dropout(attention_dropout))
        self.mlp = nn.Sequential(*layers)
        self.output_activation = nn.Softmax(dim=1) \
            if output_activation is None else output_activation

    def forward(self,
                data: torch.Tensor,
                padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Performs a forward pass.

        Parameters
        ----------
        data : torch.Tensor
            The input data, as a tensor of shape [B x S x H]
        padding_mask: torch.Tensor
            The input padding_mask, as a tensor of shape [B X S]

        Returns
        ----------
        torch.Tensor
            The output data, as a tensor of shape [B x H]

        """
        # input_tensor is 3D float tensor, batchsize x num_encs x dim
        batch_size, num_encs, dim = data.shape
        # apply input droput
        data = self.in_drop(data)

        data = self.layer_norm(data)
        # apply projection and reshape to batchsize x num_encs x num_heads
        attention_logits = self.mlp(data.reshape(-1, dim)).reshape(batch_size, num_encs, -1)
        #add self.dropout on attention logits
        attention_logits = self.dropout(attention_logits)
        attention_logits = self.output_activation(attention_logits)
        #print("attention_logits", attention_logits.shape)
        #print("padding_mask", padding_mask.unsqueeze(2).shape)
        # apply padding_mask. dimension stays batchsize x num_encs x num_heads
        """
        if padding_mask is not None:
            padding_mask = padding_mask.unsqueeze(2).float()
            attention_logits = attention_logits * padding_mask + (1. - padding_mask) * -1e20
        """
        # apply softmax. dimension stays batchsize x num_encs x num_heads
        
        attention = self.output_activation(attention_logits)
        
        # attend. attention is batchsize x num_encs x num_heads. data is batchsize x num_encs x dim
        # resulting dim is batchsize x num_heads x dim
        attended = torch.bmm(attention.transpose(1, 2), data) 
        # average over attention heads and return. dimension is batchsize x dim
        return attended.mean(dim=1)


class ReduceLayer(nn.Module):
    """Implement an sigmoid module.

    Can be used to form a classifier out of any encoder.
    Note: by default takes the log_softmax so that it can be fed to
    the NLLLoss module. You can disable this behavior through the
    `take_log` argument.

    """
    def __init__(self, 
                pool: str='average', 
                reduce_dim: int = 1, 
                padding_idx: Optional[int] = 0) -> None:
        """Initialize the SoftmaxLayer.

        Parameters
        ----------
        """
        super().__init__()
        # output of nn.embedding: B X S X E
        # input and output of RNN: S X B X H
        # Padding mask: B X S
        self.reduce_dim = reduce_dim  # Most of time, output is B x S x E, with seqlength on dimension 1
        self.pool = pool
        # self.padding_idx = padding_idx


    def forward(self,
                data: torch.Tensor,
                state: Optional[torch.Tensor] = None,
                padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Perform a forward pass through the network.

        Parameters
        ----------
        data : torch.Tensor
            The input data, as a float tensor of shape [B x S x E]
        state: Tensor
            An optional previous state of shape [L x B x H]
        padding_mask: Tensor, optional
            The padding mask of shape [B x S]

        Returns
        -------
        torch.Tensor
            The encoded output, as a float tensor of shape [B x H]

        """
        output = data
        # print('input')
        # print(output.shape)
        if padding_mask is None:
            padding_mask = torch.ones(*output.shape[:2]).to(output)
            
        # print('mask')
        # print(padding_mask.shape)

        # cast(torch.Tensor, padding_mask)
        if self.pool == 'average':
            # print(padding_mask.shape)
            # print(data.shape)
            padding_mask = padding_mask.unsqueeze(2)
            output = (output * padding_mask).sum(dim=self.reduce_dim) #BXE
            output = output / padding_mask.sum(dim=self.reduce_dim)
        elif self.pool == 'sum':
            output = (output * padding_mask.unsqueeze(2)).sum(dim=self.reduce_dim)
        elif self.pool == 'last':
            lengths = padding_mask.long().sum(dim=self.reduce_dim)
            output = output[torch.arange(output.size(0)).long(), lengths - 1, :]
        elif self.pool == 'first':
            output = output[torch.arange(output.size(0)).long(), 0, :]
        elif self.pool == 'sqrt_reduction':
            '''original implementation can be found here 
            https://github.asapp.dev/aganatra/nlp/blob/master/src/agnlp/utils/sqrt_n_reduction.py'''
            padding_mask = padding_mask.unsqueeze(2)
            output = (output * padding_mask).sum(dim=self.reduce_dim) #BXE
            output = output/sqrt(padding_mask.sum(dim=self.reduce_dim).float())
        # elif self.pool == 'decay':
        #     xxxx
        else:
            pool = self.pool
            print(pool)
            raise ValueError(f"Invalid pool type: {pool}")

        return output

import numpy as np

class GraphAttention(nn.Module):
    """
    GAT + EFA layer
    """

    def __init__(self, rec_features, pep_features, edge_features, emb_features,  out_features, dropout, alpha):
        #flowchart GAT
        print("init_parameters", rec_features, pep_features, edge_features, emb_features, out_features,  dropout, alpha)
        super(GraphAttention, self).__init__()
        
        self.dropout = nn.Dropout(dropout)
        self.alpha = alpha
        self.bn = torch.nn.BatchNorm1d(out_features)
        
        #embedding node & edge features

        #Linear transformation -> making node, edge hidden features
        self.Wp = nn.Linear(emb_features, out_features) # makes hi #[20, 12]
        self.Wr = nn.Linear(2 * emb_features, out_features) # makes hj
        #self.We = nn.Linear(emb_features, out_features) # makes eij

        self.Wv = nn.Linear(2* emb_features, out_features) #makes value 
        self.qk_dim = out_features

        #xavier_uniform_ for initializing weights
        init.xavier_uniform_(self.Wp.weight)
        init.xavier_uniform_(self.Wr.weight)
        init.xavier_uniform_(self.Wv.weight)
        
        self.pep_ln = nn.LayerNorm(emb_features)
        self.rec_ln = nn.LayerNorm(emb_features)

        self.activation = nn.LeakyReLU(self.alpha)

    def forward(self, rec_node_feat, pep_node_feat, edge_feat):
        #convert input tensor for double type
        #rec_node_feat = rec_node_feat.double()
        #pep_node_feat = pep_node_feat.double()
        #edge_feat = edge_feat.double()
        #print("pep_node_feat.shape", pep_node_feat.dtype)
        #print("pep_emb_module data type", self.pep_emb.weight.dtype)
        N_batch, N_pep, num_pep_node = pep_node_feat.shape
        N_rec = 10
        #rec_node_feat shape: (N_data, 12, 10, 42)
        #pep_node_feat shape: (N_data, 12, 42)
        #edge_feat shape: (N_data, 12, 10, 67)

        #query = peptide node_feature
        #value = receptor node_feature + edge_feature
        #key = receptor node_feature + edge_feature
        pep_emb = pep_node_feat
        rec_emb = rec_node_feat
        edge_emb = edge_feat

        #query -> mlp on peptide node_feature
        pep_query = self.Wp(pep_emb) #[N_data, 12, 32] * [32, 32] = [N_data, 12, 32]
        pep_query = pep_query.unsqueeze(2).repeat(1,1,N_rec,1) #[N_data, 12, N_rec, 32]

        
        #key -> 1) mlp on receptor node_feature  2) mlp on edge feature concat 1) & 2)
        #rec_key = self.Wr(rec_emb) #[N_data, 12, N_rec, 32] * [32, 32] = [N_data, 12, N_rec, 32]
        #edge_key = self.We(edge_emb) #[N_data, 12, N_rec, 32] * [32, 32] = [N_data, 12, N_rec, 32]

        #value = self.Wv on concatenated (rec_emb + edge_emb)
        #concat rec_emb + edge_emb
        rec_edge_emb = torch.cat((rec_emb, edge_emb), dim = 3) #[N_data, 12, N_rec, 64]
        value = self.Wv(rec_edge_emb) #[N_data, 12, N_rec, 64] * [64, 32] = [N_data, 12, N_rec, 32]


        #concat rec_key & edge_key
        key = self.Wr(torch.cat((rec_emb, edge_emb), dim = 3)) #[N_data, 12, N_rec, 64]

        #attention coeffient between pep_node_feat & rec_node_feat( attention on dim = 2 -> 10)
        pep_query = pep_query.view(N_batch * N_pep, 10, -1) #[N_data * 12, N_rec, 32])
        key = key.view(N_batch * N_pep, 10, -1) #[N_data * 12, N_rec, 64])

        #attention_logit = query * key / sqrt(d_k)
        #print the dimension of pep_query & key
        #print("pep_query.shape", pep_query.shape)
        #print("key.shape", key.shape)
        attention_logit = pep_query * key / np.sqrt(self.qk_dim) #[N_data * 12, 10, 64]
        #add self.dropout on attention_logit
        attention_logit = self.dropout(attention_logit) #[N_data * 12, 10, 64]
        attention_logit = attention_logit.sum(dim = -1) #[N_data * 12, 10]

        #softmax on attention_logit
        attention = F.softmax(attention_logit, dim = 1) #[N_data * 12, 10]
        attention = attention.view(N_batch, N_pep, 10, 1) #[N_data, 12, 10, 1]
        #apply attention on value -> concatenated hidden edge + rec node feature  [N_data, 12, N_rec, 32]
        value =  attention * value#[N_data, 12, N_rec, 32]

        # total updated pep_feature xi' = w_p * xi + sum_j (eij * hj)
        pep_node_feat = self.Wp(pep_emb) + value.sum(dim = 2) #[N_data, 12, 32] + [N_data, 12, 32] = [N_data, 12, 32]

        return pep_node_feat #[N_data, 12, 32]




class MLP_3_layer(nn.Module):
    def __init__(
        self,
        input_dim,
        hidden_dim,
        output_dim,
    ):
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        
        self.act = nn.ReLU()
        self.dropout = nn.Dropout(0.2)
        self.linear1 = nn.Linear(input_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.linear3 = nn.Linear(hidden_dim, output_dim)

    def forward(self, h):
        out = h
        h = self.linear1(h)
        h = self.act(h)
        h = self.linear2(h)
        h = self.act(h)
        h = self.linear3(h)
        return h



import torch
import torch.nn as nn
import torch.nn.functional as F
    
class FeatureExtractionModel_sep(nn.Module):
    def __init__(self, rec_features, pep_features, edge_features, emb_features, esm_emb_features, out_features, attention_heads, dropout, alpha, num_layers):
        super().__init__()
        
        #Define ESM embedding (separately for pep & rec)
        # 1280 -> esm_emb_features
        self.rec_esm_embedding = nn.Linear(1280, esm_emb_features, bias=False)
        self.pep_esm_embedding = nn.Linear(1280, esm_emb_features, bias=False)
        
        # embedding node & edge features
        self.pep_emb = nn.Linear(pep_features + esm_emb_features, emb_features, bias=False)
        self.rec_emb = nn.Linear(rec_features + esm_emb_features, emb_features, bias=False)
        self.edge_emb = nn.Linear(edge_features, emb_features, bias=False)
        
        #xavier_uniform_ all embedding linear layers
        init.xavier_uniform_(self.rec_esm_embedding.weight)
        init.xavier_uniform_(self.pep_esm_embedding.weight)
        init.xavier_uniform_(self.pep_emb.weight)
        init.xavier_uniform_(self.rec_emb.weight)
        init.xavier_uniform_(self.edge_emb.weight)
        
        self.pep_ln = nn.LayerNorm(emb_features)
        self.rec_ln = nn.LayerNorm(emb_features)

        self.graph_attentions = nn.ModuleList([GraphAttention(rec_features + esm_emb_features, pep_features + esm_emb_features, edge_features, emb_features, out_features, dropout, alpha) for _ in range(num_layers)])
        self.self_attention = SelfAttentionPooling(out_features, attention_heads, attention_units=[out_features, int(out_features / 2)], input_dropout=dropout)
        
        self.mlp = nn.Linear(out_features, int(out_features/2))

    def forward(self, pep_feat, rec_feat, edge_feat):
        rec_esm_feat = rec_feat[:, :, :, 42:]
        pep_esm_feat = pep_feat[:, :, -1280:]
        rec_esm_emb = self.rec_esm_embedding(rec_esm_feat)
        pep_esm_emb = self.pep_esm_embedding(pep_esm_feat)

        rec_feat = torch.cat((rec_feat[:, :, :, :21], rec_esm_emb), dim = 3)
        pep_feat = torch.cat((pep_feat[:, :, :27], pep_esm_emb), dim = 2)

        pep_feat = self.pep_emb(pep_feat)
        rec_feat = self.rec_emb(rec_feat)
        edge_feat = self.edge_emb(edge_feat)

        pep_feat = self.pep_ln(pep_feat)
        rec_feat = self.rec_ln(rec_feat)

        for ga in self.graph_attentions:
            pep_feat = ga(rec_feat, pep_feat, edge_feat)

        feature = self.self_attention(pep_feat, edge_feat)
        feature = self.mlp(feature)
        
        return feature
    


class FeatureExtractionModel(nn.Module):
    def __init__(self, rec_features, pep_features, edge_features, emb_features, esm_emb_features, out_features, attention_heads, dropout, alpha, num_layers):
        super().__init__()
        
        self.esm_embedding = nn.Linear(1280, esm_emb_features, bias=False)
        self.pep_emb = nn.Linear(pep_features + esm_emb_features, emb_features, bias=False)
        self.rec_emb = nn.Linear(rec_features + esm_emb_features, emb_features, bias=False)
        self.edge_emb = nn.Linear(edge_features, emb_features, bias=False)
        
        #xavier_uniform_ all embedding linear layers
        init.xavier_uniform_(self.esm_embedding.weight)
        init.xavier_uniform_(self.pep_emb.weight)
        init.xavier_uniform_(self.rec_emb.weight)
        init.xavier_uniform_(self.edge_emb.weight)

        self.pep_ln = nn.LayerNorm(emb_features)
        self.rec_ln = nn.LayerNorm(emb_features)

        self.graph_attentions = nn.ModuleList([GraphAttention(rec_features + esm_emb_features, pep_features + esm_emb_features, edge_features, emb_features, out_features, dropout, alpha) for _ in range(num_layers)])
        self.self_attention = SelfAttentionPooling(out_features, attention_heads, attention_units=[out_features, int(out_features / 2)], input_dropout=dropout)
        
        self.mlp = nn.Linear(out_features, int(out_features/2))
        #init.xavier_uniform_(self.mlp.weight)
        
    def forward(self, pep_feat, rec_feat, edge_feat):
        rec_esm_feat = rec_feat[:, :, :, 42:]
        pep_esm_feat = pep_feat[:, :, -1280:]
        rec_esm_emb = self.esm_embedding(rec_esm_feat)
        pep_esm_emb = self.esm_embedding(pep_esm_feat)

        rec_feat = torch.cat((rec_feat[:, :, :, :21], rec_esm_emb), dim = 3)
        pep_feat = torch.cat((pep_feat[:, :, :27], pep_esm_emb), dim = 2)

        pep_feat = self.pep_emb(pep_feat)
        rec_feat = self.rec_emb(rec_feat)
        edge_feat = self.edge_emb(edge_feat)

        pep_feat = self.pep_ln(pep_feat)
        rec_feat = self.rec_ln(rec_feat)

        for ga in self.graph_attentions:
            pep_feat = ga(rec_feat, pep_feat, edge_feat)

        feature = self.self_attention(pep_feat, edge_feat)
        feature = self.mlp(feature)
        
        return feature


class MyModel1(nn.Module):
    def __init__(self, rec_features, pep_features, edge_features, emb_features, esm_emb_features, out_features, attention_heads, dropout, alpha, num_layers):
        super().__init__()

        self.feature_extraction_model = FeatureExtractionModel_sep(rec_features, pep_features, edge_features, emb_features, esm_emb_features, out_features, attention_heads, dropout, alpha, num_layers)
        self.final_classifier = nn.Sequential(
            nn.Linear(int(out_features/2),int(out_features/2)),
            nn.Linear(int(out_features/2), 1))
        
        self.init_params = {
            "rec_features": rec_features,
            "pep_features": pep_features,
            "edge_features": edge_features,
            "emb_features": emb_features,
            "esm_emb_features": esm_emb_features,
            "out_features": out_features,
            "attention_heads": attention_heads,
            "dropout": dropout,
            "alpha": alpha,
            "num_layers": num_layers
        }
        

    def forward(self, pep_feat, rec_feat, edge_feat):
        feature = self.feature_extraction_model(pep_feat, rec_feat, edge_feat)
        pred = self.final_classifier(feature).squeeze(1)

        return feature, pred

class MyModel3(nn.Module):
    def __init__(self, rec_features, pep_features, edge_features, emb_features, esm_emb_features, out_features, attention_heads, dropout, alpha, num_layers):
        super().__init__()

        self.feature_extraction_model = FeatureExtractionModel(rec_features, pep_features, edge_features, emb_features, esm_emb_features, out_features, attention_heads, dropout, alpha, num_layers)
        self.final_classifier = nn.Sequential(
            nn.Linear(int(out_features/2),int(out_features/2)),
            nn.Linear(int(out_features/2), 1))
        self.init_params = {
            "rec_features": rec_features,
            "pep_features": pep_features,
            "edge_features": edge_features,
            "emb_features": emb_features,
            "esm_emb_features": esm_emb_features,
            "out_features": out_features,
            "attention_heads": attention_heads,
            "dropout": dropout,
            "alpha": alpha,
            "num_layers": num_layers
        }

    def forward(self, pep_feat, rec_feat, edge_feat):
        feature = self.feature_extraction_model(pep_feat, rec_feat, edge_feat)
        pred = self.final_classifier(feature).squeeze(1)

        return feature, pred

class MyModel4(nn.Module):
    def __init__(self, rec_features, pep_features, edge_features, emb_features, esm_emb_features, out_features, attention_heads, dropout, alpha, num_layers):
        super().__init__()

        self.feature_extraction_model = FeatureExtractionModel(rec_features, pep_features, edge_features, emb_features, esm_emb_features, out_features, attention_heads, dropout, alpha, num_layers)
        self.final_classifier = nn.Linear(int(out_features/2), 1)
        self.init_params = {
            "rec_features": rec_features,
            "pep_features": pep_features,
            "edge_features": edge_features,
            "emb_features": emb_features,
            "esm_emb_features": esm_emb_features,
            "out_features": out_features,
            "attention_heads": attention_heads,
            "dropout": dropout,
            "alpha": alpha,
            "num_layers": num_layers
        }

    def forward(self, pep_feat, rec_feat, edge_feat):
        feature = self.feature_extraction_model(pep_feat, rec_feat, edge_feat)
        pred = self.final_classifier(feature).squeeze(1)
        return feature, pred

"""
class MyModel1_motif(nn.Module):
    def __init__(self, rec_features, pep_features, edge_features, motif_features, emb_features, esm_emb_features, out_features, attention_heads, dropout, alpha, num_layers):
        super().__init__()

        self.feature_extraction_model = FeatureExtractionModel_sep(rec_features, pep_features, edge_features, emb_features, esm_emb_features, out_features, attention_heads, dropout, alpha, num_layers)
        self.final_classifier = nn.Sequential(
            nn.Linear(int(out_features/2) + 6, out_features/2),
            nn.Linear(int(out_features/2), 1))
    
        
        #self.motif_emb = nn.Linear(motif_features, 1, bias=False)

        #self.real_final_classifier = nn.Linear(14, 1)

        self.init_params = {
            "rec_features": rec_features,
            "pep_features": pep_features,
            "edge_features": edge_features,
            "motif_features": motif_features, 
            "emb_features": emb_features,
            "esm_emb_features": esm_emb_features,
            "out_features": out_features,
            "attention_heads": attention_heads,
            "dropout": dropout,
            "alpha": alpha,
            "num_layers": num_layers
        }
        

    def forward(self, pep_feat, rec_feat, edge_feat, motif_feat):
        feature = self.feature_extraction_model(pep_feat, rec_feat, edge_feat)
        #concat feature & motif_feat
        merged_feature = torch.cat((feature, motif_feat), dim = 1)
        #print("merged_feature.shape", merged_feature.shape)

        final_pred = self.final_classifier(merged_feature).squeeze(1)

        return feature, final_pred
    
"""

class MyModel1_motif(nn.Module):
    def __init__(self, rec_features, pep_features, edge_features, motif_features, emb_features, esm_emb_features, out_features, attention_heads, dropout, alpha, num_layers, device='cpu'):
        super().__init__()

        self.feature_extraction_model = FeatureExtractionModel_sep(rec_features, pep_features, edge_features, emb_features, esm_emb_features, out_features, attention_heads, dropout, alpha, num_layers).to(device)
        in_features = int(out_features/2) + 6
        mid_features = int(out_features/2)
    
        self.final_classifier = nn.Sequential(
            nn.Linear( in_features, mid_features), 
            nn.ReLU(),  # Adding activation function
            nn.Linear(mid_features, 1
        )).to(device)
    
        #self.motif_emb = nn.Linear(motif_features, 1, bias=False).to(device)

        #self.real_final_classifier = nn.Linear(14, 1).to(device)

        self.init_params = {
            "rec_features": rec_features,
            "pep_features": pep_features,
            "edge_features": edge_features,
            "motif_features": motif_features, 
            "emb_features": emb_features,
            "esm_emb_features": esm_emb_features,
            "out_features": out_features,
            "attention_heads": attention_heads,
            "dropout": dropout,
            "alpha": alpha,
            "num_layers": num_layers
        }
        

    def forward(self, pep_feat, rec_feat, edge_feat, motif_feat):
        feature = self.feature_extraction_model(pep_feat, rec_feat, edge_feat)
        #concat feature & motif_feat
        merged_feature = torch.cat((feature, motif_feat), dim=1)
        #print("merged_feature.shape", merged_feature.shape)

        final_pred = self.final_classifier(merged_feature).squeeze(1)

        return feature, final_pred
# encoding: utf-8
import os
import torch
from torch import nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
from copy import deepcopy

from layers.layer4gat import GraphAttentionLayer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def reduce_triu_tensor(triu): # remove lower triangle of adj matrix
    triu = triu[:, :-1]
    reduced = triu[:, :triu.shape[1]//2]
    for i in range(reduced.shape[1]):
        # print(triu[triu.shape[0]-i-1, triu.shape[1]-i-1 : triu.shape[1]])
        reduced[:, i, 0:i+1] = triu[:, triu.shape[1]-i-1, triu.shape[2]-i-1 : triu.shape[2]]
    return reduced

class GAT4CGAN(nn.Module):
    def __init__(self, opt, nfeat, nhid, nclass, dropout, alpha, nheads, enc_out_dim, latent_dim):
        """Dense version of GAT."""
        super(GAT4CGAN, self).__init__()
        self.dropout = dropout
        self.opt = opt

        self.attentions = [GraphAttentionLayer(nfeat, nhid, dropout=dropout, alpha=alpha, concat=True) for _ in range(nheads)]
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)

        self.fc_mu = nn.Linear(enc_out_dim, nhid*nheads)
        self.fc_var = nn.Linear(enc_out_dim, nhid*nheads)

        self.mol_emb = nn.Linear(self.opt.max_atoms*opt.num_class, latent_dim)

        self.N = torch.distributions.Normal(0, 1)

        # self.fc = nn.Linear(self.opt.max_atoms*64, self.opt.max_atoms*5)
        self.fc = nn.Linear(self.opt.max_atoms*self.opt.enc_out_dim, self.opt.max_atoms*self.opt.num_class)
        self.fc_reduced = nn.Linear(self.opt.max_atoms*opt.num_class, self.opt.max_atoms)
        self.fc_adj_ohe = [nn.Linear(self.opt.max_atoms, self.opt.max_atoms*5).to(device) for _ in range(opt.reduced_row_size)]
        self.logSoftmax = nn.LogSoftmax(dim=-1)

        self.fc_noise = nn.Linear(100, self.opt.max_atoms*64)

    def encoder(self,  xx, adj): # to encode molecules
        x = torch.cat([att(xx, adj) for att in self.attentions], dim=-1)
        # print(x.shape)
        x = self.fc(x.view(x.shape[0], -1))
        z = F.dropout(x, self.dropout, training=self.training)
        return z


    def decoder(self, z): # decode to get adj matrix 
        # adj is defined as z*z.T
        # to save time, the following opteration imitates the 'broadcasting' in numpy
        z = z.view(z.shape[0], -1)
        z = self.fc_reduced(z).to(device)
        z_1 = torch.zeros(z.shape[0], z.shape[1], z.shape[1]).to(device)
        for j in range(z_1.shape[1]):
            z_1[:, j] = z[:]

        z_1 = z_1.transpose(1, 2)
        z_2 = torch.zeros(z.shape[0], z.shape[1], z.shape[1]).to(device)
        for j in range(z_2.shape[1]):
            z_2[:, j] = z[:]
        z_1 = reduce_triu_tensor(z_1)
        z_2 = reduce_triu_tensor(z_2)
        adj = z_1 * z_2 #element-wise multiply
        z_1 = None
        z_2 = None

        adj_ohe = torch.zeros(adj.shape[0], adj.shape[1], adj.shape[2], 5).to(device)

        for j in range(adj_ohe.shape[1]):
            tmp = self.fc_adj_ohe[j](adj[:, j])
            tmp = tmp.reshape(-1, adj.shape[2], 5)
            adj_ohe[:, j] = tmp[:]

        adj_ohe = self.logSoftmax(adj_ohe)
        return adj_ohe

    def cat_noise_or_z_with_pros(self, z, conditional_pros): #+++
        mol_emb = z.view(z.size(0), -1)
        mol_emb = self.mol_emb(mol_emb) #+++
        pro_mol = torch.cat((conditional_pros, mol_emb), -1) #+++
        return pro_mol


    # def sample4_G_D_noise(self, z): #+++
    #     noise = self.fc_noise(z)
    #     noise = noise.view(z.size(0), self.opt.max_atoms, 64)
    #     return noise



class Discriminator(nn.Module):
    def __init__(self, opt):
        super(Discriminator, self).__init__()

        self.gat4cgan = GAT4CGAN(opt, nfeat=opt.feature_dim, nhid=opt.num_heads, 
            nclass=opt.num_class, dropout=opt.dropout, 
            alpha=opt.alpha, nheads=opt.num_heads, 
            enc_out_dim=opt.enc_out_dim, latent_dim=opt.latent_dim)

        self.pro_emb = nn.Linear(4834, 512) 
        self.mol_emb = nn.Linear(opt.max_atoms*opt.num_class, 512) 
        self.model = nn.Sequential(
            nn.Linear(1024, 512), 

            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 128),
            nn.Dropout(0.4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(128, 32),
            nn.Dropout(0.4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

    def forward(self, mols, conditional_pros):
        pro_emb = self.pro_emb(conditional_pros)

        mol_emb = mols.view(mols.size(0), -1)
        mol_emb = self.mol_emb(mol_emb)
        
        # d_in = torch.cat((mol, pros), -1)
        d_in = torch.cat((pro_emb, mol_emb), -1)
        validity = self.model(d_in)
        return validity


class Generator(nn.Module):
    def __init__(self, opt):
        super(Generator, self).__init__()

        self.gat4cgan = GAT4CGAN(opt, nfeat=opt.feature_dim, nhid=opt.num_heads, 
                    nclass=opt.num_class, dropout=opt.dropout, 
                    alpha=opt.alpha, nheads=opt.num_heads, 
                    enc_out_dim=opt.enc_out_dim, latent_dim=opt.latent_dim)

        def block(in_feat, out_feat, normalize=True):
            layers = [nn.Linear(in_feat, out_feat)]
            if normalize:
                layers.append(nn.BatchNorm1d(out_feat, 0.8))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model0 = nn.Sequential(
            *block(opt.latent_dim + 4834, 128, normalize=False),
            *block(128, 256),
            *block(256, 512),
            *block(512, 1024),
            nn.Linear(1024, opt.max_atoms*opt.feature_dim),
            # nn.Softmax()
            nn.LogSoftmax(dim=-1)
        )

    def forward(self, noise, mat_adj, pros):
        # Concatenate pros and sampled_z(noise)
        gen_input = self.gat4cgan.cat_noise_or_z_with_pros(noise, pros)
        mol = self.model0(gen_input)

        return mol

    # def forward_with_sampling(self, z, pros):
    #     # gen_input = torch.cat((pros, z), -1)
    #     noise = self.gat4cgan.sample4_G_D_noise(z)
    #     gen_input = self.gat4cgan.cat_noise_or_z_with_pros(noise, pros)
    #     mol = self.model0(gen_input)
    #     return mol

    def encoder(self, xx, adj):
        z = self.gat4cgan.encoder(xx, adj)
        return z

    def decoder(self, z):
        adj = self.gat4cgan.decoder(z)
        return adj

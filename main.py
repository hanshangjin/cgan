# encoding: utf-8
import argparse
import numpy as np

from torch.autograd import Variable

import torch.nn as nn
import torch.nn.functional as F
import torch
from tensorboardX import SummaryWriter

from mydb.dataset4FuzCav import ProMolDataset4FuzCav
# from mydb.dataset4FuzCav_adj import ProMolDataset4FuzCav
from layers.model4FuzCav import Generator, Discriminator
import utility.global_var as g
from utility.mol_emb import mol_emb


parser = argparse.ArgumentParser()
parser.add_argument("--n_epochs", type=int, default=20, help="number of epochs of training")
parser.add_argument("--batch_size", type=int, default=64, help="size of the batches")
parser.add_argument("--lr", type=float, default=0.0002, help="adam: learning rate")
parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.999, help="adam: decay hreads to use during batch generation")
parser.add_argument("--latent_dim", type=int, default=100, help="dimensionality of the latent space")
parser.add_argument("--max_atoms", type=int, default=g.Max_atoms)
parser.add_argument("--reduced_row_size", type=int, default=(g.Max_atoms-1)//2)
parser.add_argument("--feature_dim", type=int, default=23)
parser.add_argument("--hidden_dim", type=int, default=8, help="hidden dim")
parser.add_argument("--num_class", type=int, default=23)
parser.add_argument("--dropout", type=float, default=0.5, help="Dropout rate (1 - keep probability).")
parser.add_argument("--alpha", type=float, default=0.2, help="Alpha for leaky_relu.")
parser.add_argument("--num_heads", type=int, default=8, help="Number of head attentions")
parser.add_argument("--enc_out_dim", type=int, default=8*8, help="encoder output dim")
parser.add_argument("--latent_dim_gat", type=int, default=8*8, help="latent dim")
opt = parser.parse_args()
print(opt)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Loss functions
adversarial_loss = nn.MSELoss()
ce_loss = nn.NLLLoss()

# Initialize generator and discriminator
generator = Generator(opt)
discriminator = Discriminator(opt)

generator.to(device)
discriminator.to(device)
adversarial_loss.to(device)
ce_loss.to(device)


myDataset = ProMolDataset4FuzCav('./database/FuzCav_train.db')
dataloader = torch.utils.data.DataLoader(myDataset, batch_size=opt.batch_size, shuffle=True, drop_last=True)

# Optimizers
optimizer_G = torch.optim.Adam(generator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))

FloatTensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if torch.cuda.is_available() else torch.LongTensor


def accuracy(output, labels): #+++
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.mean()
    return correct

def batch_mean_accuracy_node(batch_x_hat, batch_labels): #+++
    acc = 0.0
    for i in range(batch_labels.size(0)):
        acc += accuracy(batch_x_hat[i], batch_labels[i])
    acc = acc / batch_labels.size(0)
    return acc

def batch_mean_accuracy_adj(batch_x_hat, batch_labels): #+++
    acc = 0.0
    for i in range(batch_labels.size(0)):
        acck = 0.0
        for k in range(batch_labels.size(1)):
            acck += accuracy(batch_x_hat[i, k], batch_labels[i, k])
        acc += acck / batch_labels.size(1)
    acc = acc / batch_labels.size(0)
    return acc

#+++++++++compute acc without carbon and single bond+++++++++++++++++++++++
global stop_size_list #stop bond accuracy computation at molecules' real size
stop_size_list = []

def accuracy_without_skip_and_stop_label(output, labels, skip_label, stop_label): #+++
    preds = output.max(1)[1].type_as(labels)
    # correct = preds.eq(labels).double()
    # correct = correct.mean()
    count = 0
    correct = 0.0
    for i in range(labels.size(0)):
        if labels[i].item() == stop_label:
            break
        if labels[i].item() == skip_label:
            continue
        if preds[i].item() == labels[i].item():
            correct += 1.0
        count += 1
    if count != 0:
        correct = correct / count
    return correct, count

def accuracy_without_skip_label_with_stop_size(output, labels, skip_label, stop_size): #+++
    preds = output.max(1)[1].type_as(labels)
    # correct = preds.eq(labels).double()
    # correct = correct.mean()
    count = 0
    correct = 0.0
    for i in range(labels.size(0)):
        if i == stop_size:
            break
        # if labels[i].item() == skip_label[0] or labels[i].item() == skip_label[1]:
        # if labels[i].item() in skip_label:
        if labels[i].item() == skip_label[1]:
            continue
        if preds[i].item() == labels[i].item():
            correct += 1.0
        count += 1
    if count != 0:
        correct = correct / count
    return correct


def batch_mean_accuracy_node_without_carbon(batch_x_hat, batch_labels): #+++
    acc = 0.0
    for i in range(batch_labels.size(0)):
        correct, count = accuracy_without_skip_and_stop_label(batch_x_hat[i], batch_labels[i], 2, 22) #skip_label == 2, stop_label==22
        acc += correct
        stop_size_list.append(count)
    acc = acc / batch_labels.size(0)
    return acc

def batch_mean_accuracy_adj_without_single(batch_x_hat, batch_labels): #+++
    acc = 0.0
    for i in range(batch_labels.size(0)):
        acck = 0.0
        for k in range(batch_labels.size(1)):
            acck += accuracy_without_skip_label_with_stop_size(batch_x_hat[i, k], batch_labels[i, k], [0, 4], stop_size_list[i]) #skip single == 0, unspecified == 4
        acc += acck / batch_labels.size(1)
    acc = acc / batch_labels.size(0)
    return acc
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# ----------
#  Training
# ----------
logdir = 'logs/run33'
tb = SummaryWriter(logdir=logdir, purge_step=1, 
						filename_suffix='-fuzcav',
						log_dir=logdir) #+++

batch_size = opt.batch_size

for epoch in range(opt.n_epochs):
    a=1
    # for i, (pro_id, pro_emb, node_feature, adj_mat, node_atom_label) in enumerate(dataloader):
    for i, (pro_id, pros, mols, adj_mat, node_atom_label, adj_ohe) in enumerate(dataloader):

        # Adversarial ground truths
        valid = Variable(FloatTensor(batch_size, 1).fill_(1.0), requires_grad=False)
        fake = Variable(FloatTensor(batch_size, 1).fill_(0.0), requires_grad=False)

        x_real = Variable(mols.type(FloatTensor))
        x_real_emb = mol_emb(opt, mols, adj_mat)

        # -----------------
        #  Train Generator
        # -----------------
        labels = node_atom_label
        adj_ohe = adj_ohe.argmax(dim=-1)

        optimizer_G.zero_grad()
        noise = Variable(FloatTensor(np.random.normal(0, 1, (batch_size, opt.max_atoms, opt.num_class))))
        x_gen = generator.forward(noise, adj_mat, pros)
        x_gen_hat = x_gen.view(x_gen.size(0), opt.max_atoms, 23)
        batch_gen_node_ce_loss = Variable(FloatTensor(batch_size, 1).fill_(0.0), )
        for j in range(batch_size):
            batch_gen_node_ce_loss[j, 0] = ce_loss(x_gen_hat[j], labels[j])
        g_loss1 = batch_gen_node_ce_loss.mean()

        gen_adj = generator.decoder(x_gen) #+++
        batch_gen_adj_ce_loss = Variable(FloatTensor(batch_size, opt.reduced_row_size, 1).fill_(0.0))
        for j in range(batch_size):
            for k in range(opt.reduced_row_size):
                batch_gen_adj_ce_loss[j, k, 0] = ce_loss(gen_adj[j, k], adj_ohe[j, k]) #+++

        g_loss2 = batch_gen_adj_ce_loss.mean()

        g_loss = (g_loss1 + g_loss2) / 2

        g_loss.backward()

        if hasattr(torch.cuda, 'empty_cache'):
            torch.cuda.empty_cache()


        # acc without carbon and single
        with torch.no_grad():
            node_pred_acc = batch_mean_accuracy_node_without_carbon(x_gen_hat, labels)

        with torch.no_grad():
            adj_pred_acc = batch_mean_accuracy_adj_without_single(gen_adj, adj_ohe)

        # ---------------------
        #  Train Discriminator
        # ---------------------
        optimizer_D.zero_grad()

        # Loss for real
        validity_real = discriminator(x_real_emb, pros)
        d_real_loss = adversarial_loss(validity_real, valid)

        # Loss for fake
        validity_fake = discriminator(x_gen.detach(), pros)
        d_fake_loss = adversarial_loss(validity_fake, fake)

        # Total discriminator loss
        d_loss = (d_real_loss + d_fake_loss) / 2

        # generator.requires_grad = False # When training D, fix the weight of G

        d_loss.backward()
        optimizer_D.step()
        optimizer_G.step()
        if hasattr(torch.cuda, 'empty_cache'):
            torch.cuda.empty_cache()
        # generator.requires_grad = True


        print(
            "[Epoch %d/%d][Batch %d/%d][D loss: %f][G node pred loss: %f][G adj pred loss: %f][G node pred acc: %f][G adj pred acc: %f]"
            % (epoch, opt.n_epochs, i, len(dataloader), d_loss.item(), g_loss1, g_loss2, node_pred_acc, adj_pred_acc)
        )

        num_step = epoch*batch_size + i

        tb.add_scalar('D_loss_step:', d_loss.item(), num_step)
        tb.add_scalar('G_gen_loss_step:', g_loss1.item(), num_step)
        tb.add_scalar('G_gen_adj_loss_step:', g_loss2.item(), num_step)
        tb.add_scalar('G_acc_step:', node_pred_acc, num_step)
        tb.add_scalar('G_acc_adj_step:', adj_pred_acc, num_step)

    tb.add_scalar('D_loss_epoch:', d_loss.item(), epoch)
    tb.add_scalar('G_gen_loss_epoch:', g_loss1.item(), epoch)
    tb.add_scalar('G_gen_adj_loss_epoch:', g_loss2.item(), epoch)
    tb.add_scalar('G_acc_epoch:', node_pred_acc, epoch)
    tb.add_scalar('G_acc_adj_epoch:', adj_pred_acc, epoch)


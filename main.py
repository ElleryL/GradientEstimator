import random
import torch
import numpy as np
from models.vae import VAE
from itertools import chain
import matplotlib.pyplot as plt

import argparse
import os
import time
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import torch
from torch import nn, optim
from torchvision import datasets, transforms
from torchvision.utils import save_image

from torch.utils.tensorboard import SummaryWriter

import time


parser = argparse.ArgumentParser(description='VAE MNIST Example')
parser.add_argument('--batch-size', type=int, default=128, metavar='N',
                    help='input batch size for training (default: 128)')
parser.add_argument('--epochs', type=int, default=10, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='enables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--train_k', type=int, default=1,
                    help='number of iwae samples during training')
parser.add_argument('--test_k', type=int, default=1,
                    help='number of iwae samples during testing')
parser.add_argument('--lr', type=float, default=1e-3,
                    help='learning rate for Adam')
parser.add_argument('--z_dim', type=int, default=20,
                    help='number of latent variables')
parser.add_argument('--dreg', action='store_true',default=0,
                    help='use DReG for inference network gradients')
parser.add_argument('--ablation', action='store_true')
parser.add_argument("--dataset",type=str,default="mnist")
parser.add_argument("--cond_distr",type=str,default="bernoulli")
parser.add_argument("--training",type=int,default=1)
parser.add_argument("--alpha",type=float,default=None, help="the re-weighted value for wake update")
parser.add_argument('--load_checkpoint', type=int, default=0,
                    help='if zero, then we train a new model, if 1 we load a model')


args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()


seed=args.seed
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

device = torch.device("cuda" if args.cuda else "cpu")
print(device)

if not os.path.isdir("./results"):
    os.mkdir("./results")

if not os.path.isdir("./results/" + args.dataset):
    os.mkdir("./results/" + args.dataset)

if args.dreg:
    if not os.path.isdir("./results/" + args.dataset + "/dreg"):
        os.mkdir("./results/" + args.dataset + "/dreg")

    if not os.path.isdir("./results/"  + args.dataset + "/dreg" + "/reconstructedImages%d" % args.seed):
        os.mkdir("./results/"  + args.dataset + "/dreg" + "/reconstructedImages%d" % args.seed)

    if not os.path.isdir("./results/"  + args.dataset + "/dreg" + "/generatedImages%d" % args.seed):
        os.mkdir("./results/"  + args.dataset + "/dreg" + "/generatedImages%d" % args.seed)

    dir_to_save = "./results/" + args.dataset + "/dreg"

else:
    if not os.path.isdir("./results/" + args.dataset + "/iwae"):
        os.mkdir("./results/" + args.dataset + "/iwae")

    if not os.path.isdir("./results/"  + args.dataset + "/iwae" + "/reconstructedImages%d" % args.seed):
        os.mkdir("./results/"  + args.dataset + "/iwae" + "/reconstructedImages%d" % args.seed)

    if not os.path.isdir("./results/"  + args.dataset + "/iwae" + "/generatedImages%d" % args.seed):
        os.mkdir("./results/"  + args.dataset + "/iwae" + "/generatedImages%d" % args.seed)

    dir_to_save = "./results/" + args.dataset + "/iwae"

if not os.path.isdir(dir_to_save + "/" + "checkpoints"):
    os.mkdir(dir_to_save + "/" + "checkpoints")

if not os.path.isdir(dir_to_save + "/log_%d"%args.seed):
    os.mkdir(dir_to_save + "/log_%d"%args.seed)

writer = SummaryWriter(log_dir=dir_to_save + "/log_%d"%args.seed)

if args.dataset == "mnist":
    kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}
    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../data', train=True, download=True,
                       transform=transforms.ToTensor()),
        batch_size=args.batch_size, shuffle=True, **kwargs)
    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../data', train=False, transform=transforms.ToTensor()),
        batch_size=args.batch_size, shuffle=True, **kwargs)

    in_channel = 1
    in_height = in_width = 28
    model = VAE(height=in_height,width=in_width,in_channel=in_channel,
                z_dim=args.z_dim,k=args.train_k).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

def train_iwae(epoch):
  model.train()
  train_loss = 0
  for batch_idx, (data, _) in enumerate(train_loader):
    data = data.to(device)

    optimizer.zero_grad()
    x_prime, z, mu, logvar = model(data)
    loss = model.neg_elbo_iwae(data, x_prime, z, mu, logvar, cond_distr=args.cond_distr)
    #loss.backward()

    grad_dec = torch.autograd.grad(loss, model.dec.parameters(),
                                   retain_graph=True)

    for i, p in enumerate(model.dec.parameters()):
        p.grad = grad_dec[i].clone()

    grad_enc = torch.autograd.grad(loss,
                                   chain(model.enc.parameters(), model.mu.parameters(), model.sigma.parameters()))
    enc_params = chain(model.enc.parameters(), model.mu.parameters(), model.sigma.parameters())

    grad_var = 0  # TODO: compute gradient variance
    for i, p in enumerate(enc_params):
        p.grad = grad_enc[i].clone()
        grad_var += grad_enc[i].clone().detach().pow(2).mean() - grad_enc[i].clone().detach().mean().pow(2)  # squared first moments
    grad_var /= i
    #writer.add_scalar('grad_variance/iwae', grad_var, epoch)
    stats_info["Grads_Var"].append(grad_var)

    train_loss += loss.item()
    optimizer.step()

    if batch_idx % args.log_interval == 0:
      print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
          epoch, batch_idx * len(data), len(train_loader.dataset),
          100. * batch_idx / len(train_loader),
          loss.item()))

  avg_loss = train_loss / (batch_idx + 1)
  stats_info["Train_Loss"].append(avg_loss)
  print('====> Epoch: {} Average loss: {:.4f}'.format(
        epoch, avg_loss))

  return avg_loss


def train_dreg(epoch):
    model.train()
    train_loss = 0
    for batch_idx, (data, _) in enumerate(train_loader):
        data = data.to(device)

        optimizer.zero_grad()
        x_prime, z, mu, logvar = model(data)
        likelihood_loss, infer_loss = model.neg_elbo_dreg(data, x_prime, z, mu, logvar, cond_distr=args.cond_distr)


        grad_dec = torch.autograd.grad(likelihood_loss, model.dec.parameters(),retain_graph=True) #TODO: compute gradient variance

        for i,p in enumerate(model.dec.parameters()):
            p.grad = grad_dec[i].clone()

        grad_enc = torch.autograd.grad(infer_loss, chain(model.enc.parameters(), model.mu.parameters(), model.sigma.parameters()))
        enc_params = chain(model.enc.parameters(), model.mu.parameters(), model.sigma.parameters())

        grad_var = 0
        for i,p in enumerate(enc_params):
            p.grad = grad_enc[i].clone()
            grad_var += grad_enc[i].clone().detach().pow(2).mean() - grad_enc[i].clone().detach().mean().pow(
                2)  # squared first moments
        grad_var /= i
        #writer.add_scalar('grad_variance/dreg', grad_var, epoch)
        stats_info["Grads_Var"].append(grad_var)


        loss = likelihood_loss
        train_loss += loss.item()
        optimizer.step()

        if batch_idx % args.log_interval == 0:
          print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
              epoch, batch_idx * len(data), len(train_loader.dataset),
              100. * batch_idx / len(train_loader),
              loss.item()))

    avg_loss = train_loss / (batch_idx + 1)
    stats_info["Train_Loss"].append(avg_loss)
    print('====> Epoch: {} Average loss: {:.4f}'.format(
            epoch, avg_loss))
    return avg_loss





def test(epoch):
  model.eval()
  test_loss = 0
  with torch.no_grad():
    for i, (data, _) in enumerate(test_loader):
      data = data.to(device)
      x_prime, z, mu, logvar = model(data)
      test_loss += model.neg_elbo_iwae(data, x_prime, z, mu, logvar, cond_distr=args.cond_distr)
      if i == 0:
        n = min(data.size(0), 8)
        comparison = torch.cat([data[:n],
                                x_prime.view(args.batch_size * args.train_k,in_channel,in_height,in_width)[:n]])
        save_image(comparison.cpu(),
                   './results/reconstruction_' + str(epoch) + '.png', nrow=n)

        sample = torch.randn(64, 20).to(device)
        sample = model.decode(sample).cpu()
        save_image(sample.view(64, in_channel,in_width, in_height),
                   './results/sample_' + str(epoch) + '.png')

  test_loss /= (i + 1)
  stats_info["Test_Loss"].append(test_loss)
  print('====> Test set loss: {:.4f}'.format(test_loss))
  return test_loss

if __name__ == "__main__":
    stats_info = {}

    if args.training:

        if args.load_checkpoint:
            print("Loading on dataset {} with likleihood {} and save to {} with seed {}".format(args.dataset, args.cond_distr,
                                                                                    dir_to_save,args.seed))
            checkpoint = torch.load(dir_to_save + "/" + "checkpoints" + "/model%d" % args.seed)
            model.load_state_dict(checkpoint['model_state_dict'])

            stats_info["cur_epoch"] = checkpoint['stats_info']['cur_epoch']
            stats_info["Train_Loss"] = checkpoint['stats_info']['Train_Loss']
            stats_info["Test_Loss"] = checkpoint['stats_info']['Test_Loss']
            stats_info["Grads_Var"] = checkpoint['stats_info']['Grads_Var']

            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        else:
            print("Training on dataset {} with likleihood {} and save to {} with seed {}".format(args.dataset, args.cond_distr,
                                                                                    dir_to_save,args.seed))
            stats_info["Train_Loss"] = []
            stats_info["Test_Loss"] = []
            stats_info["Grads_Var"] = []
            stats_info["cur_epoch"] = 0

        train = train_dreg if args.dreg else train_iwae

        start_time = time.time()
        for epoch in range(stats_info["cur_epoch"] + 1, stats_info["cur_epoch"] + args.epochs + 1):
            train(epoch)
            if epoch % 1 == 0:
                test(epoch)
        stats_info["cur_epoch"] = epoch
        torch.save({
                'stats_info': stats_info,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
        }, dir_to_save + "/" + "checkpoints" + "/model%d" % args.seed)

    else:
        checkpoint = torch.load(dir_to_save + "/" + "checkpoints" + "/model%d" % args.seed)
        model.load_state_dict(checkpoint['model_state_dict'])
    print("Takes {:.4f} seconds".format(time.time() - start_time))
    print("DONE !")

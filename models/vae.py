import math

import torch
from torch import nn
from torch.nn import functional as F

from models.utils import gaussian_log_prob,bernoulli_log_prob

class VAE(nn.Module):

    def __init__(self,height=28,width=28,in_channel=1,num_layers=2,h_dim=400,z_dim=10,k=10):
        super(VAE, self).__init__()
        self.z_dim=z_dim
        self.num_layers=num_layers
        self.h_dim=h_dim
        self.in_channel=in_channel
        self.height=height
        self.width=width

        self.enc = torch.nn.Sequential()

        in_dim = height * width
        for _ in range(num_layers):
            self.enc.add_module("enc_mlp_%d" % (_),nn.Linear(in_dim,h_dim))
            self.enc.add_module("enc_leaky_relu%d" % (_), nn.ReLU())
            in_dim = h_dim
        self.mu = nn.Linear(h_dim, z_dim) # mu
        self.sigma = nn.Linear(h_dim, z_dim) # sigma
        self.k=k


        self.d1 = nn.Linear(z_dim, h_dim)
        self.dec = torch.nn.Sequential()
        in_dim = h_dim

        self.dec.add_module("dec_mlp", self.d1)
        self.dec.add_module("dec_leaky_relu", nn.LeakyReLU(0.2))
        for _ in range(num_layers-1):
            self.dec.add_module("dec_mlp_%d" % (_),nn.Linear(in_dim,h_dim))
            self.dec.add_module("dec_leaky_relu%d" % (_), nn.ReLU())
            in_dim = h_dim
        self.dec.add_module("dec_mlp_%d" % (_ + 1),nn.Linear(h_dim,height * width))

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.distributions.Normal(torch.zeros_like(mu),torch.ones_like(std)).sample((self.k,)) # (k, batch_size, z_dim)
        return mu + eps * std

    def encode(self, x):
        x = x.reshape(-1, self.width * self.height)
        h = self.enc(x)
        h = h.view(x.size(0), -1)
        mu, logvar = self.mu(h), self.sigma(h)
        z = self.reparameterize(mu, logvar)
        return z,mu,logvar

    def decode(self, z):
        recon_x = nn.Sigmoid()(self.dec(z))
        return recon_x

    def forward(self, x):
        z, mu, logvar = self.encode(x)
        x_prime = self.decode(z)
        return x_prime, z, mu, logvar

    def neg_elbo_iwae_(self, x, x_prime, z, mu, logvar, cond_distr="bernoulli"):
        batch_size = x.shape[0]
        x_prime = x_prime.reshape(-1, batch_size, self.in_channel,self.width,self.height) # (K,batch_size, channel, width, height)
        log_prior_z = gaussian_log_prob(torch.zeros_like(z),torch.ones_like(z),z).sum(-1)  # (K,batch_size)
        std = torch.exp(0.5 * logvar)
        log_q_z_x = gaussian_log_prob(mu,std,z).sum(-1) #(K,batch_size)

        if cond_distr == "bernoulli":
            log_p_x_z =  bernoulli_log_prob(x_prime,x).sum(-1).sum(-1).sum(-1)
            #log_p_x_z = torch.distributions.Bernoulli(probs=x_prime).log_prob(x).sum(-1).sum(-1).sum(-1)

        log_w = log_p_x_z + log_prior_z - log_q_z_x
        loss = (torch.logsumexp(log_w,dim=0) - math.log(self.k)).mean(0)
        return -loss

    def neg_elbo_iwae(self, x, x_prime, z, mu, logvar, cond_distr="bernoulli"):
        batch_size = x.shape[0]
        x_prime = x_prime.reshape(-1, batch_size, self.in_channel,self.width,self.height) # (K,batch_size, channel, width, height)
        log_prior_z = gaussian_log_prob(torch.zeros_like(z),torch.ones_like(z),z).sum(-1)  # (K,batch_size)
        std = torch.exp(0.5 * logvar)
        log_q_z_x = gaussian_log_prob(mu,std,z).sum(-1) #(K,batch_size)

        if cond_distr == "bernoulli":
            log_p_x_z =  bernoulli_log_prob(x_prime,x).sum(-1).sum(-1).sum(-1)
            #log_p_x_z = torch.distributions.Bernoulli(probs=x_prime).log_prob(x).sum(-1).sum(-1).sum(-1)

        log_w = log_p_x_z + log_prior_z - log_q_z_x
        normalized_weights = (log_w - torch.logsumexp(log_w,0)).exp().detach()

        loss = (normalized_weights * log_w).sum(0).mean()
        return -loss

    def neg_elbo_dreg(self,x,x_prime,z,mu,logvar,cond_distr="bernoulli",alpha=None):
        '''

        :param x: true_image
        :param x_prime: reconstructed_image
        :param z: latent samples
        :param mu: latent distribution mean
        :param logvar: latent distribution variance
        :param cond_distr:
        :param alpha is None indicates no RWS
        :return: three loss terms; neg_log_likelihood, log_prior, log_posterior
        '''
        batch_size = x.shape[0]
        x_prime = x_prime.reshape(-1, batch_size, self.in_channel,self.width,self.height) # (K,batch_size, channel, width, height)
        log_prior_z = gaussian_log_prob(torch.zeros_like(z),torch.ones_like(z),z).sum(-1)  # (K,batch_size)
        std = torch.exp(0.5 * logvar)

        # since z is just a sample; the gradients stops here
        log_q_z_x = gaussian_log_prob(mu,std,z).sum(-1) # (K,batch_size)
        stop_grad_log_q_z_x = gaussian_log_prob(mu.detach(),std.detach(),z).sum(-1)

        if cond_distr == "bernoulli":
            log_p_x_z = bernoulli_log_prob(x_prime, x).sum(-1).sum(-1).sum(-1)

        log_w = log_p_x_z + log_prior_z - log_q_z_x # (K,batch_size)
        # compute gradients w.r.t. decoder parameters
        likelihood_loss = - (torch.logsumexp(log_w,dim=0) - math.log(self.k)).mean(0)

        # compute gradients w.r.t. encoder parameters
        stop_grad_log_w = log_p_x_z + log_prior_z - stop_grad_log_q_z_x
        normalized_weights = (stop_grad_log_w - torch.logsumexp(stop_grad_log_w,dim=0)).detach().exp() # weights here aren't backpropogated
        if alpha:
            infer_loss = (1 - 2 * alpha) * (normalized_weights.pow(2) * stop_grad_log_w).sum(0) + (alpha * normalized_weights * stop_grad_log_w).sum(0)
            infer_loss = - infer_loss.mean()
        else:
            infer_loss = - (normalized_weights.pow(2) * stop_grad_log_w).sum(0).mean()

        return likelihood_loss, infer_loss





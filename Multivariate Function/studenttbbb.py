import torch
from torch import nn
import numpy as np

class T_MeanFieldLayer(nn.Module):
    """Represents a mean-field Student's t-distribution over each layer of the network."""

    def __init__(self, input_dim, output_dim, df, prior_var=1):
        super(T_MeanFieldLayer, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.df = df
        self.prior_scale = np.sqrt(((df-2)/df * prior_var))
        
        # Prior parameters p(theta)
        self.w_loc_p = torch.zeros(input_dim, output_dim)
        self.w_log_scale_p = torch.ones(input_dim, output_dim) * torch.log(torch.tensor(self.prior_scale))

        self.b_loc_p = torch.zeros(output_dim)
        self.b_log_scale_p = torch.ones(output_dim) * torch.log(torch.tensor(self.prior_scale))


        # Variational parameters q(theta)
        self.w_loc_q = nn.Parameter(torch.zeros(input_dim, output_dim), requires_grad=True)
        self.w_log_scale_q = nn.Parameter(
            torch.ones(input_dim, output_dim) * torch.log(torch.tensor(self.prior_scale)), requires_grad=True
        )  
        self.b_loc_q = nn.Parameter(torch.zeros(output_dim), requires_grad=True)
        self.b_log_scale_q = nn.Parameter(
            torch.ones(output_dim) * torch.log(torch.tensor(self.prior_scale)), requires_grad=True
        )

    # the priors do not change so could be stored as attributes, but
    # it feels cleaner to access them in the same way as the posteriors
    def p_w(self):
        """weight prior distribution"""
        return torch.distributions.StudentT(self.df, self.w_loc_p, self.w_log_scale_p.exp())

    def p_b(self):
        """bias prior distribution"""
        return torch.distributions.StudentT(self.df, self.b_loc_p, self.b_log_scale_p.exp())

    def q_w(self):
        """variational weight posterior"""
        return torch.distributions.StudentT(self.df, self.w_loc_q, torch.exp(self.w_log_scale_q))

    def q_b(self):
        """variational bias posterior"""
        return torch.distributions.StudentT(self.df, self.b_loc_q, torch.exp(self.b_log_scale_q))
    
    # def kl(self, num_samples = 1):
    #     weight_samples = self.q_w().rsample((num_samples,))  # rsample allows for gradient propagation
    #     bias_samples = self.q_b().rsample((num_samples,))

    #     log_q_w = self.q_w().log_prob(weight_samples)  
    #     log_p_w = self.p_w().log_prob(weight_samples)

    #     log_q_b = self.q_b().log_prob(bias_samples)
    #     log_p_b = self.p_b().log_prob(bias_samples)

    #     weight_kl = (log_q_w - log_p_w).mean()  # Average over samples
    #     bias_kl = (log_q_b - log_p_b).mean()
    #     return weight_kl + bias_kl

    def kl(self, param_sample):
        weights = param_sample[0]
        biases = param_sample[1]
        weight_kl = self.q_w().log_prob(weights) - self.p_w().log_prob(weights)
        bias_kl = self.q_b().log_prob(biases) - self.p_b().log_prob(biases)
        return weight_kl.sum(dim=[1,2]).mean(0) + bias_kl.sum(dim=[1,2]).mean(0)

    

    def forward(self, x):
        """Propagates x through this layer by sampling weights from the posterior"""
        assert (len(x.shape) == 3), "x should be shape (num_samples, batch_size, input_dim)."
        assert x.shape[-1] == self.input_dim

        num_samples = x.shape[0]
        # rsample carries out reparameterisation trick for us   
        weights = self.q_w().rsample((num_samples,))  # (num_samples, input_dim, output_dim).

        biases = self.q_b().rsample((num_samples,)).unsqueeze(1)  # (num_samples, batch_size, output_dim)

        return x @ weights + biases, [weights, biases] # (num_samples, batch_size, output_dim).




class T_MeanFieldBNN(nn.Module):
    """Mean-field variational inference BNN."""

    def __init__(
        self,
        input_dim,
        hidden_dims,
        output_dim,
        df,
        activation = nn.ELU(),
        noise_std = 1.0,
        prior_var = 1
    ):
        super(T_MeanFieldBNN, self).__init__()
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.output_dim = output_dim
        self.activation = activation
        self.log_noise_var = torch.log(torch.tensor(noise_std**2))
        
        self.network = nn.ModuleList()
        for i in range(len(hidden_dims) + 1):
            if i == 0:
                self.network.append(T_MeanFieldLayer(self.input_dim, self.hidden_dims[i], df=df, prior_var = prior_var))
                self.network.append(self.activation)
            elif i == len(hidden_dims):
                self.network.append(
                    T_MeanFieldLayer(self.hidden_dims[i - 1], self.output_dim, df=df, prior_var = prior_var)
                )
            else:
                self.network.append(
                    T_MeanFieldLayer(self.hidden_dims[i - 1], self.hidden_dims[i], df=df, prior_var = prior_var)
                )
                self.network.append(self.activation) 

    def forward(self, x, num_samples=1):
        """Propagate the inputs through the network using num_samples weights.

        Args:
            x (torch.tensor): Inputs to the network.
            num_samples (int, optional): Number of samples to use. Defaults to 1.
        """
        assert len(x.shape) == 2, "x.shape must be (batch_size, input_dim)."
        param_samples = []
        # Expand dimensions of x to (num_samples, batch_size, input_dim).
        x = torch.unsqueeze(x, 0).repeat(num_samples, 1, 1)
        # Propagate x through network

        for layer in self.network:
            if isinstance(layer, T_MeanFieldLayer):
                x, param_sample = layer(x)
                param_samples.append(param_sample)
            else:
                x = layer(x)

        assert len(x.shape) == 3, "x.shape must be (num_samples, batch_size, output_dim)"
        assert x.shape[-1] == self.output_dim

        return x, param_samples

    def ll(self, y_obs, y_pred, num_samples=1):
        """Computes the log likelihood of the outputs of self.forward(x)"""
        l = torch.distributions.normal.Normal(y_pred, torch.sqrt(torch.exp(self.log_noise_var)))
        
        # take mean over num_samples dim, sum over batch_size dim
        # note that after taking mean, batch_size becomes dim 0
        return l.log_prob(y_obs.unsqueeze(0).repeat(num_samples, 1, 1)).mean(0).sum(0).squeeze()

    def kl(self,param_samples):
        """Computes the KL divergence between the approximate posterior and the prior for the network."""
        return sum([layer.kl(param_samples[i//2]) for i, layer in enumerate(self.network) if isinstance(layer, T_MeanFieldLayer)])

    def loss(self, x, y, num_samples=1):
        """Computes the ELBO and returns its negative"""

        y_pred, param_samples = self.forward(x, num_samples=num_samples)
        
        exp_ll = self.ll(y, y_pred, num_samples=num_samples)
        kl = self.kl(param_samples)

        return kl - exp_ll, exp_ll, kl
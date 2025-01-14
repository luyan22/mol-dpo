from equivariant_diffusion import utils
import numpy as np
import math
import torch
from egnn import models
from torch.nn import functional as F
from equivariant_diffusion import utils as diffusion_utils
from torch_scatter import scatter_mean
import os
import qm9.visualizer as vis
import random

# Defining some useful util functions.
def expm1(x: torch.Tensor) -> torch.Tensor:
    return torch.expm1(x)


def softplus(x: torch.Tensor) -> torch.Tensor:
    return F.softplus(x)


def sum_except_batch(x):
    return x.view(x.size(0), -1).sum(-1)


def clip_noise_schedule(alphas2, clip_value=0.001):
    """
    For a noise schedule given by alpha^2, this clips alpha_t / alpha_t-1. This may help improve stability during
    sampling.
    """
    alphas2 = np.concatenate([np.ones(1), alphas2], axis=0)

    alphas_step = (alphas2[1:] / alphas2[:-1])

    alphas_step = np.clip(alphas_step, a_min=clip_value, a_max=1.)
    alphas2 = np.cumprod(alphas_step, axis=0)

    return alphas2


def polynomial_schedule(timesteps: int, s=1e-4, power=3.):
    """
    A noise schedule based on a simple polynomial equation: 1 - x^power.
    """
    steps = timesteps + 1
    x = np.linspace(0, steps, steps)
    alphas2 = (1 - np.power(x / steps, power))**2

    alphas2 = clip_noise_schedule(alphas2, clip_value=0.001)

    precision = 1 - 2 * s

    alphas2 = precision * alphas2 + s

    return alphas2


def cosine_beta_schedule(timesteps, s=0.008, raise_to_power: float = 1):
    """
    cosine schedule
    as proposed in https://openreview.net/forum?id=-NEXDKk8gZ
    """
    steps = timesteps + 2
    x = np.linspace(0, steps, steps)
    alphas_cumprod = np.cos(((x / steps) + s) / (1 + s) * np.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    betas = np.clip(betas, a_min=0, a_max=0.999)
    alphas = 1. - betas
    alphas_cumprod = np.cumprod(alphas, axis=0)

    if raise_to_power != 1:
        alphas_cumprod = np.power(alphas_cumprod, raise_to_power)

    return alphas_cumprod


def gaussian_entropy(mu, sigma):
    # In case sigma needed to be broadcast (which is very likely in this code).
    zeros = torch.zeros_like(mu)
    return sum_except_batch(
        zeros + 0.5 * torch.log(2 * np.pi * sigma**2) + 0.5
    )


def gaussian_KL(q_mu, q_sigma, p_mu, p_sigma, node_mask):
    """Computes the KL distance between two normal distributions.

        Args:
            q_mu: Mean of distribution q.
            q_sigma: Standard deviation of distribution q.
            p_mu: Mean of distribution p.
            p_sigma: Standard deviation of distribution p.
        Returns:
            The KL distance, summed over all dimensions except the batch dim.
        """
    return sum_except_batch(
            (
                torch.log(p_sigma / q_sigma)
                + 0.5 * (q_sigma**2 + (q_mu - p_mu)**2) / (p_sigma**2)
                - 0.5
            ) * node_mask
        )


def gaussian_KL_for_dimension(q_mu, q_sigma, p_mu, p_sigma, d):
    """Computes the KL distance between two normal distributions.

        Args:
            q_mu: Mean of distribution q.
            q_sigma: Standard deviation of distribution q.
            p_mu: Mean of distribution p.
            p_sigma: Standard deviation of distribution p.
        Returns:
            The KL distance, summed over all dimensions except the batch dim.
        """
    mu_norm2 = sum_except_batch((q_mu - p_mu)**2)
    assert len(q_sigma.size()) == 1
    assert len(p_sigma.size()) == 1
    return d * torch.log(p_sigma / q_sigma) + 0.5 * (d * q_sigma**2 + mu_norm2) / (p_sigma**2) - 0.5 * d


class PositiveLinear(torch.nn.Module):
    """Linear layer with weights forced to be positive."""

    def __init__(self, in_features: int, out_features: int, bias: bool = True,
                 weight_init_offset: int = -2):
        super(PositiveLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = torch.nn.Parameter(
            torch.empty((out_features, in_features)))
        if bias:
            self.bias = torch.nn.Parameter(torch.empty(out_features))
        else:
            self.register_parameter('bias', None)
        self.weight_init_offset = weight_init_offset
        self.reset_parameters()

    def reset_parameters(self) -> None:
        torch.nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

        with torch.no_grad():
            self.weight.add_(self.weight_init_offset)

        if self.bias is not None:
            fan_in, _ = torch.nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            torch.nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, input):
        positive_weight = softplus(self.weight)
        return F.linear(input, positive_weight, self.bias)


class SinusoidalPosEmb(torch.nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        x = x.squeeze() * 1000
        assert len(x.shape) == 1
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb
    

class PredefinedNoiseSchedule(torch.nn.Module):
    """
    Predefined noise schedule. Essentially creates a lookup array for predefined (non-learned) noise schedules.
    """
    def __init__(self, noise_schedule, timesteps, precision):
        super(PredefinedNoiseSchedule, self).__init__()
        self.timesteps = timesteps

        if noise_schedule == 'cosine':
            alphas2 = cosine_beta_schedule(timesteps)
        elif 'polynomial' in noise_schedule:
            splits = noise_schedule.split('_')
            assert len(splits) == 2
            power = float(splits[1])
            alphas2 = polynomial_schedule(timesteps, s=precision, power=power)
        else:
            raise ValueError(noise_schedule)

        print('alphas2', alphas2)

        sigmas2 = 1 - alphas2

        log_alphas2 = np.log(alphas2)
        log_sigmas2 = np.log(sigmas2)

        log_alphas2_to_sigmas2 = log_alphas2 - log_sigmas2

        print('gamma', -log_alphas2_to_sigmas2)

        self.gamma = torch.nn.Parameter(
            torch.from_numpy(-log_alphas2_to_sigmas2).float(),
            requires_grad=False)

    def forward(self, t):
        t_int = torch.round(t * self.timesteps).long()
        return self.gamma[t_int]


class GammaNetwork(torch.nn.Module):
    """The gamma network models a monotonic increasing function. Construction as in the VDM paper."""
    def __init__(self):
        super().__init__()

        self.l1 = PositiveLinear(1, 1)
        self.l2 = PositiveLinear(1, 1024)
        self.l3 = PositiveLinear(1024, 1)

        self.gamma_0 = torch.nn.Parameter(torch.tensor([-5.]))
        self.gamma_1 = torch.nn.Parameter(torch.tensor([10.]))
        self.show_schedule()

    def show_schedule(self, num_steps=50):
        t = torch.linspace(0, 1, num_steps).view(num_steps, 1)
        gamma = self.forward(t)
        print('Gamma schedule:')
        print(gamma.detach().cpu().numpy().reshape(num_steps))

    def gamma_tilde(self, t):
        l1_t = self.l1(t)
        return l1_t + self.l3(torch.sigmoid(self.l2(l1_t)))

    def forward(self, t):
        zeros, ones = torch.zeros_like(t), torch.ones_like(t)
        # Not super efficient.
        gamma_tilde_0 = self.gamma_tilde(zeros)
        gamma_tilde_1 = self.gamma_tilde(ones)
        gamma_tilde_t = self.gamma_tilde(t)

        # Normalize to [0, 1]
        normalized_gamma = (gamma_tilde_t - gamma_tilde_0) / (
                gamma_tilde_1 - gamma_tilde_0)

        # Rescale to [gamma_0, gamma_1]
        gamma = self.gamma_0 + (self.gamma_1 - self.gamma_0) * normalized_gamma

        return gamma


def cdf_standard_gaussian(x):
    return 0.5 * (1. + torch.erf(x / math.sqrt(2)))


class EnVariationalDiffusion(torch.nn.Module):
    """
    The E(n) Diffusion Module.
    """
    def __init__(
            self,
            dynamics: models.EGNN_dynamics_QM9, in_node_nf: int, n_dims: int,
            timesteps: int = 1000, parametrization='eps', noise_schedule='learned',
            noise_precision=1e-4, loss_type='vlb', norm_values=(1., 1., 1.),
            norm_biases=(None, 0., 0.), include_charges=True, uni_diffusion=False, timesteps2: int = 1000, pre_training=False,
            property_pred=False, prediction_threshold_t=10, target_property=None, use_prop_pred=1, freeze_gradient=False, unnormal_time_step=False, only_noisy_node=False, half_noisy_node=False, sep_noisy_node=False,
            relay_sampling=0, second_dynamics=None, sampling_threshold_t=10, atom_type_pred=True,
            condGenConfig=None):
        super().__init__()
        self.cond_gen_loss = None
        self.condGenConfig = None # Load before eval conditional generation, done in eval_conditional_qm9(task edm)
        
        self.property_pred = property_pred
        self.prediction_threshold_t = prediction_threshold_t
        self.target_property = target_property
        self.use_prop_pred = use_prop_pred
        
        # add relay sampling and second_phi
        self.relay_sampling = relay_sampling
        self.second_dynamics = second_dynamics
        self.sampling_threshold_t = sampling_threshold_t
        
        assert loss_type in {'vlb', 'l2'}
        self.loss_type = loss_type
        self.include_charges = include_charges
        if noise_schedule == 'learned':
            assert loss_type == 'vlb', 'A noise schedule can only be learned' \
                                       ' with a vlb objective.'

        # Only supported parametrization.
        assert parametrization == 'eps'

        if noise_schedule == 'learned':
            self.gamma = GammaNetwork()
        else:
            self.gamma = PredefinedNoiseSchedule(noise_schedule, timesteps=timesteps,
                                                 precision=noise_precision)

        # The network that will predict the denoising.
        self.dynamics = dynamics

        self.in_node_nf = in_node_nf
        self.n_dims = n_dims
        self.num_classes = self.in_node_nf - self.include_charges

        self.T = timesteps
        self.parametrization = parametrization
        
        self.uni_diffusion = uni_diffusion
        
        if uni_diffusion:
            # self.uni_diffusion = uni_diffusion
            self.T2 = timesteps2
            if noise_schedule == 'learned':
                self.gamma2 = GammaNetwork()
            else:
                self.gamma2 = PredefinedNoiseSchedule(noise_schedule, timesteps=timesteps2,
                                                    precision=noise_precision)
        

        self.norm_values = norm_values
        self.norm_biases = norm_biases
        self.register_buffer('buffer', torch.zeros(1))
        
        self.pre_training = pre_training
        if self.pre_training:
            self.mask_indicator = False
        else:
            self.mask_indicator = None

        if noise_schedule != 'learned':
            self.check_issues_norm_values()
            
        self.freeze_gradient = freeze_gradient
        
        self.unnormal_time_step = unnormal_time_step
        
        self.only_noisy_node = only_noisy_node
        self.half_noisy_node = half_noisy_node
        
        self.sep_noisy_node = sep_noisy_node
        
        self.atom_type_pred = atom_type_pred
        # self.gamma_lst = np.load('/mnt/nfs-ssd/data/fengshikun/e3_diffusion_for_molecules/gamma.npy')
        # self.gamma_lst = np.load("/home/AI4Science/luy2402/e3_diffusion_for_molecules/data/gamma_luyan/gamma.npy")

    def check_issues_norm_values(self, num_stdevs=8):
        zeros = torch.zeros((1, 1))
        gamma_0 = self.gamma(zeros)
        sigma_0 = self.sigma(gamma_0, target_tensor=zeros).item()

        # Checked if 1 / norm_value is still larger than 10 * standard
        # deviation.
        max_norm_value = max(self.norm_values[1], self.norm_values[2])

        if sigma_0 * num_stdevs > 1. / max_norm_value:
            raise ValueError(
                f'Value for normalization value {max_norm_value} probably too '
                f'large with sigma_0 {sigma_0:.5f} and '
                f'1 / norm_value = {1. / max_norm_value}')

    def phi(self, x, t, node_mask, edge_mask, context, t2=None, mask_y=None, no_noise_xh=None):
        # noise predict network
        # EGNN_dynamics_QM9 for diffusion
        # print(f"x.shape: {x.shape if type(x)==torch.Tensor else type(x)}, node_mask.shape: {node_mask.shape if type(node_mask)==torch.Tensor else type(node_mask)}, edge_mask.shape: {edge_mask.shape if type(edge_mask)==torch.Tensor else type(edge_mask)}, context.shape: {context.shape if type(context)==torch.Tensor else type(context)}, t2: {t2}, mask_y: {mask_y}")
        if self.relay_sampling == 0:
            net_out = self.dynamics._forward(t, x, node_mask, edge_mask, context, t2=t2, mask_y=mask_y, no_noise_xh=no_noise_xh)
            # print("pred in phi: ", net_out[1])
        else:
            if t > self.sampling_threshold_t:
                net_out = self.dynamics._forward(t, x, node_mask, edge_mask, context, t2=t2, mask_y=mask_y, no_noise_xh=no_noise_xh)
            else:
                print("relay_sampling t: ", t)
                assert isinstance(self.second_dynamics, models.EGNN_dynamics_QM9)
                net_out = self.second_dynamics._forward(t, x, node_mask, edge_mask, context, t2=t2, mask_y=mask_y, no_noise_xh=no_noise_xh)
        return net_out

    def inflate_batch_array(self, array, target):
        """
        Inflates the batch array (array) with only a single axis (i.e. shape = (batch_size,), or possibly more empty
        axes (i.e. shape (batch_size, 1, ..., 1)) to match the target shape.
        """
        target_shape = (array.size(0),) + (1,) * (len(target.size()) - 1)
        return array.view(target_shape)

    def sigma(self, gamma, target_tensor):
        """Computes sigma given gamma."""
        # print("gamma[0]: ", gamma[0])
        # print("target_tensor.shape, target_tensor[0]: ", target_tensor.shape, target_tensor[0])
        return self.inflate_batch_array(torch.sqrt(torch.sigmoid(gamma)), target_tensor)

    def alpha(self, gamma, target_tensor):
        """Computes alpha given gamma."""
        return self.inflate_batch_array(torch.sqrt(torch.sigmoid(-gamma)), target_tensor)

    def SNR(self, gamma):
        """Computes signal to noise ratio (alpha^2/sigma^2) given gamma."""
        return torch.exp(-gamma)

    def subspace_dimensionality(self, node_mask):
        """Compute the dimensionality on translation-invariant linear subspace where distributions on x are defined."""
        number_of_nodes = torch.sum(node_mask.squeeze(2), dim=1)
        return (number_of_nodes - 1) * self.n_dims

    def normalize(self, x, h, node_mask):
        x = x / self.norm_values[0]
        delta_log_px = -self.subspace_dimensionality(node_mask) * np.log(self.norm_values[0])

        # Casting to float in case h still has long or int type.
        h_cat = (h['categorical'].float() - self.norm_biases[1]) / self.norm_values[1] * node_mask
        # print("before norm h_cat", h['categorical'][0])
        # print("norm_bias", self.norm_biases[1])
        # print("norm_values", self.norm_values[1])
        # print("node_mask", node_mask[0])
        # print("after norm h_cat", h_cat[0])
        h_int = (h['integer'].float() - self.norm_biases[2]) / self.norm_values[2]

        if self.include_charges:
            h_int = h_int * node_mask

        # Create new h dictionary.
        h = {'categorical': h_cat, 'integer': h_int}

        return x, h, delta_log_px

    def unnormalize(self, x, h_cat, h_int, node_mask):
        x = x * self.norm_values[0]
        h_cat = h_cat * self.norm_values[1] + self.norm_biases[1]
        h_cat = h_cat * node_mask
        h_int = h_int * self.norm_values[2] + self.norm_biases[2]

        if self.include_charges:
            h_int = h_int * node_mask

        return x, h_cat, h_int

    def unnormalize_z(self, z, node_mask):
        # Parse from z
        x, h_cat = z[:, :, 0:self.n_dims], z[:, :, self.n_dims:self.n_dims+self.num_classes]
        h_int = z[:, :, self.n_dims+self.num_classes:self.n_dims+self.num_classes+1]
        assert h_int.size(2) == self.include_charges

        # Unnormalize
        # print("x_shape: ", x.shape)
        # print("h_cat_shape: ", h_cat.shape)
        # print("h_int_shape: ", h_int.shape)
        x, h_cat, h_int = self.unnormalize(x, h_cat, h_int, node_mask)
        if self.dynamics.mode == "PAT" or self.atom_type_pred:
            h_fix = torch.ones_like(torch.cat([h_cat, h_int], dim=2)) 
            return torch.cat([x, h_fix], dim=2)
        output = torch.cat([x, h_cat, h_int], dim=2)
        return output

    def sigma_and_alpha_t_given_s(self, gamma_t: torch.Tensor, gamma_s: torch.Tensor, target_tensor: torch.Tensor):
        """
        Computes sigma t given s, using gamma_t and gamma_s. Used during sampling.

        These are defined as:
            alpha t given s = alpha t / alpha s,
            sigma t given s = sqrt(1 - (alpha t given s) ^2 ).
        """
        sigma2_t_given_s = self.inflate_batch_array(
            -expm1(softplus(gamma_s) - softplus(gamma_t)), target_tensor
        )

        # alpha_t_given_s = alpha_t / alpha_s
        log_alpha2_t = F.logsigmoid(-gamma_t)
        log_alpha2_s = F.logsigmoid(-gamma_s)
        log_alpha2_t_given_s = log_alpha2_t - log_alpha2_s

        alpha_t_given_s = torch.exp(0.5 * log_alpha2_t_given_s)
        alpha_t_given_s = self.inflate_batch_array(
            alpha_t_given_s, target_tensor)

        sigma_t_given_s = torch.sqrt(sigma2_t_given_s)

        return sigma2_t_given_s, sigma_t_given_s, alpha_t_given_s

    def kl_prior(self, xh, node_mask):
        """Computes the KL between q(zT | x) and the prior p(zT) = Normal(0, 1).

        This is essentially a lot of work for something that is in practice negligible in the loss. However, you
        compute it so that you see it when you've made a mistake in your noise schedule.
        """
        # Compute the last alpha value, alpha_T.
        ones = torch.ones((xh.size(0), 1), device=xh.device)
        gamma_T = self.gamma(ones)
        alpha_T = self.alpha(gamma_T, xh)

        # Compute means.
        mu_T = alpha_T * xh
        mu_T_x, mu_T_h = mu_T[:, :, :self.n_dims], mu_T[:, :, self.n_dims:]

        # Compute standard deviations (only batch axis for x-part, inflated for h-part).
        sigma_T_x = self.sigma(gamma_T, mu_T_x).squeeze()  # Remove inflate, only keep batch dimension for x-part.
        sigma_T_h = self.sigma(gamma_T, mu_T_h)

        # Compute KL for h-part.
        zeros, ones = torch.zeros_like(mu_T_h), torch.ones_like(sigma_T_h)
        kl_distance_h = gaussian_KL(mu_T_h, sigma_T_h, zeros, ones, node_mask)

        # Compute KL for x-part.
        zeros, ones = torch.zeros_like(mu_T_x), torch.ones_like(sigma_T_x)
        subspace_d = self.subspace_dimensionality(node_mask)
        kl_distance_x = gaussian_KL_for_dimension(mu_T_x, sigma_T_x, zeros, ones, d=subspace_d)

        if self.dynamics.mode == "PAT" or self.atom_type_pred:
            kl_distance_h = 0.0
        return kl_distance_x + kl_distance_h

    def compute_x_pred(self, net_out, zt, gamma_t):
        # print(f"net_out.shape: {net_out.shape}, zt.shape: {zt.shape}, gamma_t.shape: {gamma_t.shape}")
        """Commputes x_pred, i.e. the most likely prediction of x."""
        if self.parametrization == 'x':
            x_pred = net_out
        elif self.parametrization == 'eps':
            # if self.dynamics.mode == "PAT":
            #     net_out_new = net_out[:, :, :self.n_dims]
            # else:
            net_out_new = net_out
            sigma_t = self.sigma(gamma_t, target_tensor=net_out_new)
            alpha_t = self.alpha(gamma_t, target_tensor=net_out_new)
            eps_t = net_out_new
            x_pred = 1. / alpha_t * (zt - sigma_t * eps_t)
        else:
            raise ValueError(self.parametrization)

        return x_pred

    def compute_error(self, net_out, gamma_t, eps):
        """Computes error, i.e. the most likely prediction of x."""
        eps_t = net_out
        if self.dynamics.mode == "PAT" or self.atom_type_pred: # get x_only for PAT loss
            eps_t = eps_t[:, :, :self.n_dims]
            eps = eps[:, :, :self.n_dims]
     
        if self.training and self.loss_type == 'l2':
            denom = (self.n_dims + self.in_node_nf) * eps_t.shape[1]
            if self.dynamics.mode == "PAT" or self.atom_type_pred:
                denom = (self.n_dims) * eps_t.shape[1]
            error = sum_except_batch((eps - eps_t) ** 2) / denom
        else:
            error = sum_except_batch((eps - eps_t) ** 2)
        return error

    def log_constants_p_x_given_z0(self, x, node_mask):
        """Computes p(x|z0)."""
        batch_size = x.size(0)

        n_nodes = node_mask.squeeze(2).sum(1)  # N has shape [B]
        assert n_nodes.size() == (batch_size,)
        degrees_of_freedom_x = (n_nodes - 1) * self.n_dims

        zeros = torch.zeros((x.size(0), 1), device=x.device)
        gamma_0 = self.gamma(zeros)

        # Recall that sigma_x = sqrt(sigma_0^2 / alpha_0^2) = SNR(-0.5 gamma_0).
        log_sigma_x = 0.5 * gamma_0.view(batch_size)

        return degrees_of_freedom_x * (- log_sigma_x - 0.5 * np.log(2 * np.pi))

    def sample_p_xh_given_z0(self, z0, node_mask, edge_mask, context, fix_noise=False, zt_chain=None, eps_t_chain=None):
        """Samples x ~ p(x|z0)."""
        zeros = torch.zeros(size=(z0.size(0), 1), device=z0.device)
        gamma_0 = self.gamma(zeros)
        # Computes sqrt(sigma_0^2 / alpha_0^2)
        sigma_x = self.SNR(-0.5 * gamma_0).unsqueeze(1)
        
        if self.uni_diffusion:
            zeros2 = torch.zeros(size=(z0.size(0), 1), device=z0.device)
            gamma_0_2 = self.gamma2(zeros2)
            sigma_x_2 = self.SNR(-0.5 * gamma_0_2).unsqueeze(1)
            net_out, pred = self.phi(z0, zeros, node_mask, edge_mask, context, zeros2)
        else:
            if self.property_pred:
                net_out, pred = self.phi(z0, zeros, node_mask, edge_mask, context)
            else:
                net_out = self.phi(z0, zeros, node_mask, edge_mask, context)
    
        # Compute mu for p(zs | zt).
        mu_x = self.compute_x_pred(net_out, z0, gamma_0)
        if self.dynamics.mode == "PAT" or self.atom_type_pred:
            xh = self.sample_normal(mu=mu_x[:,:,:3], sigma=sigma_x, node_mask=node_mask, fix_noise=fix_noise)
        else:
            xh = self.sample_normal(mu=mu_x, sigma=sigma_x, node_mask=node_mask, fix_noise=fix_noise)
        x = xh[:, :, :self.n_dims]

        h_int = z0[:, :, -1:] if self.include_charges else torch.zeros(0).to(z0.device)
        if self.dynamics.mode == "PAT" or self.atom_type_pred:
            h_cat = net_out[:, :, self.n_dims:self.n_dims+self.num_classes]
            h_int = net_out[:, :, self.n_dims+self.num_classes:self.n_dims+self.num_classes+1]
        else:
            x, h_cat, h_int = self.unnormalize(x, z0[:, :, self.n_dims:-1], h_int, node_mask)
        # print(f"h_cat: {h_cat.shape}, h_int: {h_int.shape}")
        # print(f"h_cat: {h_cat}, h_int: {h_int}")
        h_cat = F.one_hot(torch.argmax(h_cat, dim=2), self.num_classes) * node_mask
        h_int = torch.round(h_int).long() * node_mask

        h = {'integer': h_int, 'categorical': h_cat}
        if zt_chain is not None:
            assert False
            assert self.parametrization == 'eps', 'zt_chain only works with eps parametrization'
            zt_chain.append(torch.cat([x, h_cat], dim=2))
            eps_t_chain.append(net_out)
        if self.property_pred:
            return x, h, pred
        return x, h

    def sample_normal(self, mu, sigma, node_mask, fix_noise=False, only_coord=False):
        """Samples from a Normal distribution."""
        bs = 1 if fix_noise else mu.size(0)
        if only_coord:
            eps = self.sample_combined_position_noise(bs, mu.size(1), node_mask)
        else:
            eps = self.sample_combined_position_feature_noise(bs, mu.size(1), node_mask)
        if eps.shape[2] != mu.shape[2]:
            print("eps: ", eps.shape)
            print("mu: ", mu.shape)
        # print("mu: ", mu.shape, mu)
        # print("sigma: ", sigma.shape, sigma)
        # print("eps: ", eps.shape, eps)
        return mu + sigma * eps
    
    def sample_normal2(self, mu, sigma, node_mask, fix_noise=False):
        """Samples from a Normal distribution."""
        bs = 1 if fix_noise else mu.size(0)
        eps = utils.sample_gaussian_with_mask(
                size=(mu.size(0), 1, 1), device=node_mask.device,
                node_mask=node_mask)
        
        return mu + sigma * eps

    def log_pxh_given_z0_without_constants(
            self, x, h, z_t, gamma_0, eps, net_out, node_mask, epsilon=1e-10):
        # Discrete properties are predicted directly from z_t.
        z_h_cat = z_t[:, :, self.n_dims:-1] if self.include_charges else z_t[:, :, self.n_dims:]
        z_h_int = z_t[:, :, -1:] if self.include_charges else torch.zeros(0).to(z_t.device)

        # Take only part over x.
        eps_x = eps[:, :, :self.n_dims]
        net_x = net_out[:, :, :self.n_dims]

        # Compute sigma_0 and rescale to the integer scale of the data.
        sigma_0 = self.sigma(gamma_0, target_tensor=z_t)
        sigma_0_cat = sigma_0 * self.norm_values[1]
        sigma_0_int = sigma_0 * self.norm_values[2]

        # Computes the error for the distribution N(x | 1 / alpha_0 z_0 + sigma_0/alpha_0 eps_0, sigma_0 / alpha_0),
        # the weighting in the epsilon parametrization is exactly '1'.
        log_p_x_given_z_without_constants = -0.5 * self.compute_error(net_x, gamma_0, eps_x) # L_zero first item

        # Compute delta indicator masks.
        h_integer = torch.round(h['integer'] * self.norm_values[2] + self.norm_biases[2]).long()
        onehot = h['categorical'] * self.norm_values[1] + self.norm_biases[1]

        estimated_h_integer = z_h_int * self.norm_values[2] + self.norm_biases[2]
        estimated_h_cat = z_h_cat * self.norm_values[1] + self.norm_biases[1]
        assert h_integer.size() == estimated_h_integer.size()

        h_integer_centered = h_integer - estimated_h_integer

        # Compute integral from -0.5 to 0.5 of the normal distribution
        # N(mean=h_integer_centered, stdev=sigma_0_int)
        log_ph_integer = torch.log(
            cdf_standard_gaussian((h_integer_centered + 0.5) / sigma_0_int)
            - cdf_standard_gaussian((h_integer_centered - 0.5) / sigma_0_int)
            + epsilon)
        log_ph_integer = sum_except_batch(log_ph_integer * node_mask)

        # Centered h_cat around 1, since onehot encoded.
        centered_h_cat = estimated_h_cat - 1

        # Compute integrals from 0.5 to 1.5 of the normal distribution
        # N(mean=z_h_cat, stdev=sigma_0_cat)
        log_ph_cat_proportional = torch.log(
            cdf_standard_gaussian((centered_h_cat + 0.5) / sigma_0_cat)
            - cdf_standard_gaussian((centered_h_cat - 0.5) / sigma_0_cat)
            + epsilon)

        # Normalize the distribution over the categories.
        log_Z = torch.logsumexp(log_ph_cat_proportional, dim=2, keepdim=True)
        log_probabilities = log_ph_cat_proportional - log_Z

        # Select the log_prob of the current category usign the onehot
        # representation.
        log_ph_cat = sum_except_batch(log_probabilities * onehot * node_mask)

        # Combine categorical and integer log-probabilities.
        log_p_h_given_z = log_ph_integer + log_ph_cat

        # Combine log probabilities for x and h.
        log_p_xh_given_z = log_p_x_given_z_without_constants + log_p_h_given_z

        return log_p_xh_given_z
    
    
    def compute_loss_exp(self, x, h, node_mask, edge_mask, context, t0_always, sigma_0=0.04):
        """expand frist and add guassian secondly"""
        
        # expand first
        lowest_t = 0
        t_int_c = torch.randint(
            lowest_t, self.T, size=(x.size(0), 1), device=x.device).float()
        
        scale = t_int_c / self.T # 0-1
        
        # first step
        batch_size = x.shape[0]
        
        x_new = x.clone()
        for i in range(batch_size):
            x_new[i] = x[i] * scale[i] # scale molecular for each batch
        
        
        # the atom type part
        if t0_always:
            # loss_term_0 will be computed separately.
            # estimator = loss_0 + loss_t,  where t ~ U({1, ..., T})
            lowest_t = 1
        else:
            # estimator = loss_t,           where t ~ U({0, ..., T})
            # indicate the training, we can switch the mask indicator
            # if self.pre_training:
            #     self.mask_indicator = not self.mask_indicator
            lowest_t = 0

        # Sample a timestep t.
        t_int = torch.randint(
            lowest_t, self.T + 1, size=(x.size(0), 1), device=x.device).float()
        
        s_int = t_int - 1
        t_is_zero = (t_int == 0).float()  # Important to compute log p(x | z0).

        # Normalize t to [0, 1]. Note that the negative
        # step of s will never be used, since then p(x | z0) is computed.
        # t_int[0] = 1
        # t_int[1] = 2
        s = s_int / self.T
        t = t_int / self.T

        
        # Compute gamma_s and gamma_t via the network.
        gamma_s = self.inflate_batch_array(self.gamma(s), x)
        gamma_t = self.inflate_batch_array(self.gamma(t), x)

        # Compute alpha_t and sigma_t from gamma.
        alpha_t = self.alpha(gamma_t, x)
        sigma_t = self.sigma(gamma_t, x)
        
        xh = torch.cat([x_new, h['categorical'], h['integer']], dim=2)
        
        eps = self.sample_combined_position_feature_noise(
            n_samples=x.size(0), n_nodes=x.size(1), node_mask=node_mask)
        
        alpha_tr = alpha_t.repeat(1, xh.shape[1], xh.shape[2])
        sigma_tr = sigma_t.repeat(1, xh.shape[1], xh.shape[2])
        
        # z_t = alpha_tr * xh + sigma_tr * eps
        alpha_tr[:,:, :3] = 1
        sigma_tr[:,:, :3] = sigma_0
        
        z_t = alpha_tr * xh + sigma_tr * eps
        diffusion_utils.assert_mean_zero_with_mask(z_t[:, :, :self.n_dims], node_mask)
        
        net_out = self.phi(z_t, t, node_mask, edge_mask, context)
        
        error = self.compute_error(net_out, gamma_t, eps)
        
        if self.training and self.loss_type == 'l2':
            SNR_weight = torch.ones_like(error)
        else:
            # Compute weighting with SNR: (SNR(s-t) - 1) for epsilon parametrization.
            SNR_weight = (self.SNR(gamma_s - gamma_t) - 1).squeeze(1).squeeze(1)
        assert error.size() == SNR_weight.size()
        loss_t_larger_than_zero = 0.5 * SNR_weight * error

        # The _constants_ depending on sigma_0 from the
        # cross entropy term E_q(z0 | x) [log p(x | z0)].
        neg_log_constants = -self.log_constants_p_x_given_z0(x, node_mask)

        # Reset constants during training with l2 loss.
        if self.training and self.loss_type == 'l2':
            neg_log_constants = torch.zeros_like(neg_log_constants)

        # The KL between q(zT | x) and p(zT) = Normal(0, 1). Should be close to zero.
        kl_prior = self.kl_prior(xh, node_mask)

        # Combining the terms
        if t0_always:
            loss_t = loss_t_larger_than_zero
            num_terms = self.T  # Since t=0 is not included here.
            estimator_loss_terms = num_terms * loss_t

            # Compute noise values for t = 0.
            t_zeros = torch.zeros_like(s)
            gamma_0 = self.inflate_batch_array(self.gamma(t_zeros), x)
            alpha_0 = self.alpha(gamma_0, x)
            sigma_0 = self.sigma(gamma_0, x)

            # Sample z_0 given x, h for timestep t, from q(z_t | x, h)
            eps_0 = self.sample_combined_position_feature_noise(
                n_samples=x.size(0), n_nodes=x.size(1), node_mask=node_mask)
            z_0 = alpha_0 * xh + sigma_0 * eps_0 # TODO fix this
            
            
            
            net_out = self.phi(z_0, t_zeros, node_mask, edge_mask, context)

            loss_term_0 = -self.log_pxh_given_z0_without_constants(
                x, h, z_0, gamma_0, eps_0, net_out, node_mask)

            assert kl_prior.size() == estimator_loss_terms.size()
            assert kl_prior.size() == neg_log_constants.size()
            assert kl_prior.size() == loss_term_0.size()
            
            loss = kl_prior + estimator_loss_terms + neg_log_constants + loss_term_0

        else:
            # Computes the L_0 term (even if gamma_t is not actually gamma_0)
            # and this will later be selected via masking.
            loss_term_0 = -self.log_pxh_given_z0_without_constants(
                x, h, z_t, gamma_t, eps, net_out, node_mask)

            t_is_not_zero = 1 - t_is_zero

            loss_t = loss_term_0 * t_is_zero.squeeze() + t_is_not_zero.squeeze() * loss_t_larger_than_zero

            # Only upweigh estimator if using the vlb objective.
            if self.training and self.loss_type == 'l2':
                estimator_loss_terms = loss_t
            else:
                num_terms = self.T + 1  # Includes t = 0.
                estimator_loss_terms = num_terms * loss_t

            assert kl_prior.size() == estimator_loss_terms.size()
            assert kl_prior.size() == neg_log_constants.size()

            loss = kl_prior + estimator_loss_terms + neg_log_constants
        

        assert len(loss.shape) == 1, f'{loss.shape} has more than only batch dim.'

        

        
        return loss, {'t': t_int.squeeze(), 'loss_t': loss.squeeze(),
                      'error': error.squeeze()}        

        

    def compute_loss(self, x, h, node_mask, edge_mask, context, t0_always, mask_indicator=None,
                     property_label=None, time_upperbond=-1, train_prop_pred_4condition_only=False):
        """Computes an estimator for the variational lower bound, or the simple loss (MSE)."""

        # This part is about whether to include loss term 0 always.
        if t0_always:
            # loss_term_0 will be computed separately.
            # estimator = loss_0 + loss_t,  where t ~ U({1, ..., T})
            lowest_t = 1
        else:
            # estimator = loss_t,           where t ~ U({0, ..., T})
            # indicate the training, we can switch the mask indicator
            # if self.pre_training:
            #     self.mask_indicator = not self.mask_indicator
            lowest_t = 0

        # Sample a timestep t.
        if self.property_pred:
            # 0.5 rate use T+1, 0.5 rate use prediction_threshold_t
            random_number = torch.rand(1).item()
            # if random_number < -1:
            if self.unnormal_time_step:
                random_th = 0.5
            elif self.only_noisy_node:
                random_th = 1
            else:
                random_th = -1
            
            if random_number < random_th:
                random_prop = True
                t_int = torch.randint(
                    lowest_t, self.prediction_threshold_t + 1, size=(x.size(0), 1), device=x.device).float()#lowest_t+1
            else:
                random_prop = False
                t_int = torch.randint(
                    lowest_t, self.T + 1, size=(x.size(0), 1), device=x.device).float()#lowest_t+1
            # print("t_int: ", t_int)
        else:
            t_int = torch.randint(
                    lowest_t, self.T + 1, size=(x.size(0), 1), device=x.device).float()
        
        if time_upperbond >= 0:
            t_int = torch.ones_like(t_int) * time_upperbond
            # t_int = torch.randint(
            #         lowest_t, time_upperbond + 1, size=(x.size(0), 1), device=x.device).float()
        
        if self.half_noisy_node:
            batch_size = x.size(0)
            half_batch_size = batch_size // 2
            t_int[half_batch_size:,:] = torch.randint(
                    lowest_t, self.prediction_threshold_t + 1, size=(half_batch_size, 1), device=x.device).float()
            t_int[:half_batch_size,:] = torch.randint(lowest_t, self.T + 1, size=(half_batch_size, 1), device=x.device).float()
        
        if self.sep_noisy_node:
            batch_size = x.size(0)
            half_batch_size = batch_size // 2
            t_int[half_batch_size:,:] = torch.randint(
                    lowest_t, self.prediction_threshold_t + 1, size=(half_batch_size, 1), device=x.device).float()
            t_int[:half_batch_size,:] = torch.randint(self.prediction_threshold_t + 1, self.T + 1, size=(half_batch_size, 1), device=x.device).float()
        
        if self.uni_diffusion:
            
            if self.pre_training and mask_indicator:
                # set latter half of t_int to 28
                batch_size = x.size(0)
                t_int[batch_size//2:,:] = 28
            else:
                # custom sampling strategy
                batch_size = x.size(0)
                quarter_batch_size = batch_size // 4
                if t0_always:
                    t_int[quarter_batch_size:quarter_batch_size * 2, :] = 1 # if t0_always: lowest_t = 1
                else:  
                    t_int[quarter_batch_size:quarter_batch_size * 2, :] = 0
        
        
        s_int = t_int - 1
        t_is_zero = (t_int == 0).float()  # Important to compute log p(x | z0).

        # Normalize t to [0, 1]. Note that the negative
        # step of s will never be used, since then p(x | z0) is computed.
        # t_int[0] = 1
        # t_int[1] = 2
        s = s_int / self.T
        t = t_int / self.T

        
        # Compute gamma_s and gamma_t via the network.
        gamma_s = self.inflate_batch_array(self.gamma(s), x)
        gamma_t = self.inflate_batch_array(self.gamma(t), x)

        # Compute alpha_t and sigma_t from gamma.
        alpha_t = self.alpha(gamma_t, x)
        sigma_t = self.sigma(gamma_t, x)
        
        # alpha: torch.sqrt(torch.sigmoid(-gamma)
        # sigma: torch.sqrt(torch.sigmoid(gamma))
        # alpha_vlst = []
        # sigma_vlst = []
        # for i in range(1, 1001):
        #     vi = i / 1000.0
        #     vi = self.gamma(torch.tensor(vi))
        #     alpha_v = torch.sqrt(torch.sigmoid(-vi))
        #     sigma_v = torch.sqrt(torch.sigmoid(vi))
        #     alpha_vlst.append(alpha_v.item())
        #     sigma_vlst.append(sigma_v.item())
            
        #     print(f"i: {i}, alpha: {alpha_v}, sigma: {sigma_v}")
        # # save alpha and sigma as npy
        # np.save("alpha.npy", alpha_vlst)
        # np.save("sigma.npy", sigma_vlst)

        # Sample zt ~ Normal(alpha_t x, sigma_t)
        eps = self.sample_combined_position_feature_noise(
            n_samples=x.size(0), n_nodes=x.size(1), node_mask=node_mask)

        # Concatenate x, h[integer] and h[categorical].
        xh = torch.cat([x, h['categorical'], h['integer']], dim=2)
        
        fix_h = torch.ones_like(torch.cat([h['categorical'], h['integer']], dim=2))
        # Sample z_t given x, h for timestep t, from q(z_t | x, h)
        if self.dynamics.mode == "PAT" or self.atom_type_pred:
            z_t = alpha_t * x + sigma_t * eps
            z_t = torch.cat([z_t, fix_h], dim=2)
        else:
            z_t = alpha_t * xh + sigma_t * eps

        diffusion_utils.assert_mean_zero_with_mask(z_t[:, :, :self.n_dims], node_mask)
        
        if self.uni_diffusion:
            t_int2 = torch.randint(
                lowest_t, self.T2 + 1, size=(x.size(0), 1), device=x.device).float()
            
            quarter_batch_size = batch_size // 4
            # sampling stategy
            t_int2[:quarter_batch_size, :] = 0
            
            third_quarter_numbers = t_int[quarter_batch_size * 2:quarter_batch_size * 3]
            
            
            for idx, num in enumerate(third_quarter_numbers):
                # t_int2[idx + quarter_batch_size * 2, 0] = num // 100                
                t_int2[idx + quarter_batch_size * 2, 0] = num
            
            s_int2 = t_int2 - 1
            t_is_zero2 = (t_int2 == 0).float()  # Important to compute log p(x | z0).

            # Normalize t to [0, 1]. Note that the negative
            # step of s will never be used, since then p(x | z0) is computed.
            # t_int2[0] = 1
            # t_int2[1] = 2
            s2 = s_int2 / self.T2
            t2 = t_int2 / self.T2

            
            # Compute gamma_s and gamma_t via the network.
            gamma_s2 = self.inflate_batch_array(self.gamma2(s2), context)
            gamma_t2 = self.inflate_batch_array(self.gamma2(t2), context)

            # Compute alpha_t and sigma_t from gamma.
            alpha_t2 = self.alpha(gamma_t2, context)
            sigma_t2 = self.sigma(gamma_t2, context)

            # Sample zt ~ Normal(alpha_t x, sigma_t)
            # eps = self.sample_combined_position_feature_noise(
            #     n_samples=x.size(0), n_nodes=x.size(1), node_mask=node_mask)
            
            
            eps2 = utils.sample_gaussian_with_mask(
                size=(x.size(0), 1, context.shape[-1]), device=node_mask.device,
                node_mask=node_mask)

            # Concatenate x, h[integer] and h[categorical].
            # xh = torch.cat([x, h['categorical'], h['integer']], dim=2)
            # Sample z_t given x, h for timestep t, from q(z_t | x, h)
            context_t = alpha_t2 * context + sigma_t2 * eps2
            
            if self.pre_training and mask_indicator:
                net_out, property_pred = self.phi(z_t, t, node_mask, edge_mask, context_t, t2, mask_y=mask_indicator, no_noise_xh=xh)
                # self.mask_indicator = False
                # print('mask_indicator', self.mask_indicator)
            else:
                # if self.mask_indicator is not None:
                #     self.mask_indicator = True
                #     print('mask_indicator', self.mask_indicator)
                net_out, property_pred = self.phi(z_t, t, node_mask, edge_mask, context_t, t2, no_noise_xh=xh)
        
        else:
            # print("property_pred: ", self.property_pred)
            if self.property_pred or train_prop_pred_4condition_only:
                net_out, property_pred = self.phi(z_t, t, node_mask, edge_mask, context, no_noise_xh=xh)
            else:
                # Neural net prediction.
                net_out = self.phi(z_t, t, node_mask, edge_mask, context)

        # Compute the error.
        if mask_indicator is not None and mask_indicator:
            # split the eps into two parts
            half_batch_size = eps.size(0) // 2
            eps1 = eps[:half_batch_size]
            eps2 = eps[half_batch_size:][:, :, :3]
            error = self.compute_error(net_out[:half_batch_size], gamma_t, eps1)
            # self.dynamics.pos_normalizer
            
            node_num = x.size(1)
            node_mask_later_half = node_mask[half_batch_size:].reshape(half_batch_size* node_num, -1)
            eps2_compress = eps2.reshape(half_batch_size* node_num, 3)
            net_out_compress = net_out[half_batch_size:][:, :, :3].reshape(half_batch_size* node_num, 3)
            # normalise the eps2
            
            eps2_compress = eps2_compress[node_mask_later_half.squeeze().to(torch.bool)]
            net_out_compress = net_out_compress[node_mask_later_half.squeeze().to(torch.bool)]
            
            eps2_compress = self.dynamics.pos_normalizer(eps2_compress)
            
            denoise_error = self.compute_error(net_out_compress, gamma_t, eps2_compress)
            
            atom_num_lst = node_mask[half_batch_size:].sum(dim=1)
            batch_lst = []
            for i, atom_num in enumerate(atom_num_lst):
                current_lst = torch.full([int(atom_num.item())], i)
                batch_lst.append(current_lst)
            batch = torch.cat(batch_lst).to(eps2.device)
            
            denoise_error = scatter_mean(denoise_error, batch, dim=0)
            
            if mask_indicator == 2:# only pretraininig with denoising
                denoise_error = torch.zeros_like(denoise_error)
            
            # concat error and denoise_error
            error = torch.cat([error, denoise_error])
        else:
            error = self.compute_error(net_out, gamma_t, eps)
            # if self.training and self.atom_type_pred and self.property_pred and self.use_prop_pred:
            #     batch_size = error.shape[0]
            #     error[:batch_size//2] *= 99/50 # t > 10
            #     error[batch_size//2:] *= 1/50 # t < 10

        if self.uni_diffusion:
            if mask_indicator is None or not mask_indicator:
                error2 = self.compute_error(property_pred, gamma_t2, eps2[:,0,:])


        if self.training and self.loss_type == 'l2':
            SNR_weight = torch.ones_like(error)
        else:
            # Compute weighting with SNR: (SNR(s-t) - 1) for epsilon parametrization.
            SNR_weight = (self.SNR(gamma_s - gamma_t) - 1).squeeze(1).squeeze(1)
        assert error.size() == SNR_weight.size()
        loss_t_larger_than_zero = 0.5 * SNR_weight * error

        # The _constants_ depending on sigma_0 from the
        # cross entropy term E_q(z0 | x) [log p(x | z0)].
        neg_log_constants = -self.log_constants_p_x_given_z0(x, node_mask)

        # Reset constants during training with l2 loss.
        if self.training and self.loss_type == 'l2':
            neg_log_constants = torch.zeros_like(neg_log_constants)

        # The KL between q(zT | x) and p(zT) = Normal(0, 1). Should be close to zero.
        kl_prior = self.kl_prior(xh, node_mask)

        # Combining the terms
        if t0_always:
            loss_t = loss_t_larger_than_zero
            num_terms = self.T  # Since t=0 is not included here.
            estimator_loss_terms = num_terms * loss_t

            # Compute noise values for t = 0.
            t_zeros = torch.zeros_like(s)
            gamma_0 = self.inflate_batch_array(self.gamma(t_zeros), x)
            alpha_0 = self.alpha(gamma_0, x)
            sigma_0 = self.sigma(gamma_0, x)

            # Sample z_0 given x, h for timestep t, from q(z_t | x, h)
            eps_0 = self.sample_combined_position_feature_noise(
                n_samples=x.size(0), n_nodes=x.size(1), node_mask=node_mask)
            if self.dynamics.mode == "PAT" or self.atom_type_pred:
                z_0 = alpha_0 * x + sigma_0 * eps_0
                z_0 = torch.cat([z_0, fix_h], dim=2)
            else:
                z_0 = alpha_0 * xh + sigma_0 * eps_0
            
            
            if self.uni_diffusion:
                t_zeros2 = torch.zeros_like(s2)
                gamma_02 = self.inflate_batch_array(self.gamma2(t_zeros2), context)
                alpha_02 = self.alpha(gamma_02, context)
                sigma_02 = self.sigma(gamma_02, context)

                # Sample z_0 given x, h for timestep t, from q(z_t | x, h)
                eps_02 = utils.sample_gaussian_with_mask(
                    size=(x.size(0), 1, context.shape[-1]), device=node_mask.device,
                    node_mask=node_mask)
                context_0 = alpha_02 * context + sigma_02 * eps_02
                
                net_out, property_pred = self.phi(z_t, t, node_mask, edge_mask, context_0, t2)
                
                error2 = self.compute_error(property_pred, gamma_t2, eps_02[:,0,:])
                
            else:
                if self.property_pred:
                    net_out, property_pred = self.phi(z_0, t_zeros, node_mask, edge_mask, context)
                else:
                    net_out = self.phi(z_0, t_zeros, node_mask, edge_mask, context)

            loss_term_0 = -self.log_pxh_given_z0_without_constants(
                x, h, z_0, gamma_0, eps_0, net_out, node_mask)

            assert kl_prior.size() == estimator_loss_terms.size()
            assert kl_prior.size() == neg_log_constants.size()
            assert kl_prior.size() == loss_term_0.size()
            
            if self.uni_diffusion:
                loss = kl_prior + estimator_loss_terms + neg_log_constants + loss_term_0 + error2
            else:
                loss = kl_prior + estimator_loss_terms + neg_log_constants + loss_term_0

        else:
            # Computes the L_0 term (even if gamma_t is not actually gamma_0)
            # and this will later be selected via masking.
            loss_term_0 = -self.log_pxh_given_z0_without_constants(
                x, h, z_t, gamma_t, eps, net_out, node_mask)

            t_is_not_zero = 1 - t_is_zero

            loss_t = loss_term_0 * t_is_zero.squeeze() + t_is_not_zero.squeeze() * loss_t_larger_than_zero

            # Only upweigh estimator if using the vlb objective.
            if self.training and self.loss_type == 'l2':
                estimator_loss_terms = loss_t
            else:
                num_terms = self.T + 1  # Includes t = 0.
                estimator_loss_terms = num_terms * loss_t

            assert kl_prior.size() == estimator_loss_terms.size()
            assert kl_prior.size() == neg_log_constants.size()

            loss = kl_prior + estimator_loss_terms + neg_log_constants
        
            if self.uni_diffusion:
                if mask_indicator is None or not mask_indicator:
                    loss += error2

        assert len(loss.shape) == 1, f'{loss.shape} has more than only batch dim.'

        #calc loss for prediction
        if self.property_pred or train_prop_pred_4condition_only:
            if self.target_property is not None:
                #loss for prediction use l1 loss
                loss_l1 = torch.nn.L1Loss(reduction='none')
                # print(property_pred.size(), property_label.size())
                assert property_pred.size() == property_label.size(), f"property_pred size: {property_pred.size()}, property_label size: {property_label.size()}"
                pred_loss = loss_l1(property_pred, property_label)
                if pred_loss.dim() > 1 and pred_loss.size(1) == 53: # basic prob
                    pred_loss = pred_loss.mean(dim=1)
                # print("property_label: ", property_label)
                # print("property_pred: ", property_pred)
                # print("pred_loss", pred_loss)
            else:
                #0 loss for prediction
                pred_loss = torch.zeros_like(property_pred)
            
        if self.dynamics.mode == "PAT" or self.atom_type_pred:
            # calculate the loss for atom type
            h_true = torch.cat([h['categorical'], h['integer']], dim=2).clone().detach().requires_grad_(True).to(torch.float32).to(x.device)
            h_pred = net_out[:, :, 3:]
            #保留第0维度不计算loss，因为第0维度是batch维度
            l1_loss = torch.nn.L1Loss(reduction='none')
            atom_type_loss = l1_loss(h_true, h_pred)
            
            # atom_type_loss = l1_loss(h_true, h_pred)
            atom_type_loss = atom_type_loss * node_mask
            atom_type_loss = atom_type_loss.mean(dim=2).mean(dim=1)
            
            # atom_type_loss = atom_type_loss.sum(dim=2)
            # atom_type_loss = atom_type_loss.sum(dim=1)

        if self.uni_diffusion:
            if mask_indicator is not None and mask_indicator:
                return loss, {'t': t_int.squeeze(), 'loss_t': loss.squeeze(),
                      'error': error.squeeze(), 'denoise_error': denoise_error.squeeze()} # error3: only denoising error
            else:
                return loss, {'t': t_int.squeeze(), 'loss_t': loss.squeeze(),
                      'error': error.squeeze(), 'error2': error2.squeeze()}
        else:
            loss_dict = {'t': t_int.squeeze(), 'loss_t': loss.squeeze(),
                      'error': error.squeeze()}
            if self.property_pred:
                assert self.dynamics.mode == "DGAP", "only DGAP mode support property prediction"
                # TODO check the pred_mask and pred_loss
                #mask the loss if the threshold is reached
                #Set a tensor with the same dimension as t_int, 1 means that t_int is less than or equal to prediction_threshold_t, and 0 means that t_int is greater than prediction_threshold_t.
                pred_loss_mask = (t_int <= self.prediction_threshold_t).float()
                if train_prop_pred_4condition_only: # mask 取反
                    pred_loss_mask = 1 - pred_loss_mask
                if self.use_prop_pred == 0:
                    pred_loss_mask = torch.zeros_like(pred_loss_mask).to(pred_loss_mask.device)
                pred_rate = pred_loss_mask.sum() / pred_loss_mask.size(0)
                loss_dict["pred_rate"] = pred_rate
                pred_loss_mask = pred_loss_mask.squeeze(1)
                pred_loss = pred_loss * pred_loss_mask
                # pred_loss = pred_loss.squeeze(1)
                
                if not t0_always: # training mode
                    if self.freeze_gradient and random_prop:
                        loss = 0 # do not generation
                        self.dynamics.egnn.eval() # freeze backbone, when do the property prediction.
                    elif self.freeze_gradient:
                        pred_loss = 0 # do not do the property prediction when random seed is not less than 0.5
                        self.dynamics.egnn.train() # unfreeze the backbone
                    else:
                        self.dynamics.egnn.train() # unfreeze the backbone
                
                # dynamic adjust the weight
                # pred_loss_weight = (error.mean() / pred_loss.mean()).item()
                pred_loss_weight = 1
                
                
                loss_dict['pred_loss'] = pred_loss * pred_loss_weight
                loss += pred_loss
                
                # loss_dict = {'t': t_int.squeeze(), 'loss_t': loss.squeeze(),
                #       'error': error.squeeze(), "pred_loss": pred_loss, "pred_rate": pred_rate}
            if self.atom_type_pred:
                pred_loss_mask = (t_int <= self.prediction_threshold_t).float()
                pred_loss_mask = pred_loss_mask.squeeze(1)
                atom_type_loss = atom_type_loss * pred_loss_mask
                loss_dict["atom_type_loss"] = atom_type_loss
                loss += atom_type_loss
                
                return loss, loss_dict
            elif self.dynamics.mode == "PAT":
                pred_loss_mask = (t_int <= self.prediction_threshold_t).float()
                pred_rate = pred_loss_mask.sum() / pred_loss_mask.size(0)
                pred_loss_mask = pred_loss_mask.squeeze(1)   
                atom_type_loss = atom_type_loss * pred_loss_mask
                loss += atom_type_loss
                return loss, {'t': t_int.squeeze(), 'loss_t': loss.squeeze(), 'error': error.squeeze(), "atom_type_loss": atom_type_loss, "pred_rate": pred_rate}
            else:
                return loss, {'t': t_int.squeeze(), 'loss_t': loss.squeeze(),
                      'error': error.squeeze()}

    def evaluate_property(self, x, h, org_context, node_mask=None, edge_mask=None):
        # Normalize data, take into account volume change in x.
        x, h, delta_log_px = self.normalize(x, h, node_mask)
        t_int = torch.ones((x.size(0), 1), device=x.device).float() # t_int all zero
        s_int = t_int - 1
        s_array = s_int / self.T
        t_array = t_int / self.T

        
        # Compute gamma_s and gamma_t via the network.
        gamma_s = self.inflate_batch_array(self.gamma(s_array), x)
        gamma_t = self.inflate_batch_array(self.gamma(t_array), x)

        # Compute alpha_t and sigma_t from gamma.
        alpha_t = self.alpha(gamma_t, x)
        sigma_t = self.sigma(gamma_t, x)

        # Sample zt ~ Normal(alpha_t x, sigma_t)
        eps = self.sample_combined_position_feature_noise(
            n_samples=x.size(0), n_nodes=x.size(1), node_mask=node_mask)

        # Concatenate x, h[integer] and h[categorical].
        xh = torch.cat([x, h['categorical'], h['integer']], dim=2)
        # Sample z_t given x, h for timestep t, from q(z_t | x, h)
        z_t = alpha_t * xh + sigma_t * eps

        diffusion_utils.assert_mean_zero_with_mask(z_t[:, :, :self.n_dims], node_mask)
        
        new_context = None
        
        
        # perpare the context, z_t, keep unchanged, copy the sample method
        for s in reversed(range(0, self.T)):
            n_samples= x.size(0)

            s_array2 = torch.full((n_samples, 1), fill_value=s, device=x.device)
            t_array2 = s_array2 + 1
            s_array2 = s_array2 / self.T
            t_array2 = t_array2 / self.T
            
            # sample new_context
            if new_context is None:
                new_context = utils.sample_gaussian_with_mask(
                    size=(x.size(0), 1, org_context.size(-1)), device=node_mask.device,
                    node_mask=node_mask)
            
            z, new_context = self.sample_p_zs_given_zt(
                    s_array, t_array, z_t, node_mask, edge_mask, new_context, fix_noise=False, yt=t_array2, ys=s_array2, force_t_zero=True) # z_t and t keep unchanged
        
        # calcuate the mae between new_context and org_context
        mae = torch.mean(torch.abs(new_context - org_context))
        
        
        return new_context, mae
    
    def forward(self, x, h, node_mask=None, edge_mask=None, context=None, mask_indicator=None, expand_diff=False, property_label=None, train_prop_pred_4condition_only=False):
        """
        Computes the loss (type l2 or NLL) if training. And if eval then always computes NLL.
        """
        if self.property_pred:
            assert property_label is not None, "property_label should not be None in training"
        
        # Normalize data, take into account volume change in x.
        x, h, delta_log_px = self.normalize(x, h, node_mask)
        # print("x.shape: ", x.shape, "x[0]", x[0])
        # print("h[categorical].shape: ", h["categorical"].shape, "h[0]", h["categorical"][0])
        # print("h[integer].shape: ", h["integer"].shape, "h[0]", h["integer"][0])
        # Reset delta_log_px if not vlb objective.
        if self.training and self.loss_type == 'l2':
            delta_log_px = torch.zeros_like(delta_log_px)
            
        # self.eval()
        # with torch.no_grad():
        #     # self.eval()
        #     denoise_error_lst = []
        #     T = 1000
        #     from tqdm import tqdm
        #     for t in tqdm(range(T)):
        #         # if t % 2 == 0:
        #         #     time_upperbond = 10
        #         # else:
        #         #     time_upperbond = 1000
        #         time_upperbond = t
        #         loss, loss_dict = self.compute_loss(x, h, node_mask, edge_mask, context, t0_always=False, mask_indicator=mask_indicator, property_label=property_label, time_upperbond=time_upperbond)
        #         denoise_error_lst.append(loss_dict['error'])
        #         print(f'upperbond: {t}, error: {loss_dict["error"].mean().item()}')
        #     denoise_error = torch.stack(denoise_error_lst, dim=1)
        #     # save denoise_error
        #     torch.save(denoise_error, 'denoise_error_new.pt')
        #     exit(0)

        if self.training:
            # Only 1 forward pass when t0_always is False.
            if expand_diff:
                loss, loss_dict = self.compute_loss_exp(x, h, node_mask, edge_mask, context, t0_always=False)
            else:
                loss, loss_dict = self.compute_loss(x, h, node_mask, edge_mask, context, t0_always=False, mask_indicator=mask_indicator, property_label=property_label,train_prop_pred_4condition_only=train_prop_pred_4condition_only)
        else:
            # Less variance in the estimator, costs two forward passes.
            loss, loss_dict = self.compute_loss(x, h, node_mask, edge_mask, context, t0_always=True, mask_indicator=mask_indicator, property_label=property_label, train_prop_pred_4condition_only=train_prop_pred_4condition_only)

        neg_log_pxh = loss

        # Correct for normalization on x.
        assert neg_log_pxh.size() == delta_log_px.size()
        neg_log_pxh = neg_log_pxh - delta_log_px
        
        return neg_log_pxh, loss_dict
        
        if self.uni_diffusion:
            return neg_log_pxh, loss_dict

        return neg_log_pxh

    def add_guidance(self, pseudo_context, zt, s, t, node_mask, edge_mask, grad_zt, guidance_scale, loss, pred):
        # TODO mask the guidance if loss changing too fast
        # 定义pred变化超过20%的loss，则认为loss变化过快，则不添加guidance
        # add guidance to the loss
        zt_after = zt - grad_zt * guidance_scale
        _, pred_after = self.phi(zt_after, t, node_mask, edge_mask, context=None)
        pred_after = pred_after.unsqueeze(1)
        # print("pred: ", pred, "pred_after: ", pred_after)
        grad_mask = (torch.abs(pred_after - pred) / torch.abs(pred) < 0.2).float().unsqueeze(2)
        # print("changing rate: ", torch.abs(pred_after - pred) / torch.abs(pred))
        # print("grad_mask: ", grad_mask)
        grad_zt = grad_zt * grad_mask
        zt = zt - grad_zt * guidance_scale
        return zt

    
    def single_add_guidance(self, pseudo_context, zt, s, t, node_mask, edge_mask, context, eps_t, mean=None, mad=None):
        t_int = (t*1000)[0].item()
        # When pseudo context is not None, ensuring condGenConfig is added to the model.
        if pseudo_context is not None and t_int < 1000: # for conditional generation
            with torch.enable_grad():
                loss_fn = torch.nn.L1Loss(reduction='none')
                zt = zt.clone().detach().requires_grad_(True)
                
                self.dynamics.zero_grad()
                # Compute gamma_s and gamma_t via the network.
                gamma_s = self.inflate_batch_array(self.gamma(s), zt)
                gamma_t = self.inflate_batch_array(self.gamma(t), zt)
                # Compute alpha_t and sigma_t from gamma.
                alpha_t = self.alpha(gamma_t, zt)
                sigma_t = self.sigma(gamma_t, zt)

                xh, pred = self.phi(zt, t, node_mask, edge_mask, context)
                # for i in range(xh.shape[0]): # visualize all the samples
                #     one_hot = xh[:,:,3:8]
                #     assert one_hot.shape[2] == 5
                #     charges = None
                #     x = xh[:,:,:3]
                #     mol_dir = os.path.join("condGen", 'eval_molecules')
                #     mol_path = os.path.join(mol_dir, f"{i}")

                #     os.makedirs(mol_path, exist_ok=True)
                #     vis.save_xyz_file(
                #         mol_path,
                #         one_hot[i:i+1], charges, x[i:i+1],
                #         id_from=i, name='molecule_stable',
                #         dataset_info=self.dataset_info,
                #         node_mask=node_mask[i:i+1])
                #     print('Visualizing molecules.')
                #     vis.visualize(
                #         mol_dir, self.dataset_info,
                #         max_num=100, spheres_3d=True)
                
                pred = pred.unsqueeze(1)
                # print("pred: ", pred, flush=True)
                # print("pseudo_context: ", pseudo_context, flush=True)
                # classifier_pred = self.classifier_pred(xh)
                # print("classifier_pred: ", classifier_pred)
                # loss = loss_fn(pred, pseudo_context)
                # print("mean in sampling: ", mean)
                if t_int >= 10:
                    loss = loss_fn(mad * pred + mean, pseudo_context)
                    # print("renormalization property: ", mad * pred + mean)
                else:
                    if self.target_property == "lumo" or self.target_property == "gap" or self.target_property == "homo":
                        print("no renormalization")
                        loss = loss_fn(pred, pseudo_context)
                    else:
                        loss = loss_fn(mad * pred + mean, pseudo_context)
                        # print("renormalization property: ", mad * pred + mean)
                ori_loss = loss.clone().detach()
                # print("loss.shape: ", loss.shape, flush=True)
                loss = loss.sum() / loss.shape[0]
                # print("LOSS: ", loss)
                print("t: ", int((t*1000)[0].item()))
                if t_int < 10:
                    print("growing conditional generation prop pred loss: ", loss)
                else:
                    print("nucleation conditional generation prop pred loss: ", loss)
                loss.backward()              
                                
                grad_zt = zt.grad
                # print(f't is {(t*1000)[0].item()} grad_zt: {grad_zt[node_mask.squeeze().to(torch.bool)].abs().mean()}, l1 loss: {loss}', flush=True)
                if True:#loss < self.condGenConfig["loss_maximum"] and loss < self.cond_gen_loss
                    if self.cond_gen_loss is None:
                        self.cond_gen_loss = ori_loss
                    grad_mask = (ori_loss <= self.cond_gen_loss).unsqueeze(2).to(zt.device)
                    self.cond_gen_loss = torch.min(self.cond_gen_loss, ori_loss)
                    # print("update zt")
                    if t_int < 10:
                            # zt = zt - self.condGenConfig["growing"]["guidance_scale"] * grad_zt 
                            zt = self.add_guidance(pseudo_context, zt, s, t, node_mask, edge_mask, grad_zt, self.condGenConfig["growing"]["guidance_scale"], loss, pred)
                    elif (self.condGenConfig["nucleation"]["guidance_end_time"] < t_int < self.condGenConfig["nucleation"]["guidance_time"]):
                        if True:#loss.item() <= self.cond_gen_loss
                            # zt = zt - self.condGenConfig["nucleation"]["guidance_scale"] * grad_zt 
                            zt = self.add_guidance(pseudo_context, zt, s, t, node_mask, edge_mask, grad_zt, self.condGenConfig["nucleation"]["guidance_scale"], loss, pred)
                        # self.cond_gen_loss = loss.item()
                    elif (t_int >= 10 and t_int < self.condGenConfig["nucleation"]["guidance_end_time"]):
                        # 从nucleation guidance_scale到growing guidance_scale平滑下降
                        guidance_scale = self.condGenConfig["nucleation"]["guidance_scale"] + (self.condGenConfig["growing"]["guidance_scale"] - self.condGenConfig["nucleation"]["guidance_scale"]) * (t_int - self.condGenConfig["nucleation"]["guidance_end_time"]) / (10 - self.condGenConfig["nucleation"]["guidance_end_time"])
                        # print(f"guidance_scale: {guidance_scale}")
                        # zt = zt - self.condGenConfig["nucleation"]["guidance_scale"] * grad_zt 
                        zt = self.add_guidance(pseudo_context, zt, s, t, node_mask, edge_mask, grad_zt, guidance_scale, loss, pred)
                # _, pred_after = self.phi(zt, t-0.001, node_mask, edge_mask, context)
                # print("pred after: ", pred_after)
        return zt
    
    def iter_add_guidance(self, pseudo_context, zt, s, t, node_mask, edge_mask, context, eps_t):
        t_int = (t*1000)[0].item()
        # When pseudo context is not None, ensuring condGenConfig is added to the model.
        if pseudo_context is not None and t_int < self.condGenConfig["guidance_time"] and (t_int % 3): # for conditional generation
            with torch.enable_grad():
                loss_fn = torch.nn.L1Loss()
                zt = zt.clone().detach().requires_grad_(True)
                if (t*1000)[0].item() < 10:
                    # its = 20
                    its = self.condGenConfig["growing"]["iter_time"]
                    # opt = torch.optim.Adam([zt], lr=0.001)
                    # opt = torch.optim.SGD([zt], lr=0.001)
                    if self.condGenConfig["growing"]["optimizer"] == "adam":
                        opt = torch.optim.Adam([zt], lr=self.condGenConfig["growing"]["lr"])
                    elif self.condGenConfig["growing"]["optimizer"] == "sgd":
                        opt = torch.optim.SGD([zt], lr=self.condGenConfig["growing"]["lr"])
                else:
                    # its = 5
                    its = self.condGenConfig["nucleation"]["iter_time"]
                    if self.condGenConfig["nucleation"]["optimizer"] == "adam":
                        opt = torch.optim.Adam([zt], lr=self.condGenConfig["nucleation"]["lr"])
                    elif self.condGenConfig["nucleation"]["optimizer"] == "sgd":
                        opt = torch.optim.SGD([zt], lr=self.condGenConfig["nucleation"]["lr"])
                for i in range(its):
                    self.dynamics.zero_grad()
                    # Compute gamma_s and gamma_t via the network.
                    gamma_s = self.inflate_batch_array(self.gamma(s), zt)
                    gamma_t = self.inflate_batch_array(self.gamma(t), zt)
                    # Compute alpha_t and sigma_t from gamma.
                    alpha_t = self.alpha(gamma_t, zt)
                    sigma_t = self.sigma(gamma_t, zt)
                    
                    if zt.shape[-1] != eps_t.shape[-1]:
                        eps_tmp = eps_t[:,:,:3].clone().detach()
                    else:
                        eps_tmp = eps_t.clone().detach()
                    
                    # z0 = (zt * node_mask - sigma_t * eps_tmp) / alpha_t
                    # z0 = diffusion_utils.remove_mean_with_mask(z0,
                    #                                     node_mask)
                    # t0 = torch.ones_like(t) * 0.001 
                    # _, pred = self.phi(zt, t0, node_mask, edge_mask, context)
                    
                    
                    # _, pred = self.phi(z0, t0, node_mask, edge_mask, context)
                    _, pred = self.phi(zt, t, node_mask, edge_mask, context)
                    
                    # print("pred: ", pred)
                    # print("pseudo_context: ", pseudo_context)
                    # print(pred.shape, pseudo_context.shape)
                    loss = loss_fn(pred, pseudo_context)
                    if i == its - 1:
                        print("t: ", int((t*1000)[0].item()))
                        if (t*1000)[0].item() < 10:
                            print("growing conditional generation prop pred loss: ", loss)
                        else:
                            print("nucleation conditional generation prop pred loss: ", loss)
                    # print("conditional generation prop pred loss: ", loss)
                    # grad_zt = torch.autograd.grad(loss, zt, create_graph=True)[0]
                    # if zt.grad is not None:
                    #     zt.grad.zero_()
                    loss.backward()
                    # if int((t*1000)[0].item()) == 10 or int((t*1000)[0].item()) == 99:
                    #     # 显示更新了哪些参数
                    #     print("nucleation time" if int((t*1000)[0].item()) == 99 else "growing time")
                    #     for name, param in self.named_parameters():
                    #         if param.grad is not None and param.grad.abs().mean() > 0:
                    #             print(f"{name}")
                            #前缀匹配 dynamics.blur_node_decode
                            # if 'dynamics.blur_node_decode' in name:
                            #     print(f"{name}: {param.grad} is trainable: {param.requires_grad} param: {param}")

                    # 更新参数
                    opt.step()

                    # 显示参数更新后的值
                    # print("\n更新后的参数值:")
                    # for name, param in self.named_parameters():
                    #     print(f"{name}: {param.data}")
                    
                                    
                    grad_zt = zt.grad
                    if i == its - 1:
                        # if grad_zt[node_mask.squeeze().to(torch.bool)].abs().mean() <= 1e-6:
                        #     # output all grad not zero
                        #     for name, param in self.named_parameters():
                        #         if param.grad is not None:# and param.grad.abs().mean() > 0
                        #             print(f"{name}: {param.grad} is trainable: {param.requires_grad} param: {param}")
                        print(f't is {(t*1000)[0].item()} grad_zt: {grad_zt[node_mask.squeeze().to(torch.bool)].abs().mean()}, l1 loss: {loss.item()}')
                    # zt = zt - 0.001 * grad_zt
                    
                    opt.zero_grad() # TODO    
        return zt
        

    def sample_p_zs_given_zt(self, s, t, zt, node_mask, edge_mask, context, fix_noise=False, yt=None, ys=None, force_t_zero=False, force_t2_zero=False, pseudo_context=None, mean=None, mad=None, conditional_sampling=False, no_noise_xh=None,
    zt_chain=None, eps_t_chain=None):
        """Samples from zs ~ p(zs | zt). Only used during sampling."""
        assert t.max() < 1 + 1e-3, f"t.max() is {t.max()}"
        gamma_s = self.gamma(s)
        gamma_t = self.gamma(t)

        sigma2_t_given_s, sigma_t_given_s, alpha_t_given_s = \
            self.sigma_and_alpha_t_given_s(gamma_t, gamma_s, zt)

        sigma_s = self.sigma(gamma_s, target_tensor=zt)
        sigma_t = self.sigma(gamma_t, target_tensor=zt)
        
        # print("z_t shape: ", zt.shape)

        if self.uni_diffusion:
            gamma_s2 = self.gamma2(ys)
            gamma_t2 = self.gamma2(yt)
            
            sigma2_t_given_s2, sigma_t_given_s2, alpha_t_given_s2 = \
                self.sigma_and_alpha_t_given_s(gamma_t2, gamma_s2, context)
            
            sigma_s2 = self.sigma(gamma_s2, target_tensor=context)
            sigma_t2 = self.sigma(gamma_t2, target_tensor=context)
            
            # Neural net prediction. TODO t=0 when property prediction
            if force_t_zero: # conditional generate property
                t = torch.zeros_like(t)
            if force_t2_zero: # conditional generate molecule
                yt = torch.zeros_like(yt)    
            
            eps_t, property_pred = self.phi(zt, t, node_mask, edge_mask, context, t2=yt)
            
            property_pred_pad = torch.zeros_like(context)
            
            for i in range(context.size(0)):
                pad_idx = node_mask[i].squeeze().to(torch.bool)
                property_pred_pad[i,pad_idx,:] = property_pred[i]
            
            
            # get property prediction mu2 and sigma2
            mu2 = context / alpha_t_given_s2 - (sigma2_t_given_s2 / alpha_t_given_s2 / sigma_t2) * property_pred_pad
            sigma2 = sigma_t_given_s2 * sigma_s2 / sigma_t2
            
            property_pred_update = self.sample_normal2(mu2, sigma2, node_mask, fix_noise)
            
            
            

        # Neural net prediction.
        else:
            # with torch.enable_grad():
            if self.property_pred:
                '''
                return in phi function
                if self.property_pred:
                    return (torch.cat([vel, h_final], dim=2), pred)
                '''
                # if pseudo_context is not None:
                #     zt = zt.clone().detach().requires_grad_(True)
                    # zt.requires_grad = True
                eps_t, pred = self.phi(zt, t, node_mask, edge_mask, context, no_noise_xh=no_noise_xh)
            else:
                eps_t = self.phi(zt.detach(), t, node_mask, edge_mask, context, no_noise_xh=no_noise_xh)
                # eps_t1 = self.phi(zt.detach(), t, node_mask, edge_mask, context, no_noise_xh=no_noise_xh)
                # print("delta eps: ", torch.sum(torch.abs(eps_t - eps_t1)))
        
        # if pseudo_context is not None:
            # and (t*1000)[0].item() < 100:
            
            
        # Compute mu for p(zs | zt).
        diffusion_utils.assert_mean_zero_with_mask(zt[:, :, :self.n_dims], node_mask)
        diffusion_utils.assert_mean_zero_with_mask(eps_t[:, :, :self.n_dims], node_mask)
        # print("eps_t_size: ", eps_t.size())
        # print("zt_size: ", zt.size())
        # print("t: ", t.values)

        if self.condGenConfig != None:
            if self.condGenConfig["guidance_type"] == "iter":#NO USE
                zt = self.iter_add_guidance(pseudo_context, zt, s, t, node_mask, edge_mask, context, eps_t)
            elif self.condGenConfig["guidance_type"] == "single":
                zt = self.single_add_guidance(pseudo_context, zt, s, t, node_mask, edge_mask, context, eps_t, mean=mean, mad=mad)
        

        if self.dynamics.mode == "PAT" or self.atom_type_pred:
            atom_type_pred = eps_t[:, :, 3:]
            mu = zt / alpha_t_given_s - (sigma2_t_given_s / alpha_t_given_s / sigma_t) * eps_t[:,:,0:3].clone()
        else:
            mu = zt / alpha_t_given_s - (sigma2_t_given_s / alpha_t_given_s / sigma_t) * eps_t.clone()


        # Compute sigma for p(zs | zt).
        sigma = sigma_t_given_s * sigma_s / sigma_t

        # Sample zs given the paramters derived from zt.
        zs = self.sample_normal(mu, sigma, node_mask, fix_noise)

        # if zt_chain is not None:
        #     # assert self.condGenConfig == None, "no guidance in DPO"
        #     if self.atom_type_pred:
        #         zt_chain.append(zs[:, :, :self.n_dims].clone())
        #         eps_t_chain.append(eps_t[:, :, :self.n_dims].clone())
        #     else:
        #         assert zs.shape[-1] == self.n_dims+5+self.include_charges, f"zs shape is not correct, {zs.shape}"
        #         zt_chain.append(zs[:, :, :self.n_dims+5].clone())
        #         eps_t_chain.append(eps_t[:, :, :self.n_dims+5].clone())
                

        # Project down to avoid numerical runaway of the center of gravity.
        if self.dynamics.mode == "PAT" or self.atom_type_pred:
            # print("remove mean with mask zs[:, :, :self.n_dims]: ", zs[:, :, :self.n_dims])
            # print("node_mask: ", node_mask)
            zs = torch.cat(
                [diffusion_utils.remove_mean_with_mask(zs[:, :, :self.n_dims],
                                                    node_mask),
                atom_type_pred], dim=2
            )
        else:
            zs = torch.cat(
                [diffusion_utils.remove_mean_with_mask(zs[:, :, :self.n_dims],
                                                    node_mask),
                zs[:, :, self.n_dims:]], dim=2
            )

        if zt_chain is not None:
            # assert self.condGenConfig == None, "no guidance in DPO"
            if self.atom_type_pred:
                zt_chain.append(zs[:, :, :self.n_dims].clone())
                eps_t_chain.append(eps_t[:, :, :self.n_dims].clone())
            else:
                assert zs.shape[-1] == self.n_dims+5+self.include_charges, f"zs shape is not correct, {zs.shape}"
                zt_chain.append(zs[:, :, :self.n_dims+5].clone())
                eps_t_chain.append(eps_t[:, :, :self.n_dims+5].clone())
        
        if self.uni_diffusion:
            return zs, property_pred_update
        
        return zs
    
    
    def sample_p_zs_given_zt_annel_lang(self, s, t, zt, node_mask, edge_mask, context, fix_noise=False, yt=None, ys=None, force_t_zero=False, force_t2_zero=False, T2=10, sigma_n=0.04):
        
        
        
        """Samples from zs ~ p(zs | zt). Only used during sampling."""
        gamma_s = self.gamma(s)
        gamma_t = self.gamma(t)

        sigma2_t_given_s, sigma_t_given_s, alpha_t_given_s = \
            self.sigma_and_alpha_t_given_s(gamma_t, gamma_s, zt)

        sigma_s = self.sigma(gamma_s, target_tensor=zt)
        sigma_t = self.sigma(gamma_t, target_tensor=zt)


        if self.uni_diffusion:
            gamma_s2 = self.gamma2(ys)
            gamma_t2 = self.gamma2(yt)
            
            sigma2_t_given_s2, sigma_t_given_s2, alpha_t_given_s2 = \
                self.sigma_and_alpha_t_given_s(gamma_t2, gamma_s2, context)
            
            sigma_s2 = self.sigma(gamma_s2, target_tensor=context)
            sigma_t2 = self.sigma(gamma_t2, target_tensor=context)
            
            # Neural net prediction. TODO t=0 when property prediction
            if force_t_zero: # conditional generate property
                t = torch.zeros_like(t)
            if force_t2_zero: # conditional generate molecule
                yt = torch.zeros_like(yt)    
            
            eps_t, property_pred = self.phi(zt, t, node_mask, edge_mask, context, t2=yt)
            
            property_pred_pad = torch.zeros_like(context)
            
            for i in range(context.size(0)):
                pad_idx = node_mask[i].squeeze().to(torch.bool)
                property_pred_pad[i,pad_idx,:] = property_pred[i]
            
            
            # get property prediction mu2 and sigma2
            mu2 = context / alpha_t_given_s2 - (sigma2_t_given_s2 / alpha_t_given_s2 / sigma_t2) * property_pred_pad
            sigma2 = sigma_t_given_s2 * sigma_s2 / sigma_t2
            
            property_pred_update = self.sample_normal2(mu2, sigma2, node_mask, fix_noise)
            
            
            

        # Neural net prediction.
        else:
            eps_t = self.phi(zt, t, node_mask, edge_mask, context)

        # Compute mu for p(zs | zt).
        diffusion_utils.assert_mean_zero_with_mask(zt[:, :, :self.n_dims], node_mask)
        diffusion_utils.assert_mean_zero_with_mask(eps_t[:, :, :self.n_dims], node_mask)
        mu = zt / alpha_t_given_s - (sigma2_t_given_s / alpha_t_given_s / sigma_t) * eps_t

        # Compute sigma for p(zs | zt).
        sigma = sigma_t_given_s * sigma_s / sigma_t

        # Sample zs given the paramters derived from zt.
        zs = self.sample_normal(mu, sigma, node_mask, fix_noise)

        # print(f't is {t[0].item()}, sigma: {sigma[0].item()}, z coeffient: {(1 / alpha_t_given_s)[0][0].item()}, nn output coeffient: {(sigma2_t_given_s / alpha_t_given_s / sigma_t)[0][0].item()}')
        # Project down to avoid numerical runaway of the center of gravity.
        
        # split the zs into two parts
        z_coord = zs[:, :, :self.n_dims]
        z_atomtype = zs[:, :, self.n_dims:]
        
        
        step = self.T // T2
        for i in range(T2):
            # caluate the mean
            gamma_idx = step * i
            gamma_t = torch.tensor(self.gamma_lst[gamma_idx], dtype=torch.float32, device=zs.device)
            eps_coord = eps_t[:,:,:self.n_dims]
            
            
            z_coord_mu = z_coord - (1 /sigma_n) * gamma_t * eps_coord
            # + torch.sqrt(2 * gamma_t) * zs_coords
            z_coord_sigma = torch.sqrt(2 * gamma_t).repeat(z_coord_mu.size(0), 1, 1)
            
            zs_coords = self.sample_normal(z_coord_mu, z_coord_sigma, node_mask, fix_noise, only_coord=True)
            
            # update the zs and the eps_t
            zs = torch.cat([zs_coords, z_atomtype], dim=2)
            eps_t = self.phi(zs, t, node_mask, edge_mask, context)
            
            # pass
        
        zs = torch.cat(
            [diffusion_utils.remove_mean_with_mask(zs[:, :, :self.n_dims],
                                                   node_mask),
             zs[:, :, self.n_dims:]], dim=2
        )
        
        if self.uni_diffusion:
            return zs, property_pred_update
        
        return zs

    def sample_p_ys_given_yt(self, s, t, zt, node_mask, edge_mask, context, fix_noise=False):
        """Samples from zs ~ p(zs | zt). Only used during sampling."""
        gamma_s = self.gamma(s)
        gamma_t = self.gamma(t)

        sigma2_t_given_s, sigma_t_given_s, alpha_t_given_s = \
            self.sigma_and_alpha_t_given_s(gamma_t, gamma_s, zt)

        sigma_s = self.sigma(gamma_s, target_tensor=zt)
        sigma_t = self.sigma(gamma_t, target_tensor=zt)

        # Neural net prediction.
        if self.property_pred:
            eps_t, pred = self.phi(zt, t, node_mask, edge_mask, context)
        else:
            eps_t = self.phi(zt, t, node_mask, edge_mask, context)

        # Compute mu for p(zs | zt).
        diffusion_utils.assert_mean_zero_with_mask(zt[:, :, :self.n_dims], node_mask)
        diffusion_utils.assert_mean_zero_with_mask(eps_t[:, :, :self.n_dims], node_mask)
        mu = zt / alpha_t_given_s - (sigma2_t_given_s / alpha_t_given_s / sigma_t) * eps_t

        # Compute sigma for p(zs | zt).
        sigma = sigma_t_given_s * sigma_s / sigma_t

        # Sample zs given the paramters derived from zt.
        zs = self.sample_normal(mu, sigma, node_mask, fix_noise)

        # Project down to avoid numerical runaway of the center of gravity.
        zs = torch.cat(
            [diffusion_utils.remove_mean_with_mask(zs[:, :, :self.n_dims],
                                                   node_mask),
             zs[:, :, self.n_dims:]], dim=2
        )
        return zs


    def sample_combined_position_feature_noise(self, n_samples, n_nodes, node_mask):
        """
        Samples mean-centered normal noise for z_x, and standard normal noise for z_h.
        """
        z_x = utils.sample_center_gravity_zero_gaussian_with_mask(
            size=(n_samples, n_nodes, self.n_dims), device=node_mask.device,
            node_mask=node_mask)
        z_h = utils.sample_gaussian_with_mask(
            size=(n_samples, n_nodes, self.in_node_nf), device=node_mask.device,
            node_mask=node_mask)
        z = torch.cat([z_x, z_h], dim=2)
        if self.dynamics.mode == "PAT" or self.atom_type_pred:
            return z_x
        return z

    def sample_combined_position_noise(self, n_samples, n_nodes, node_mask):
        """
        Samples mean-centered normal noise for z_x, and standard normal noise for z_h.
        """
        z_x = utils.sample_center_gravity_zero_gaussian_with_mask(
            size=(n_samples, n_nodes, self.n_dims), device=node_mask.device,
            node_mask=node_mask)
        return z_x
    
    # @torch.no_grad()
    def dpo_reward(self, z, node_mask, edge_mask, context, dpo_beta, mean, mad, reward_func):
        """
        Computes the reward for the DPO algorithm.
        Reward is negative corespond with property prediction loss.
        return gamma=exp(1 / [dpo_beta * r(z,c)])
        """
        n_samples = z.shape[0]
        assert len(z.shape) == 3, "z0 should be of shape (batch_size, n_nodes, n_features)"
        assert z.shape[2] == 3 + 5, "z0 should be of shape (batch_size, n_nodes, coord_dim + atom_type_dim)"
        t = 1.0 / self.T
        # t = 0.0
        # print("dpo reward t: ", t)
        t = torch.full((z.shape[0], 1), t, device=z.device)
        # print("t: ", t.shape, t[0])
        # print("dpo reward z0: ", z0.shape, z0[0])
        if self.atom_type_pred:
            # print("self.include_charges: ", self.include_charges)
            z = torch.cat([z[:, :, :self.n_dims], torch.ones_like(z[:, :, 0:5+self.include_charges])], dim=2)
            # print("z0.shape: ", z0.shape)
        _, pred = self.phi(z, t, node_mask, edge_mask, context=None) # Predict the property, normalized to [-1, 1]
        loss_fn = torch.nn.L1Loss(reduction='none')
        assert context.shape[0] == n_samples and len(context.shape) == 1, f"context shape {context.shape} should be ({n_samples})"
        assert context.shape == pred.shape, f"context shape {context.shape} should be equal to pred shape {pred.shape}"
        loss = loss_fn(pred, context).sum() / pred.shape[0] 
        loss_reparam = loss_fn(pred * mad + mean, context * mad + mean).sum() / pred.shape[0]
        # assert loss_reparam.mean() < loss.mean(), f"loss_reparam should be smaller than loss, but got {loss_reparam.mean()} and {loss.mean()}"
        # print("dpo reward loss: ", loss)
        # print("dpo reward loss_reparam: ", loss_reparam)
        # gamma = torch.exp(1 / (dpo_beta * loss))
        if reward_func == "minus":
            gamma = torch.exp((1 / dpo_beta) * (-loss))
            assert gamma.item() >= 0 and gamma.item() <= 1, f"gamma should be between 0 and 1, but got {gamma.value}"
        elif reward_func == "exp":
            gamma = torch.exp((1 / dpo_beta) * torch.exp(-loss))
        elif reward_func == "inverse":
            gamma = torch.exp(1 / (dpo_beta * loss))
        else:
            assert False, f"reward function {reward_func} not supported"
        assert gamma.shape == loss_reparam.shape, f"gamma shape {gamma.shape} should be equal to loss shape {loss.shape}"
        # gamma = torch.zeros_like(gamma)
        return gamma, loss_reparam
    
    def dpo_finetune_step(self, z, ref_zt_chain, ref_eps_t_chain, n_samples, gamma, node_mask, edge_mask, context, fix_noise, conditional_sampling, max_n_nodes, optim, wandb=None, stability_mask=None, lr_dict=None, training_scheduler="increase_t"):
        self.train()
        if wandb is not None:
            wandb.log({"γ_mean": gamma.mean()})

        loss_l2 = torch.nn.MSELoss(reduction='none')

        finetune_zt_chain = []
        finetune_eps_t_chain = []

        assert len(ref_zt_chain) == self.T + 1, f"ref_zt_chain should have length {self.T + 1}, but got {len(ref_zt_chain)}"
        assert len(ref_eps_t_chain) == self.T, f"ref_eps_t_chain should have length {self.T}, but got {len(ref_eps_t_chain)}"

        print("ref diffusion mode: ", self.dynamics.mode)
    
        loss_all = 0
        loss_hist = {}

        assert training_scheduler in ["increase_t", "decrease_t", "random"], f"training_scheduler should be increase_t or decrease_t or random, but got {training_scheduler}"

        s_range = range(0, self.T) if training_scheduler == "increase_t" else reversed(range(self.T))
        # z0 = ref_zt_chain[0]

        for s in s_range:
            if training_scheduler == "random":
                s = random.randint(0, self.T-1)
            print("finetune step: ", s, flush=True)
            t = s + 1
            zt = ref_zt_chain[t] # TODO check id of ref_zt_chain[t]
            s_array = torch.full((n_samples, 1), fill_value=s, device=zt.device)
            t_array = s_array + 1
            s_array = s_array / self.T
            t_array = t_array / self.T

            # assert self.dynamics.mode == "PAT" or self.atom_type_pred, "atom type should be predicted"
            if self.dynamics.mode == "PAT" or self.atom_type_pred:
                zt[:, :, self.n_dims:] = 1 # set the atom type to 1 for PAT


            if self.dynamics.mode == "PAT" or self.atom_type_pred:
                assert False, f"use edm to finetune, current mode: {self.dynamics.mode}"
                self.sample_p_zs_given_zt(s_array, t_array, zt[:,:,0:self.n_dims], node_mask, edge_mask, context, fix_noise=fix_noise, conditional_sampling=conditional_sampling, zt_chain=finetune_zt_chain, eps_t_chain=finetune_eps_t_chain)
            else:
                self.sample_p_zs_given_zt(s_array, t_array, zt, node_mask, edge_mask, context, fix_noise=fix_noise, conditional_sampling=conditional_sampling, zt_chain=finetune_zt_chain, eps_t_chain=finetune_eps_t_chain) # 将采样结果记录在列表末端

            gamma_t = self.inflate_batch_array(self.gamma(t_array), zt).detach()
            alpha_t = self.alpha(gamma_t, zt)
            sigma_t = self.sigma(gamma_t, zt)
            # sigma_t = self.sigma(t_array, zt)

            epsilon_t = (zt - (alpha_t * z)) / sigma_t
            assert epsilon_t.shape == (n_samples, max_n_nodes, self.n_dims + 5 + self.include_charges), f"epsilon_t shape {epsilon_t.shape} should be ({n_samples}, {max_n_nodes}, {self.n_dims + 5 + self.include_charges})"
            assert ref_eps_t_chain[s].shape == (n_samples, max_n_nodes, self.n_dims + 5 + self.include_charges), f"ref_eps_t_chain[s] shape {ref_eps_t_chain[s].shape} should be ({n_samples}, {max_n_nodes}, {self.n_dims + 5 + self.include_charges})"
            # print("zt: ", zt.mean(), zt[0][0])
            # print("z: ", z.mean(), z[0][0])
            # print("epsilon_t: ", epsilon_t.mean())
            # print("ref_eps_t_chain[s]: ", ref_eps_t_chain[s].mean())

            
            phi_star = finetune_eps_t_chain[-1] # 最后一个是当前结果

            def prop_gamma(gamma):
                assert gamma.shape[0] % 2 == 0, f"gamma shape {gamma.shape} should be even"
                # calculate possibility for each two samples 
                gamma_sum = gamma[0::2] + gamma[1::2]
                for i in range(gamma.shape[0]):
                    gamma[i] = gamma[i] / gamma_sum[i//2]
                return gamma
            
            gamma_prop = prop_gamma(gamma)
            gamma_reshape = gamma_prop.view(-1, 1, 1)
            assert gamma_reshape.shape == (n_samples, 1, 1), f"gamma_reshape shape {gamma_reshape.shape} should be ({n_samples}, 1, 1)"
            RHS = phi_star
            LHS = (gamma_reshape * epsilon_t + (1 - gamma_reshape) * ref_eps_t_chain[s]).detach() # 不需要梯度下降 TODO check id of ref_eps_t_chain[s]

            # RHS = RHS[:, :, :self.n_dims]
            # LHS = LHS[:, :, :self.n_dims]

            # print("alpha_t: ", alpha_t.mean())
            # print("sigma_t: ", sigma_t.mean())

            # print("epsilon_t - ref_eps: ", (epsilon_t - ref_eps_t_chain[t-1]).shape, loss_l2(epsilon_t, ref_eps_t_chain[s]).mean())
            # print("gamma: ", gamma)

            assert LHS.requires_grad == False and RHS.requires_grad == True, f"LHS requires grad: {LHS.requires_grad}, RHS requires grad: {RHS.requires_grad}"
            loss_t = loss_l2(LHS, RHS)
            # print("phi_star - ref_eps_t_chain[s]", loss_l2(phi_star, ref_eps_t_chain[s]).mean())
            # print("another loss: ", loss_l2(phi_star - ref_eps_t_chain[s], gamma*(epsilon_t-ref_eps_t_chain[s])).mean())
            loss_t = loss_t.mean(dim=2).mean(dim=1)
            assert len(loss_t.shape) == 1 and loss_t.shape[0] == n_samples, f"loss_t shape {loss_t.shape} should be ({n_samples})"

            stb_rate = stability_mask.sum() / stability_mask.shape[0]
            # print("stb_rate: ", stb_rate)
            assert stability_mask.shape == loss_t.shape and stability_mask.shape[0] == n_samples, f"stability_mask shape {stability_mask.shape} should be equal to loss_t shape {loss_t.shape}"
            loss_t = loss_t * stability_mask
            loss_t = loss_t.mean() / stb_rate
            assert len(loss_t.shape) == 0, f"loss_t shape {loss_t.shape} should be ()"
            if gamma.mean() == 0:
                print("gamma is 0, loss_t is: ")
            print("loss_t: ", loss_t, flush=True)
            # time.sleep(10)
            loss_all += loss_t.detach()
            loss_hist[s/self.T] = loss_t.detach().item()
            # print("loss_t: ", loss_t)
            optim.zero_grad()
            loss_t.backward()
            # for name, param in self.named_parameters():
            #     assert param.requires_grad == True, f"param {param} should require grad"
            #     # print(name, param.grad.mean().item() if param.grad is not None else "None", flush=True)
            #     if param.grad is not None:
            #         param.grad.data.clamp_(-1, 1)
            if lr_dict is not None:
                for param_group in optim.param_groups:
                    print("lr: ", param_group['lr'])
                    param_group['lr'] = lr_dict[s/self.T]
                    # param_group['lr'] = 0
            optim.step()
            optim.zero_grad()
            # if wandb is not None:
            #     wandb.log({f"loss": loss_t.item()})
        return loss_all, loss_hist

    @torch.no_grad()
    def sample_dpo_chain(self, n_samples, max_n_nodes, node_mask, edge_mask, context, fix_noise=False, conditional_sampling=False, atom_type_need=False):
        """
        存储每一个t的zt与预测的噪声
        """
        self.dynamics.eval()
        zt_chain = []
        eps_t_chain = []

        assert self.atom_type_pred == 0, "atom type should not be predicted"
        z = self.sample_combined_position_feature_noise(n_samples, max_n_nodes, node_mask) # sample z_T
        zt_chain.append(z[:, :, :self.n_dims] if self.atom_type_pred else z)        

        diffusion_utils.assert_mean_zero_with_mask(z[:, :, :self.n_dims], node_mask)

        # Iteratively sample p(z_s | z_t) for t = 1, ..., T, with s = t - 1.
        
        
        # used for uni diffusion
        s_array2_org = torch.full((n_samples, 1), fill_value=-1, device=z.device)
        
        
        for s in reversed(range(0, self.T)):
            s_array = torch.full((n_samples, 1), fill_value=s, device=z.device)
            t_array = s_array + 1
            s_array = s_array / self.T
            t_array = t_array / self.T
            
            # assert self.dynamics.mode == "PAT" or self.atom_type_pred, "atom type should be predicted"
            if self.dynamics.mode == "PAT" or self.atom_type_pred:
                z[:, :, self.n_dims:] = 1 # set the atom type to 1 for PAT
            
            # TODO uni diffusion, uni generate the molecule and corespond property
            #  if uni_diffusion:
            
            if self.dynamics.mode == "PAT" or self.atom_type_pred:
                assert False, f"use edm to finetune, current mode: {self.dynamics.mode}"
                z = self.sample_p_zs_given_zt(s_array, t_array, z[:,:,0:self.n_dims], node_mask, edge_mask, context, fix_noise=fix_noise, conditional_sampling=conditional_sampling, zt_chain=zt_chain, eps_t_chain=eps_t_chain)
            else:
                z = self.sample_p_zs_given_zt(s_array, t_array, z, node_mask, edge_mask, context, fix_noise=fix_noise, conditional_sampling=conditional_sampling, zt_chain=zt_chain, eps_t_chain=eps_t_chain)


        # Finally sample p(x, h | z_0).
        if self.property_pred:   
            if self.atom_type_pred:
                z[:,:,self.n_dims:] = 1
            x, h, pred = self.sample_p_xh_given_z0(z, node_mask, edge_mask, context, fix_noise=fix_noise)#, zt_chain=zt_chain, eps_t_chain=eps_t_chain
            assert False, f"property prediction should not be used, current mode: {self.dynamics.mode}"
        else:
            if self.dynamics.mode == "PAT" or self.atom_type_pred:
                assert False, f"atom type should be predicted, current mode: {self.dynamics.mode}"
                z[:,:,self.n_dims:] = 1 # set the atom type to 1 for PAT
            x, h = self.sample_p_xh_given_z0(z, node_mask, edge_mask, context, fix_noise=fix_noise)#, zt_chain=zt_chain, eps_t_chain=eps_t_chain
        


        diffusion_utils.assert_mean_zero_with_mask(x, node_mask)

        assert len(zt_chain) == self.T + 1, "zt_chain should have length T+1"
        assert len(eps_t_chain) == self.T, "eps_t_chain should have length T"
        #倒序
        zt_chain = list(reversed(zt_chain))
        eps_t_chain = list(reversed(eps_t_chain))

        max_cog = torch.sum(x, dim=1, keepdim=True).abs().max().item()
        if max_cog > 5e-2:
            print(f'Warning cog drift with error {max_cog:.3f}. Projecting '
                  f'the positions down.')
            x = diffusion_utils.remove_mean_with_mask(x, node_mask)
            # zt_chain[0] = torch.cat([x, zt_chain[0][:, :, self.n_dims:]], dim=2)
        if self.property_pred:
            return x, h, pred, zt_chain, eps_t_chain
        return x, h, zt_chain, eps_t_chain

    @torch.no_grad()
    def sample_z_in_cond_gen(self, z, n_samples, n_nodes, node_mask, edge_mask, pseudo_context):
        t = torch.full((n_samples, 1), 0.999, device=z.device)
        _, pred = self.phi(z, t, node_mask, edge_mask=edge_mask, context=None)
        # print("loss v1: ", torch.nn.L1Loss()(pred, pseudo_context).item())
        pred = pred.unsqueeze(1)
        loss_fn = torch.nn.L1Loss(reduction='none')
        # loss = torch.nn.L1Loss()(pred, pseudo_context).item()
        assert pred.size(0) == n_samples and n_samples == 4
        print("LOSS: ", loss_fn(pred, pseudo_context))
        print("pred.shape: ", pred.shape)
        loss = loss_fn(pred, pseudo_context).sum() / pred.shape[0]
        while loss > self.condGenConfig["loss_maximum"]:
            print("Control pred loss under {}, current loss is {}".format(self.condGenConfig["loss_maximum"], loss))
            # print
            print("pred: ", pred)
            print("pseudo_context: ", pseudo_context, "\n")
            # print("loss single: ", loss)
            z = self.sample_combined_position_feature_noise(n_samples, n_nodes, node_mask)
            # print("z: ", z.shape, z[0])
            _, pred = self.phi(z, t, node_mask, edge_mask=edge_mask, context=None)
            pred = pred.unsqueeze(1)
            loss = loss_fn(pred, pseudo_context).sum() / pred.shape[0]
            print("loss: ", loss_fn(pred, pseudo_context))
        return z

    @torch.no_grad()
    def sample(self, n_samples, n_nodes, node_mask, edge_mask, context, fix_noise=False, condition_generate_x=False, annel_l=False, pseudo_context=None, conditional_sampling=False, mean=None, mad=None):
        """
        Draw samples from the generative model.
        """
        if fix_noise:
            # Noise is broadcasted over the batch axis, useful for visualizations.
            z = self.sample_combined_position_feature_noise(1, n_nodes, node_mask)
        else:
            z = self.sample_combined_position_feature_noise(n_samples, n_nodes, node_mask)
            # if self.condGenConfig is not None:
                # print("Control pred loss under {}".format(self.condGenConfig["loss_maximum"]))
                # print("node_mask: ", node_mask.sum().item())
                # print("node_mask single: ", node_mask.sum(dim=1))
                # z = self.sample_z_in_cond_gen(z, n_samples, n_nodes, node_mask, edge_mask, pseudo_context)
                


        if self.uni_diffusion:
            org_context = context
            
            if z.size(-1) == 3 + 22: # 3 coordinate + 22 atom type ==> pcqm
                context_size = 53
            else:
                context_size = 1
            
            context = utils.sample_gaussian_with_mask(
                size=(z.size(0), 1, context_size), device=node_mask.device,
                node_mask=node_mask)

        diffusion_utils.assert_mean_zero_with_mask(z[:, :, :self.n_dims], node_mask)

        # Iteratively sample p(z_s | z_t) for t = 1, ..., T, with s = t - 1.
        
        
        # used for uni diffusion
        s_array2_org = torch.full((n_samples, 1), fill_value=-1, device=z.device)
        
        zt_chain = []
        eps_chain = []
        
        for s in reversed(range(0, self.T)):
            s_array = torch.full((n_samples, 1), fill_value=s, device=z.device)
            t_array = s_array + 1
            s_array = s_array / self.T
            t_array = t_array / self.T
            
            if self.dynamics.mode == "PAT" or self.atom_type_pred:
                z[:, :, self.n_dims:] = 1 # set the atom type to 1 for PAT
            
            # TODO uni diffusion, uni generate the molecule and cooorespond property
            #  if uni_diffusion:
            if self.uni_diffusion:
                s_array2 = s_array.clone()
                # s_array2 = s_array * 1000 // 100
                # t_array2 = s_array2 + 1
                t_array2 = t_array.clone()
                
                update_context = True
                # if s_array2_org[0].item() >= 0 and s_array2_org[0,0].item() != s_array2[0,0].item(): # when s_array2 change, context need to update
                #     s_array2_org = s_array2
                #     update_context = True
                # else:
                #     if s_array2_org[0].item() < 0: # first update, context not change
                #         s_array2_org = s_array2
                #     update_context = False
                
                # s_array2 = s_array2 / self.T2
                # t_array2 = t_array2 / self.T2
                if condition_generate_x:
                    context = org_context
                    force_t2_zero = True                    
                    t_int2 = torch.ones((t_array2.size(0), 1), device=t_array2.device).float() # t_int all zero
                    s_int2 = t_int2 - 1
                    s_array2 = s_int2 / self.T
                    t_array2 = t_int2 / self.T
                    
                else:
                    force_t2_zero = False
                
                z, new_context = self.sample_p_zs_given_zt(
                    s_array, t_array, z, node_mask, edge_mask, context, fix_noise=fix_noise, yt=t_array2, ys=s_array2, force_t2_zero=force_t2_zero)
                
                if update_context:
                    context = new_context
            elif annel_l:
                z = self.sample_p_zs_given_zt_annel_lang(s_array, t_array, z, node_mask, edge_mask, context, fix_noise=fix_noise)
            elif self.dynamics.mode == "PAT" or self.atom_type_pred:
                # z = self.sample_p_zs_given_zt(s_array, t_array, z[:,:,:3], node_mask, edge_mask, context, fix_noise=fix_noise, pseudo_context=pseudo_context, mean=mean, mad=mad)
                z = self.sample_p_zs_given_zt(s_array, t_array, z[:,:,:3], node_mask, edge_mask, context, fix_noise=fix_noise, pseudo_context=pseudo_context, mean=mean, mad=mad, zt_chain=zt_chain, eps_t_chain=eps_chain)
            else:
                z = self.sample_p_zs_given_zt(s_array, t_array, z, node_mask, edge_mask, context, fix_noise=fix_noise, conditional_sampling=conditional_sampling)

        # Finally sample p(x, h | z_0).
        if self.property_pred:
            if self.atom_type_pred:
                atom_type_dim = 6 if self.include_charges else 5
                z[:,:,self.n_dims:self.n_dims+atom_type_dim] = 1
            x, h, pred = self.sample_p_xh_given_z0(z, node_mask, edge_mask, context, fix_noise=fix_noise)
        else:
            if self.dynamics.mode == "PAT" or self.atom_type_pred:
                # print("z size after padding 1", z.size())
                z[:,:,self.n_dims:] = 1 # set the atom type to 1 for PAT
            x, h = self.sample_p_xh_given_z0(z, node_mask, edge_mask, context, fix_noise=fix_noise)
        

        if self.uni_diffusion:
            # print averge mae between property_pred and context
            if org_context is not None:
                mae = torch.mean(torch.abs(context - org_context))
                print(f'property mae: {mae.item()}')

        diffusion_utils.assert_mean_zero_with_mask(x, node_mask)

        max_cog = torch.sum(x, dim=1, keepdim=True).abs().max().item()
        if max_cog > 5e-2:
            print(f'Warning cog drift with error {max_cog:.3f}. Projecting '
                  f'the positions down.')
            x = diffusion_utils.remove_mean_with_mask(x, node_mask)
        if self.property_pred:
            return x, h, pred
        return x, h

    @torch.no_grad()
    def sample_chain(self, n_samples, n_nodes, node_mask, edge_mask, context, keep_frames=None, annel_l=False):
        """
        Draw samples from the generative model, keep the intermediate states for visualization purposes.
        """
        z = self.sample_combined_position_feature_noise(n_samples, n_nodes, node_mask)
        # print("z.shape: ", z.shape, "node_mask.shape: ", node_mask.shape)
        # print("z: ", z, "node_mask: ", node_mask)
        if self.dynamics.mode == "PAT" or self.atom_type_pred:
            fix_h = torch.ones([n_samples, n_nodes, self.in_node_nf], device=z.device)
            z = torch.cat([z, fix_h], dim=2)
        
        if self.uni_diffusion and (z.size(-1) == 3 + 22): # 3 coordinate + 22 atom type ==> pcqm
            context_size = 53
            context = utils.sample_gaussian_with_mask(
                size=(z.size(0), 1, context_size), device=node_mask.device,
                node_mask=node_mask)
        
        

        diffusion_utils.assert_mean_zero_with_mask(z[:, :, :self.n_dims], node_mask)

        if keep_frames is None:
            keep_frames = self.T
        else:
            assert keep_frames <= self.T
        # print("z shape: ", z.size())
        chain = torch.zeros((keep_frames,) + z.size(), device=z.device)
        # print("chain original size", chain.size())
        # Iteratively sample p(z_s | z_t) for t = 1, ..., T, with s = t - 1.
        for s in reversed(range(0, self.T)):
            s_array = torch.full((n_samples, 1), fill_value=s, device=z.device)
            t_array = s_array + 1
            s_array = s_array / self.T
            t_array = t_array / self.T
            
            if self.uni_diffusion:
                s_array2 = s_array.clone()
                # s_array2 = s_array * 1000 // 100
                # t_array2 = s_array2 + 1
                t_array2 = t_array.clone()
                        
                z, new_context = self.sample_p_zs_given_zt(
                    s_array, t_array, z, node_mask, edge_mask, context, fix_noise=False, yt=t_array2, ys=s_array2)
                
                context = new_context
            else:
                if annel_l:
                    z = self.sample_p_zs_given_zt_annel_lang(s_array, t_array, z, node_mask, edge_mask, context, fix_noise=False)
                else:
                    if self.dynamics.mode == "PAT" or self.atom_type_pred:
                        z = self.sample_p_zs_given_zt(s_array, t_array, z[:,:,:3], node_mask, edge_mask, context)
                    else:
                        z = self.sample_p_zs_given_zt(s_array, t_array, z, node_mask, edge_mask, context)

            diffusion_utils.assert_mean_zero_with_mask(z[:, :, :self.n_dims], node_mask)

            # Write to chain tensor.
            write_index = (s * keep_frames) // self.T
            chain[write_index] = self.unnormalize_z(z, node_mask)

        if self.dynamics.mode == "PAT" or self.atom_type_pred:
            z = torch.cat([z[:,:, :self.n_dims], fix_h], dim=2)
        # Finally sample p(x, h | z_0).
        if self.property_pred:
            x, h, pred = self.sample_p_xh_given_z0(z, node_mask, edge_mask, context)
        else:
            x, h = self.sample_p_xh_given_z0(z, node_mask, edge_mask, context)

        diffusion_utils.assert_mean_zero_with_mask(x[:, :, :self.n_dims], node_mask)

        xh = torch.cat([x, h['categorical'], h['integer']], dim=2)
        chain[0] = xh  # Overwrite last frame with the resulting x and h.

        chain_flat = chain.view(n_samples * keep_frames, *z.size()[1:])

        return chain_flat

    def log_info(self):
        """
        Some info logging of the model.
        """
        gamma_0 = self.gamma(torch.zeros(1, device=self.buffer.device))
        gamma_1 = self.gamma(torch.ones(1, device=self.buffer.device))

        log_SNR_max = -gamma_0
        log_SNR_min = -gamma_1

        info = {
            'log_SNR_max': log_SNR_max.item(),
            'log_SNR_min': log_SNR_min.item()}
        print(info)

        return info

import numpy as np
import torch
import math
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from model.base import BaseModule
from model.diffusion_module import *
from einops import rearrange


# Base
class BaseModule(torch.nn.Module):
    def __init__(self):
        super(BaseModule, self).__init__()

    @property
    def nparams(self):
        num_params = 0
        for name, param in self.named_parameters():
            if param.requires_grad:
                num_params += np.prod(param.detach().cpu().numpy().shape)
        return num_params


    def relocate_input(self, x: list):
        device = next(self.parameters()).device
        for i in range(len(x)):
            if isinstance(x[i], torch.Tensor) and x[i].device != device:
                x[i] = x[i].to(device)
        return x

# Commons
def init_weights(m, mean=0.0, std=0.01):
  classname = m.__class__.__name__
  if classname.find("Conv") != -1:
    m.weight.data.normal_(mean, std)


def get_padding(kernel_size, dilation=1):
  return int((kernel_size*dilation - dilation)/2)


def convert_pad_shape(pad_shape):
  l = pad_shape[::-1]
  pad_shape = [item for sublist in l for item in sublist]
  
  return pad_shape


def intersperse(lst, item):
  result = [item] * (len(lst) * 2 + 1)
  result[1::2] = lst
  
  return result


def kl_divergence(m_p, logs_p, m_q, logs_q):
  """KL(P||Q)"""
  kl = (logs_q - logs_p) - 0.5
  kl += 0.5 * (torch.exp(2. * logs_p) + ((m_p - m_q)**2)) * torch.exp(-2. * logs_q)
  
  return kl


def rand_gumbel(shape):
  """Sample from the Gumbel distribution, protect from overflows."""
  uniform_samples = torch.rand(shape) * 0.99998 + 0.00001
  
  return -torch.log(-torch.log(uniform_samples))


def rand_gumbel_like(x):
  g = rand_gumbel(x.size()).to(dtype=x.dtype, device=x.device)
  return g


def slice_segments(x, ids_str, segment_size=4):
  ret = torch.zeros_like(x[:, :, :segment_size])
  for i in range(x.size(0)):
    idx_str = ids_str[i]
    idx_end = idx_str + segment_size
    ret[i] = x[i, :, idx_str:idx_end]
    
  return ret

def slice_segments_audio(x, ids_str, segment_size=4):
  ret = torch.zeros_like(x[:, :segment_size])
  for i in range(x.size(0)):
    idx_str = ids_str[i]
    idx_end = idx_str + segment_size
    ret[i] = x[i, idx_str:idx_end]
    
  return ret

def rand_slice_segments(x, x_lengths=None, segment_size=4):
  b, d, t = x.size()
  if x_lengths is None:
    x_lengths = t
  ids_str_max = x_lengths - segment_size + 1
  ids_str = ((torch.rand([b]).to(device=x.device) * ids_str_max).clip(0)).to(dtype=torch.long)
  ret = slice_segments(x, ids_str, segment_size)
  
  return ret, ids_str


def get_timing_signal_1d(
    length, channels, min_timescale=1.0, max_timescale=1.0e4):
  position = torch.arange(length, dtype=torch.float)
  num_timescales = channels // 2
  log_timescale_increment = (
      math.log(float(max_timescale) / float(min_timescale)) /
      (num_timescales - 1))
  inv_timescales = min_timescale * torch.exp(
      torch.arange(num_timescales, dtype=torch.float) * -log_timescale_increment)
  scaled_time = position.unsqueeze(0) * inv_timescales.unsqueeze(1)
  signal = torch.cat([torch.sin(scaled_time), torch.cos(scaled_time)], 0)
  signal = F.pad(signal, [0, 0, 0, channels % 2])
  signal = signal.view(1, channels, length)
  
  return signal


def add_timing_signal_1d(x, min_timescale=1.0, max_timescale=1.0e4):
  b, channels, length = x.size()
  signal = get_timing_signal_1d(length, channels, min_timescale, max_timescale)
  
  return x + signal.to(dtype=x.dtype, device=x.device)


def cat_timing_signal_1d(x, min_timescale=1.0, max_timescale=1.0e4, axis=1):
  b, channels, length = x.size()
  signal = get_timing_signal_1d(length, channels, min_timescale, max_timescale)
  
  return torch.cat([x, signal.to(dtype=x.dtype, device=x.device)], axis)


def subsequent_mask(length):
  mask = torch.tril(torch.ones(length, length)).unsqueeze(0).unsqueeze(0)
  
  return mask


@torch.jit.script
def fused_add_tanh_sigmoid_multiply(input_a, input_b, n_channels):
  n_channels_int = n_channels[0]
  in_act = input_a + input_b
  t_act = torch.tanh(in_act[:, :n_channels_int, :])
  s_act = torch.sigmoid(in_act[:, n_channels_int:, :])
  acts = t_act * s_act
  
  return acts


def convert_pad_shape(pad_shape):
  l = pad_shape[::-1]
  pad_shape = [item for sublist in l for item in sublist]
  
  return pad_shape


def shift_1d(x):
  x = F.pad(x, convert_pad_shape([[0, 0], [0, 0], [1, 0]]))[:, :, :-1]
  
  return x


def sequence_mask(length, max_length=None):
  if max_length is None:
    max_length = length.max()
  x = torch.arange(max_length, dtype=length.dtype, device=length.device)
  
  return x.unsqueeze(0) < length.unsqueeze(1)


def generate_path(duration, mask):
  """
  duration: [b, 1, t_x]
  mask: [b, 1, t_y, t_x]
  """
  device = duration.device
  
  b, _, t_y, t_x = mask.shape
  cum_duration = torch.cumsum(duration, -1)
  
  cum_duration_flat = cum_duration.view(b * t_x)
  path = sequence_mask(cum_duration_flat, t_y).to(mask.dtype)
  path = path.view(b, t_x, t_y)
  path = path - F.pad(path, convert_pad_shape([[0, 0], [1, 0], [0, 0]]))[:, :-1]
  path = path.unsqueeze(1).transpose(2,3) * mask
  
  return path


def clip_grad_value_(parameters, clip_value, norm_type=2):
  if isinstance(parameters, torch.Tensor):
    parameters = [parameters]
  parameters = list(filter(lambda p: p.grad is not None, parameters))
  norm_type = float(norm_type)
  if clip_value is not None:
    clip_value = float(clip_value)

  total_norm = 0
  for p in parameters:
    param_norm = p.grad.data.norm(norm_type)
    total_norm += param_norm.item() ** norm_type
    if clip_value is not None:
      p.grad.data.clamp_(min=-clip_value, max=clip_value)
  total_norm = total_norm ** (1. / norm_type)
  
  return total_norm

# Diffusion
class GradLogPEstimator(BaseModule):
    def __init__(self, dim_base, dim_cond, dim_mults=(1, 2, 4)):
        super(GradLogPEstimator, self).__init__()

        dims = [2 + dim_cond, *map(lambda m: dim_base * m, dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))

        self.time_pos_emb = SinusoidalPosEmb(dim_base)
        self.mlp = torch.nn.Sequential(torch.nn.Linear(dim_base, dim_base * 4),
                                       Mish(), torch.nn.Linear(dim_base * 4, dim_base))
        cond_total = dim_base + 256
        self.cond_block = torch.nn.Sequential(torch.nn.Linear(cond_total, 4 * dim_cond),
                                              Mish(), torch.nn.Linear(4 * dim_cond, dim_cond))

        self.downs = torch.nn.ModuleList([])
        self.ups = torch.nn.ModuleList([])

        num_resolutions = len(in_out) 

        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (num_resolutions - 1)
            self.downs.append(torch.nn.ModuleList([
                ResnetBlock(dim_in, dim_out, time_emb_dim=dim_base),
                ResnetBlock(dim_out, dim_out, time_emb_dim=dim_base),
                Residual(Rezero(LinearAttention(dim_out))),
                Downsample(dim_out) if not is_last else torch.nn.Identity()]))

        mid_dim = dims[-1]  

        self.mid_block1 = ResnetBlock(mid_dim, mid_dim, time_emb_dim=dim_base)
        self.mid_attn = Residual(Rezero(LinearAttention(mid_dim)))
        self.mid_block2 = ResnetBlock(mid_dim, mid_dim, time_emb_dim=dim_base)

        for ind, (dim_in, dim_out) in enumerate(reversed(in_out[1:])):
            self.ups.append(torch.nn.ModuleList([
                ResnetBlock(dim_out * 2, dim_in, time_emb_dim=dim_base),
                ResnetBlock(dim_in, dim_in, time_emb_dim=dim_base),
                Residual(Rezero(LinearAttention(dim_in))),
                Upsample(dim_in)]))
        self.final_block = Block(dim_base, dim_base)
        self.final_conv = torch.nn.Conv2d(dim_base, 1, 1)

    def forward(self, x, x_mask, enc_out, spk, t):
        condition = self.time_pos_emb(t) 
        t = self.mlp(condition) 

        x = torch.stack([enc_out, x], 1)
        x_mask = x_mask.unsqueeze(1)

        condition = torch.cat([condition, spk.squeeze(2)], 1) 
        condition = self.cond_block(condition).unsqueeze(-1).unsqueeze(-1)  

        condition = torch.cat(x.shape[2] * [condition], 2)  
        condition = torch.cat(x.shape[3] * [condition], 3)
        x = torch.cat([x, condition], 1)

        hiddens = []
        masks = [x_mask]

        for resnet1, resnet2, attn, downsample in self.downs:
            mask_down = masks[-1]
            x = resnet1(x, mask_down, t)
            x = resnet2(x, mask_down, t)
            x = attn(x)
            hiddens.append(x)
            x = downsample(x * mask_down)
            masks.append(mask_down[:, :, :, ::2])

        masks = masks[:-1]
        mask_mid = masks[-1]
        x = self.mid_block1(x, mask_mid, t)
        x = self.mid_attn(x)
        x = self.mid_block2(x, mask_mid, t)

        for resnet1, resnet2, attn, upsample in self.ups:
            mask_up = masks.pop()
            x = torch.cat((x, hiddens.pop()), dim=1)
            x = resnet1(x, mask_up, t)
            x = resnet2(x, mask_up, t)
            x = attn(x)
            x = upsample(x * mask_up)

        x = self.final_block(x, x_mask)
        output = self.final_conv(x * x_mask)
        return (output * x_mask).squeeze(1)

class Diffusion(BaseModule):
    def __init__(self, n_feats, dim_unet, dim_spk, beta_min, beta_max):
        super(Diffusion, self).__init__()
        self.estimator_src = GradLogPEstimator(dim_unet, dim_spk)
        self.estimator_ftr = GradLogPEstimator(dim_unet, dim_spk)

        self.n_feats = n_feats
        self.dim_unet = dim_unet
        self.dim_spk = dim_spk
        self.beta_min = beta_min
        self.beta_max = beta_max

    def get_beta(self, t):
        beta = self.beta_min + (self.beta_max - self.beta_min) * t
        return beta

    def get_gamma(self, s, t, p=1.0, use_torch=False):
        beta_integral = self.beta_min + 0.5 * (self.beta_max - self.beta_min) * (t + s)
        beta_integral *= (t - s)
        if use_torch:
            gamma = torch.exp(-0.5 * p * beta_integral).unsqueeze(-1).unsqueeze(-1)
        else:
            gamma = math.exp(-0.5 * p * beta_integral)
        return gamma

    def get_mu(self, s, t):
        a = self.get_gamma(s, t)
        b = 1.0 - self.get_gamma(0, s, p=2.0)
        c = 1.0 - self.get_gamma(0, t, p=2.0)
        return a * b / c

    def get_nu(self, s, t):
        a = self.get_gamma(0, s)
        b = 1.0 - self.get_gamma(s, t, p=2.0)
        c = 1.0 - self.get_gamma(0, t, p=2.0)
        return a * b / c

    def get_sigma(self, s, t):
        a = 1.0 - self.get_gamma(0, s, p=2.0)
        b = 1.0 - self.get_gamma(s, t, p=2.0)
        c = 1.0 - self.get_gamma(0, t, p=2.0)
        return math.sqrt(a * b / c)

    def compute_diffused_mean(self, x0, mask, src_out, ftr_out, t, use_torch=False):

        x0_weight = self.get_gamma(0, t, use_torch=use_torch)  
        mean_weight = 1.0 - x0_weight
        xt_src = x0 * x0_weight + src_out * mean_weight
        xt_ftr = x0 * x0_weight + ftr_out * mean_weight
        return xt_src * mask, xt_ftr * mask

    def forward_diffusion(self, x0, mask, src_out, ftr_out, t):
        xt_src, xt_ftr = self.compute_diffused_mean(x0, mask, src_out, ftr_out, t, use_torch=True)
        variance = 1.0 - self.get_gamma(0, t, p=2.0, use_torch=True)
        z = torch.randn(x0.shape, dtype=x0.dtype, device=x0.device, requires_grad=False)
        xt_src = xt_src + z * torch.sqrt(variance)
        xt_ftr = xt_ftr + z * torch.sqrt(variance)

        return xt_src * mask, xt_ftr * mask, z * mask

    @torch.no_grad()
    def reverse_diffusion(self, z_src, z_ftr, mask, src_out, ftr_out, spk,
                          n_timesteps, mode):
        h = 1.0 / n_timesteps
        xt_src = z_src * mask
        xt_ftr = z_ftr * mask
        for i in range(n_timesteps):
            t = 1.0 - i * h
            time = t * torch.ones(z_src.shape[0], dtype=z_src.dtype, device=z_src.device)
            beta_t = self.get_beta(t)

            if mode == 'ml':
                kappa = self.get_gamma(0, t - h) * (1.0 - self.get_gamma(t - h, t, p=2.0))
                kappa /= (self.get_gamma(0, t) * beta_t * h)
                kappa -= 1.0
                omega = self.get_nu(t - h, t) / self.get_gamma(0, t)
                omega += self.get_mu(t - h, t)
                omega -= (0.5 * beta_t * h + 1.0)
                sigma = self.get_sigma(t - h, t)

            else:
                kappa = 0.0
                omega = 0.0
                sigma = math.sqrt(beta_t * h)

            dxt_src = (src_out - xt_src) * (0.5 * beta_t * h + omega)
            dxt_ftr = (ftr_out - xt_ftr) * (0.5 * beta_t * h + omega)

            estimated_score = (self.estimator_src(xt_src, mask, src_out, spk, time) +
                               self.estimator_ftr(xt_ftr, mask, ftr_out, spk, time)) \
                              * (1.0 + kappa) * (beta_t * h)     
            dxt_src -= estimated_score
            dxt_ftr -= estimated_score
            
            sigma_n = torch.randn_like(z_src, device=z_src.device) * sigma
            dxt_src += sigma_n
            dxt_ftr += sigma_n

            xt_src = (xt_src - dxt_src) * mask
            xt_ftr = (xt_ftr - dxt_ftr) * mask

        return xt_src, xt_ftr

    @torch.no_grad()
    def forward(self, z_src, z_ftr, mask, src_out, ftr_out, spk, n_timesteps, mode):
        if mode not in ['pf', 'em', 'ml']:
            print('Inference mode must be one of [pf, em, ml]!')
            return z_src, z_ftr

        return self.reverse_diffusion(z_src, z_ftr, mask, src_out, ftr_out, spk, n_timesteps, mode)

    def loss_t(self, x0, mask, src_out, ftr_out, spk, t):
        xt_src, xt_ftr, z = self.forward_diffusion(x0, mask, src_out, ftr_out, t)

        z_estimation = self.estimator_src(xt_src, mask, src_out, spk, t)
        z_estimation += self.estimator_ftr(xt_ftr, mask, ftr_out, spk, t)

        z_estimation *= torch.sqrt(1.0 - self.get_gamma(0, t, p=2.0, use_torch=True))
        loss = torch.sum((z_estimation + z) ** 2) / (torch.sum(mask) * self.n_feats)

        return loss

    def compute_loss(self, x0, mask, src_out, ftr_out, spk, offset=1e-5):
        b = x0.shape[0]
        t = torch.rand(b, dtype=x0.dtype, device=x0.device, requires_grad=False)
        t = torch.clamp(t, offset, 1.0 - offset)

        return self.loss_t(x0, mask, src_out, ftr_out, spk, t)


# Diffusion Module
class Mish(BaseModule):
    def forward(self, x):
        return x * torch.tanh(torch.nn.functional.softplus(x))


class Upsample(BaseModule):
    def __init__(self, dim):
        super(Upsample, self).__init__()
        self.conv = torch.nn.ConvTranspose2d(dim, dim, 4, 2, 1)

    def forward(self, x):
        return self.conv(x)


class Downsample(BaseModule):
    def __init__(self, dim):
        super(Downsample, self).__init__()
        self.conv = torch.nn.Conv2d(dim, dim, 3, 2, 1)

    def forward(self, x):
        return self.conv(x)


class Rezero(BaseModule):
    def __init__(self, fn):
        super(Rezero, self).__init__()
        self.fn = fn
        self.g = torch.nn.Parameter(torch.zeros(1))

    def forward(self, x):
        return self.fn(x) * self.g


class Block(BaseModule):
    def __init__(self, dim, dim_out, groups=8):
        super(Block, self).__init__()
        self.block = torch.nn.Sequential(torch.nn.Conv2d(dim, dim_out, 3, 
                                         padding=1), torch.nn.GroupNorm(
                                         groups, dim_out), Mish())

    def forward(self, x, mask):
        output = self.block(x * mask)
        return output * mask


class ResnetBlock(BaseModule):
    def __init__(self, dim, dim_out, time_emb_dim, groups=8):
        super(ResnetBlock, self).__init__()
        self.mlp = torch.nn.Sequential(Mish(), torch.nn.Linear(time_emb_dim, 
                                                               dim_out))

        self.block1 = Block(dim, dim_out, groups=groups)
        self.block2 = Block(dim_out, dim_out, groups=groups)
        if dim != dim_out:
            self.res_conv = torch.nn.Conv2d(dim, dim_out, 1)
        else:
            self.res_conv = torch.nn.Identity()

    def forward(self, x, mask, time_emb):
        h = self.block1(x, mask)
        h += self.mlp(time_emb).unsqueeze(-1).unsqueeze(-1)
        h = self.block2(h, mask)
        output = h + self.res_conv(x * mask)
        return output


class LinearAttention(BaseModule):
    def __init__(self, dim, heads=4, dim_head=32):
        super(LinearAttention, self).__init__()
        self.heads = heads
        hidden_dim = dim_head * heads
        self.to_qkv = torch.nn.Conv2d(dim, hidden_dim * 3, 1, bias=False)
        self.to_out = torch.nn.Conv2d(hidden_dim, dim, 1)            

    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.to_qkv(x)
        q, k, v = rearrange(qkv, 'b (qkv heads c) h w -> qkv b heads c (h w)', 
                            heads = self.heads, qkv=3)            
        k = k.softmax(dim=-1)
        context = torch.einsum('bhdn,bhen->bhde', k, v)
        out = torch.einsum('bhde,bhdn->bhen', context, q)
        out = rearrange(out, 'b heads c (h w) -> b (heads c) h w', 
                        heads=self.heads, h=h, w=w)
        return self.to_out(out)


class Residual(BaseModule):
    def __init__(self, fn):
        super(Residual, self).__init__()
        self.fn = fn

    def forward(self, x, *args, **kwargs):
        output = self.fn(x, *args, **kwargs) + x
        return output


class SinusoidalPosEmb(BaseModule):
    def __init__(self, dim):
        super(SinusoidalPosEmb, self).__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device).float() * -emb)
        emb = 1000.0 * x.unsqueeze(1) * emb.unsqueeze(0)
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


class RefBlock(BaseModule):
    def __init__(self, out_dim, time_emb_dim):
        super(RefBlock, self).__init__()
        base_dim = out_dim // 4
        self.mlp1 = torch.nn.Sequential(Mish(), torch.nn.Linear(time_emb_dim, 
                                                                base_dim))
        self.mlp2 = torch.nn.Sequential(Mish(), torch.nn.Linear(time_emb_dim, 
                                                                2 * base_dim))
        self.block11 = torch.nn.Sequential(torch.nn.Conv2d(1, 2 * base_dim, 
                      3, 1, 1), torch.nn.InstanceNorm2d(2 * base_dim, affine=True),
                      torch.nn.GLU(dim=1))
        self.block12 = torch.nn.Sequential(torch.nn.Conv2d(base_dim, 2 * base_dim, 
                      3, 1, 1), torch.nn.InstanceNorm2d(2 * base_dim, affine=True),
                      torch.nn.GLU(dim=1))
        self.block21 = torch.nn.Sequential(torch.nn.Conv2d(base_dim, 4 * base_dim,
                      3, 1, 1), torch.nn.InstanceNorm2d(4 * base_dim, affine=True),
                      torch.nn.GLU(dim=1))
        self.block22 = torch.nn.Sequential(torch.nn.Conv2d(2 * base_dim, 4 * base_dim,
                      3, 1, 1), torch.nn.InstanceNorm2d(4 * base_dim, affine=True),
                      torch.nn.GLU(dim=1))
        self.block31 = torch.nn.Sequential(torch.nn.Conv2d(2 * base_dim, 8 * base_dim,
                      3, 1, 1), torch.nn.InstanceNorm2d(8 * base_dim, affine=True),
                      torch.nn.GLU(dim=1))
        self.block32 = torch.nn.Sequential(torch.nn.Conv2d(4 * base_dim, 8 * base_dim,
                      3, 1, 1), torch.nn.InstanceNorm2d(8 * base_dim, affine=True),
                      torch.nn.GLU(dim=1))
        self.final_conv = torch.nn.Conv2d(4 * base_dim, out_dim, 1)

    def forward(self, x, mask, time_emb):
        y = self.block11(x * mask)
        y = self.block12(y * mask)
        y += self.mlp1(time_emb).unsqueeze(-1).unsqueeze(-1)
        y = self.block21(y * mask)
        y = self.block22(y * mask)
        y += self.mlp2(time_emb).unsqueeze(-1).unsqueeze(-1)
        y = self.block31(y * mask)
        y = self.block32(y * mask)
        y = self.final_conv(y * mask)
        return (y * mask).sum((2, 3)) / (mask.sum((2, 3)) * x.shape[2])


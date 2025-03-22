import time
import torch
from torchinfo import summary
from model.vc_dddm_mixup import DDDM, SynthesizerTrn
from model.diffusion import Diffusion
from utils import get_hparams_from_file
from model_f0_vqvae import Quantizer

config_path = "ckpt/config.json"
hps = get_hparams_from_file(config_path)

model = DDDM(
    n_feats=hps.data.n_mel_channels,
    spk_dim=hps.diffusion.spk_dim,
    dec_dim=hps.diffusion.dec_dim,
    beta_min=hps.diffusion.beta_min,
    beta_max=hps.diffusion.beta_max,
    hps=hps
)

model = model.to("cpu")

batch_size = 32
mel_channels = hps.data.n_mel_channels
max_length = hps.train.segment_size // hps.data.hop_length 

input_shapes = {
    "x": (batch_size, mel_channels, max_length),
    "w2v_x": (batch_size, 1024, max_length), 
    "f0_x": (batch_size, max_length),
    "x_lengths": (batch_size,)
}

x = torch.randn(input_shapes["x"])
w2v_x = torch.randn(input_shapes["w2v_x"])
f0_x = torch.randint(0, hps.f0_vq_params.l_bins, input_shapes["f0_x"]) 
x_lengths = torch.randint(1, max_length, input_shapes["x_lengths"])

n_timesteps = 6  

start_time = time.time()

summary(model, input_data=(x, w2v_x, f0_x, x_lengths, n_timesteps), device="cpu")

# End timing
end_time = time.time()

# Calculate and print elapsed time
elapsed_time = end_time - start_time
print(f"Running time: {elapsed_time:.4f} seconds")

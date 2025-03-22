import torch
from torchinfo import summary
from model.vc_dddm_mixup import DDDM, SynthesizerTrn
from model.diffusion import Diffusion
from utils import get_hparams_from_file
from model_f0_vqvae import Quantizer
import torch
from torchinfo import summary
from model.vc_dddm_mixup import SynthesizerTrn
from model.diffusion import Diffusion
from utils import get_hparams_from_file

# Load configuration
config_path = "ckpt/config.json"
hps = get_hparams_from_file(config_path)

# Initialize the Encoder (SynthesizerTrn)
encoder = SynthesizerTrn(
    spec_channels=hps.data.n_mel_channels,
    segment_size=hps.train.segment_size // hps.data.hop_length,
    inter_channels=hps.model.inter_channels,
    hidden_channels=hps.model.hidden_channels,
    filter_channels=hps.model.filter_channels,
    n_heads=hps.model.n_heads,
    n_layers=hps.model.n_layers,
    kernel_size=hps.model.kernel_size,
    p_dropout=hps.model.p_dropout,
    resblock=hps.model.resblock,
    resblock_kernel_sizes=hps.model.resblock_kernel_sizes,
    resblock_dilation_sizes=hps.model.resblock_dilation_sizes,
    upsample_rates=hps.model.upsample_rates,
    upsample_initial_channel=hps.model.upsample_initial_channel,
    upsample_kernel_sizes=hps.model.upsample_kernel_sizes,
    encoder_hidden_size=hps.model.encoder_hidden_size
).to("cpu")


# Create dummy inputs based on the expected shapes
batch_size = 2
mel_channels = hps.data.n_mel_channels
max_length = hps.train.segment_size // hps.data.hop_length
w2v_dim = 1024  # Assuming Wav2Vec2 output dimension
f0_bins = hps.f0_vq_params.l_bins

# Dummy inputs for the Encoder
x = torch.randn(batch_size, mel_channels, max_length)  # Mel spectrogram
w2v_x = torch.randn(batch_size, w2v_dim, max_length)   # Wav2Vec2 features
f0_x = torch.randint(0, f0_bins, (batch_size, max_length))  # F0 codes
x_lengths = torch.randint(1, max_length, (batch_size,))  # Lengths

# Dummy inputs for the Decoder
z_src = torch.randn(batch_size, mel_channels, max_length)  # Source latent
z_ftr = torch.randn(batch_size, mel_channels, max_length)  # Feature latent
x_mask = torch.ones(batch_size, 1, max_length).to(torch.float32)  # Mask
src_new = torch.randn(batch_size, mel_channels, max_length)  # New source features
ftr_new = torch.randn(batch_size, mel_channels, max_length)  # New feature features

# Ensure the shape of spk is correct
spk = torch.randn(batch_size, hps.diffusion.spk_dim)  # Speaker embedding
print(f"Shape of spk: {spk.shape}")  # Debugging: Check the shape of spk


# Get summary for the Encoder
print("Summary for Encoder (SynthesizerTrn):")
summary(encoder, input_data=(w2v_x, f0_x, x, x_lengths), device="cpu")

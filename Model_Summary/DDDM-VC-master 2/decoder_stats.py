import torch
from fvcore.nn import FlopCountAnalysis

# Hardcode the hyperparameters (replace these with your actual values)
hparams = {
    "data": {
        "n_mel_channels": 80,  # Example value, replace with your actual value
    },
    "diffusion": {
        "dec_dim": 128,        # Example value, replace with your actual value
        "spk_dim": 128,        # Example value, replace with your actual value
        "beta_min": 0.05,      # Example value, replace with your actual value
        "beta_max": 20.0,      # Example value, replace with your actual value
    }
}

# Initialize the Diffusion module
from model.diffusion import Diffusion
diffusion_model = Diffusion(
    n_feats=hparams["data"]["n_mel_channels"],  # Number of features (mel channels)
    dim_unet=hparams["diffusion"]["dec_dim"],   # Dimension of the UNet
    dim_spk=hparams["diffusion"]["spk_dim"],    # Dimension of the speaker embedding
    beta_min=hparams["diffusion"]["beta_min"],  # Minimum beta value for diffusion
    beta_max=hparams["diffusion"]["beta_max"]   # Maximum beta value for diffusion
)

# Function to calculate size in MB
def get_size_mb(tensor):
    return tensor.element_size() * tensor.numel() / (1024 ** 2)

# Function to calculate total parameters
def get_total_params(model):
    return sum(p.numel() for p in model.parameters())


# Input tensor (example)
batch_size = 1
n_mel_channels = hparams["data"]["n_mel_channels"]
seq_length = 100  # Example sequence length
dim_unet = hparams["diffusion"]["dec_dim"]
dim_spk = hparams["diffusion"]["spk_dim"]

input_tensor = torch.randn(batch_size, n_mel_channels, seq_length).to(next(diffusion_model.parameters()).device)

# Generate dummy tensors for missing arguments
z_ftr = torch.randn(batch_size, dim_unet, seq_length).to(input_tensor.device)
mask = torch.ones(batch_size, seq_length).to(input_tensor.device)  # Assuming mask is a binary tensor
src_out = torch.randn(batch_size, dim_unet, seq_length).to(input_tensor.device)
ftr_out = torch.randn(batch_size, dim_unet, seq_length).to(input_tensor.device)
spk = torch.randn(batch_size, dim_spk).to(input_tensor.device)  # Speaker embedding
n_timesteps = torch.tensor(50).to(input_tensor.device)  # Example timestep value
mode = "ml"  # Mode string, passed as an argument

# Compute FLOPs using fvcore
flops = FlopCountAnalysis(diffusion_model, (input_tensor, z_ftr, mask, src_out, ftr_out, spk, n_timesteps, mode))

# Calculate metrics
total_params = get_total_params(diffusion_model)
input_size_mb = get_size_mb(input_tensor)

# Params size in MB
params_size_mb = sum(get_size_mb(p) for p in diffusion_model.parameters())

# Forward/Backward Pass Size (approximation)
# Assuming the forward/backward activations are approximately 2x the input size
forward_backward_size_mb = 2 * input_size_mb

# Estimated Total Size (approximation)
estimated_total_size_mb = input_size_mb + forward_backward_size_mb + params_size_mb

# Print the metrics
print("Summary of the Diffusion Module:")
print(f"Total Parameters: {total_params}")
print(f"Input Size (MB): {input_size_mb:.2f} MB")
print(f"Forward/Backward Pass Size (MB): {forward_backward_size_mb:.2f} MB")
print(f"Params Size (MB): {params_size_mb:.2f} MB")
print(f"Estimated Total Size (MB): {estimated_total_size_mb:.2f} MB")


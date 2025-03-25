import time
import torch
import torch.nn as nn
from torchinfo import summary
from models import build_model, load_F0_models, load_ASR_models

class Config:
    def __init__(self):
        self.n_mels = 80
        self.hidden_dim = 512
        self.style_dim = 128
        self.n_layer = 3
        self.n_token = 178
        self.dim_in = 64
        self.max_conv_dim = 512
        self.n_domain = 108

# Model Construction
config = Config()
model = build_model(config, text_aligner=load_ASR_models("Utils/ASR/epoch_00080.pth", "Utils/ASR/config.yml"),
                    pitch_extractor=load_F0_models("Utils/JDC/bst.t7"))

# Mel: (1, 80, 192) (batch_size=1, n_mels=80, time_steps=192)
# Assume text length (1, 100) (batch_size=1, seq_len=100)
# Mel: (1, 80, 192) (batch_size=1, n_mels=80, time_steps=192)
# Vector: (1, 128) (batch_size=1, style_dim=128)
input_mel = torch.randn(1, 80, 192) 
input_text = torch.randint(0, config.n_token, (1, 100)) 
ref_mel = torch.randn(1, 80, 192) 
style_vector = torch.randn(1, config.style_dim) 

# Munch
for key, module in model.items():
    if isinstance(module, nn.Module):
        print(f"Summary for {key}:")

        start_time = time.time()
        if key == "mel_encoder":
            summary(module, input_data=[input_mel], col_names=["input_size", "output_size", "num_params", "kernel_size", "mult_adds"])
        elif key == "decoder":
            # Adjust shape
            asr = torch.randn(1, config.hidden_dim, 96)  # ASR (Shape: [batch_size, hidden_dim, 96])
            F0 = torch.randn(1, 1, 96)  # F0 (Shape: [batch_size, 1, 96])
            N = torch.randn(1, 1, 96)  # (Shape: [batch_size, 1, 96])
            style_vector = torch.randn(1, config.style_dim)  # Example: style_vector (Shape: [batch_size, style_dim])

            # Squeeze F0 and N along axis 1 to remove the second dimension, if needed
            F0_squeezed = F0.squeeze(1)  # Shape: [batch_size, 96]
            N_squeezed = N.squeeze(1)  # Shape: [batch_size, 96]

            # Check the input data shape before summary
            print("asr shape:", asr.shape)  # (1, config.hidden_dim, 96)
            print("F0 shape (squeezed):", F0_squeezed.shape)  # (1, 96)
            print("N shape (squeezed):", N_squeezed.shape)  # (1, 96)
            print("style_vector shape:", style_vector.shape)  # (1, config.style_dim)

            # Now run summary
            summary(module, input_data=[asr, F0_squeezed, N_squeezed, style_vector],
                    col_names=["input_size", "output_size", "num_params", "kernel_size", "mult_adds"])

        # Assuming your model has a 'text_encoder' module
        elif key == "text_encoder":

            # Create input data
            input_text = torch.randint(0, config.n_token, (1, 100))  # Random text input (batch_size=1, seq_len=100)
            input_lengths = torch.tensor([100])  # All sequences have length 100
            mask = torch.zeros(1, 100).bool()  # No masking applied (all False)
            # Print the shapes to confirm the expected input format
            print("input_text shape:", input_text.shape)  # (1, 100)
            print("input_lengths shape:", input_lengths.shape)  # (1,)
            print("mask shape:", mask.shape)  # (1, 100)
            # Now run summary for text_encoder
            summary(module, input_data=[input_text, input_lengths, mask],
                    col_names=["input_size", "output_size", "num_params", "kernel_size", "mult_adds"])

        elif key == "style_encoder":

            summary(module, input_data=[input_mel.unsqueeze(1)], col_names=["input_size", "output_size", "num_params", "kernel_size", "mult_adds"])
        elif key == "discriminator":
            summary(module, input_data=[input_mel.unsqueeze(1), torch.LongTensor([0])], col_names=["input_size", "output_size", "num_params", "kernel_size", "mult_adds"])
        elif key == "linear_proj":
            summary(module, input_data=[torch.randn(1, config.hidden_dim, 96)], col_names=["input_size", "output_size", "num_params", "kernel_size", "mult_adds"])
        elif key == "text_aligner":
            summary(module, input_data=[input_mel, torch.ones(1, 96).to(torch.bool), input_text], col_names=["input_size", "output_size", "num_params", "kernel_size", "mult_adds"])
        elif key == "pitch_extractor":
            summary(module, input_data=[input_mel.unsqueeze(1)], col_names=["input_size", "output_size", "num_params", "kernel_size", "mult_adds"])
        
        end_time = time.time()  
        elapsed_time = end_time - start_time 
        print(f"Running time for {key}: {elapsed_time:.4f} seconds\n")

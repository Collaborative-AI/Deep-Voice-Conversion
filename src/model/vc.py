"""MAIN-VC model
    Modified from: https://github.com/jjery2243542/adaptive_voice_conversion
    Compare to v0, the conv_bank of ContentEncoder of AdaIN-VC is retained.
    (while APC in v0)
"""

import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
import time

# Add path to external modules if necessary
sys.path.append("..")
from models.tools import get_act_func, pad_layer, pixel_shuffle_1d, upsample, adaIn


class SpeakerEncoder(nn.Module):
    def __init__(
        self,
        c_in,
        c_h,
        c_out,
        kernel_size,
        c_bank,
        n_conv_blocks,
        n_dense_blocks,
        subsample,
        act,
        dropout_rate,
    ):
        super(SpeakerEncoder, self).__init__()

        self.c_in = c_in
        self.c_h = c_h
        self.c_out = c_out
        self.kernel_size = kernel_size
        self.c_bank = c_bank
        self.n_conv_blocks = n_conv_blocks
        self.n_dense_blocks = n_dense_blocks
        self.subsample = subsample
        self.act = get_act_func(act)

        # Build speaker encoder
        self.APC_module = nn.ModuleList(
            [
                nn.Conv1d(
                    c_in,
                    c_bank,
                    kernel_size=3,
                    padding=1,
                    dilation=1,
                    padding_mode="reflect",
                ),
                nn.Conv1d(
                    c_in,
                    c_bank,
                    kernel_size=3,
                    padding=2,
                    dilation=2,
                    padding_mode="reflect",
                ),
                nn.Conv1d(
                    c_in,
                    c_bank,
                    kernel_size=3,
                    padding=4,
                    dilation=4,
                    padding_mode="reflect",
                ),
                nn.Conv1d(
                    c_in,
                    c_bank,
                    kernel_size=3,
                    padding=6,
                    dilation=6,
                    padding_mode="reflect",
                ),
                nn.Conv1d(
                    c_in,
                    c_bank,
                    kernel_size=3,
                    padding=8,
                    dilation=8,
                    padding_mode="reflect",
                ),
            ]
        )

        in_channels = self.c_in + self.c_bank * 5
        self.in_conv_layer = nn.Conv1d(in_channels, c_h, kernel_size=1)

        self.first_conv_layers = nn.ModuleList(
            [nn.Conv1d(c_h, c_h, kernel_size=kernel_size) for _ in range(n_conv_blocks)]
        )

        self.second_conv_layers = nn.ModuleList(
            [
                nn.Conv1d(c_h, c_h, kernel_size=kernel_size, stride=sub)
                for sub, _ in zip(subsample, range(n_conv_blocks))
            ]
        )

        self.pooling_layer = nn.AdaptiveAvgPool1d(1)

        self.first_dense_layers = nn.ModuleList(
            [nn.Linear(c_h, c_h) for _ in range(n_dense_blocks)]
        )
        self.second_dense_layers = nn.ModuleList(
            [nn.Linear(c_h, c_h) for _ in range(n_dense_blocks)]
        )

        self.output_layer = nn.Linear(c_h, c_out)
        self.dropout_layer = nn.Dropout(p=dropout_rate)

    def conv_blocks(self, inData):
        outData = inData
        for l in range(self.n_conv_blocks):
            y = pad_layer(outData, self.first_conv_layers[l])
            y = self.act(y)
            y = self.dropout_layer(y)
            y = pad_layer(y, self.second_conv_layers[l])
            y = self.act(y)
            y = self.dropout_layer(y)
            if self.subsample[l] > 1:
                outData = F.avg_pool1d(
                    outData, kernel_size=self.subsample[l], ceil_mode=True
                )
            outData = y + outData
        return outData

    def dense_blocks(self, inp):
        out = inp
        for l in range(self.n_dense_blocks):
            y = self.first_dense_layers[l](out)
            y = self.act(y)
            y = self.dropout_layer(y)
            y = self.second_dense_layers[l](y)
            y = self.act(y)
            y = self.dropout_layer(y)
            out = y + out
        return out

    def APC_forward(self, inp, act):
        out_list = []
        for layer in self.APC_module:
            out_list.append(act(layer(inp)))
        outData = torch.cat(out_list + [inp], dim=1)
        return outData

    def forward(self, x):
        # APC
        out = self.APC_forward(x, act=self.act)
        # Dimension reduction
        out = pad_layer(out, self.in_conv_layer)
        out = self.act(out)
        # Conv blocks
        out = self.conv_blocks(out)
        # Avg pooling
        out = self.pooling_layer(out).squeeze(2)
        # Dense blocks
        out = self.dense_blocks(out)
        out = self.output_layer(out)
        return out


class ContentEncoder(nn.Module):
    def __init__(
        self,
        c_in,
        c_h,
        c_out,
        kernel_size,
        c_bank,
        n_conv_blocks,
        subsample,
        act,
        dropout_rate,
    ):
        super(ContentEncoder, self).__init__()
        self.c_in = c_in
        self.c_h = c_h
        self.c_bank = c_bank
        self.n_conv_blocks = n_conv_blocks
        self.subsample = subsample
        # Hard coding for testing
        self.bank_scale = 2
        self.bank_size = 9
        self.act = get_act_func(act)

        # Build content encoder
        self.conv_bank = nn.ModuleList(
            [
                nn.Conv1d(c_in, c_bank, kernel_size=k)
                for k in range(self.bank_scale, self.bank_size + 1, self.bank_scale)
            ]
        )
        in_channels = self.c_bank * (self.bank_size // self.bank_scale) + c_in
        self.in_conv_layer = nn.Conv1d(in_channels, c_h, kernel_size=1)
        self.first_conv_layers = nn.ModuleList(
            [nn.Conv1d(c_h, c_h, kernel_size=kernel_size) for _ in range(n_conv_blocks)]
        )
        self.second_conv_layers = nn.ModuleList(
            [
                nn.Conv1d(c_h, c_h, kernel_size=kernel_size, stride=sub)
                for sub, _ in zip(subsample, range(n_conv_blocks))
            ]
        )
        self.norm_layer = nn.InstanceNorm1d(c_h, affine=False)  # IN
        self.mean_layer = nn.Conv1d(c_h, c_out, kernel_size=1)
        self.std_layer = nn.Conv1d(c_h, c_out, kernel_size=1)
        self.dropout_layer = nn.Dropout(p=dropout_rate)

    def conv_bank_forward(self, x, act, pad_type="reflect"):
        outs = []
        for layer in self.conv_bank:
            out = act(pad_layer(x, layer, pad_type))
            outs.append(out)
        out = torch.cat(outs + [x], dim=1)
        return out

    def forward(self, inData):
        outData = self.conv_bank_forward(inData, act=self.act)
        outData = pad_layer(outData, self.in_conv_layer)
        outData = self.norm_layer(outData)
        outData = self.act(outData)
        outData = self.dropout_layer(outData)
        for l in range(self.n_conv_blocks):
            y = pad_layer(outData, self.first_conv_layers[l])
            y = self.norm_layer(y)
            y = self.act(y)
            y = self.dropout_layer(y)
            y = pad_layer(y, self.second_conv_layers[l])
            y = self.norm_layer(y)
            y = self.act(y)
            y = self.dropout_layer(y)
            if self.subsample[l] > 1:
                outData = F.avg_pool1d(
                    outData, kernel_size=self.subsample[l], ceil_mode=True
                )
            outData = y + outData

        mu = pad_layer(outData, self.mean_layer)
        sigma = pad_layer(outData, self.std_layer)
        return mu, sigma


class Decoder(nn.Module):
    def __init__(
        self,
        c_in,
        c_cond,
        c_h,
        c_out,
        kernel_size,
        n_conv_blocks,
        upsample,
        act,
        sn,
        dropout_rate,
    ):
        super(Decoder, self).__init__()
        self.n_conv_blocks = n_conv_blocks
        self.upsample = upsample
        self.act = get_act_func(act)
        f = nn.utils.spectral_norm if sn else lambda x: x
        self.in_conv_layer = f(nn.Conv1d(c_in, c_h, kernel_size=1))
        self.first_conv_layers = nn.ModuleList(
            [
                f(nn.Conv1d(c_h, c_h, kernel_size=kernel_size))
                for _ in range(n_conv_blocks)
            ]
        )
        self.second_conv_layers = nn.ModuleList(
            [
                f(nn.Conv1d(c_h, c_h * up, kernel_size=kernel_size))
                for _, up in zip(range(n_conv_blocks), self.upsample)
            ]
        )
        self.norm_layer = nn.InstanceNorm1d(c_h, affine=False)
        self.conv_affine_layers = nn.ModuleList(
            [f(nn.Linear(c_cond, c_h * 2)) for _ in range(n_conv_blocks * 2)]
        )
        self.output_layer = nn.Conv1d(c_h, c_out, kernel_size=1)
        self.dropout_layer = nn.Dropout(p=dropout_rate)

    def upsample_block(self, x):
        for i in range(self.n_conv_blocks):
            y = pad_layer(x, self.first_conv_layers[i])
            y = self.norm_layer(y)
            y = self.act(y)
            y = self.dropout_layer(y)
            y = pad_layer(y, self.second_conv_layers[i])
            y = self.norm_layer(y)
            y = self.act(y)
            y = self.dropout_layer(y)
            x = y + x
        return x

    def forward(self, z, c):
        # Apply affine transformation
        a = self.conv_affine_layers[0](c).unsqueeze(2)
        z = z + a[:, : z.shape[1], :]
        for i in range(self.n_conv_blocks):
            z = self.upsample_block(z)
            a = self.conv_affine_layers[i * 2 + 1](c).unsqueeze(2)
            z = z + a[:, : z.shape[1], :]
        out = pad_layer(z, self.output_layer)
        return out


class MAINVC(nn.Module):
    def __init__(self, speaker_encoder, content_encoder, decoder):
        super(MAINVC, self).__init__()
        self.speaker_encoder = speaker_encoder
        self.content_encoder = content_encoder
        self.decoder = decoder

    def forward(self, x, y):
        cond = self.speaker_encoder(x)  # Speaker encoding
        mu, sigma = self.content_encoder(y)  # Content encoding
        dec = self.decoder(mu, cond)  # Decode
        return dec


# Load configuration
with open("config.yaml") as f:
    config = yaml.load(f, Loader=yaml.FullLoader)

# Initialize models
speaker_encoder = SpeakerEncoder(**config["SpeakerEncoder"])
content_encoder = ContentEncoder(**config["ContentEncoder"])
decoder = Decoder(**config["Decoder"])
main_vc = MAINVC(speaker_encoder, content_encoder, decoder)

# Generate random input data for testing
x = torch.randn(1, 80, 128)  # Random tensor for content input
y = torch.randn(1, 80, 128)  # Random tensor for speaker input

# Inference time test
start_time = time.time()

dec = main_vc(x, y)  # Forward pass

end_time = time.time()

print(f"Inference time cost: {end_time - start_time:.4f} seconds")
print(f"Converted mel shape: {dec.shape}")



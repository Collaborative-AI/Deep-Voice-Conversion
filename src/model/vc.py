"""MAIN-VC model
    Modified from: https://github.com/jjery2243542/adaptive_voice_conversion
    Compare to v0, the conv_bank of ContentEncoder of AdaIN-VC is retained.
    (while APC in v0)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class SpeakerEncoder(nn.Module):
    def __init__(self, c_in, c_h, c_out, kernel_size, c_bank, n_conv_blocks, n_dense_blocks, subsample, act, dropout_rate):
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

        # APC (autoregressive predictive coding) module
        self.APC_module = nn.ModuleList([
            nn.Conv1d(c_in, c_bank, kernel_size=3, padding=d, dilation=d, padding_mode="reflect")
            for d in [1, 2, 4, 6, 8]
        ])

        # Input layer (after concatenating APC)
        self.in_conv_layer = nn.Conv1d(c_in + c_bank * 5, c_h, kernel_size=1)

        # Conv block layers
        self.first_conv_layers = nn.ModuleList([
            nn.Conv1d(c_h, c_h, kernel_size=kernel_size) for _ in range(n_conv_blocks)
        ])
        self.second_conv_layers = nn.ModuleList([
            nn.Conv1d(c_h, c_h, kernel_size=kernel_size, stride=sub) for sub in subsample
        ])

        # Pooling and dense layers
        self.pooling_layer = nn.AdaptiveAvgPool1d(1)
        self.first_dense_layers = nn.ModuleList([nn.Linear(c_h, c_h) for _ in range(n_dense_blocks)])
        self.second_dense_layers = nn.ModuleList([nn.Linear(c_h, c_h) for _ in range(n_dense_blocks)])

        self.output_layer = nn.Linear(c_h, c_out)
        self.dropout_layer = nn.Dropout(p=dropout_rate)

    def APC_forward(self, inp):
        out_list = [self.act(layer(inp)) for layer in self.APC_module]
        return torch.cat(out_list + [inp], dim=1)

    def forward(self, x):
        # APC forward pass
        out = self.APC_forward(x)

        # Dimension reduction
        out = self.act(pad_layer(out, self.in_conv_layer))

        # Conv blocks
        for l in range(self.n_conv_blocks):
            y = self.act(pad_layer(out, self.first_conv_layers[l]))
            y = self.dropout_layer(y)
            y = self.act(pad_layer(y, self.second_conv_layers[l]))
            y = self.dropout_layer(y)
            if self.subsample[l] > 1:
                out = F.avg_pool1d(out, kernel_size=self.subsample[l], ceil_mode=True)
            out = y + out

        # Pooling and dense blocks
        out = self.pooling_layer(out).squeeze(2)
        for l in range(self.n_dense_blocks):
            y = self.act(self.first_dense_layers[l](out))
            y = self.dropout_layer(y)
            y = self.act(self.second_dense_layers[l](y))
            y = self.dropout_layer(y)
            out = y + out

        return self.output_layer(out)

class ContentEncoder(nn.Module):
    def __init__(self, c_in, c_h, c_out, kernel_size, c_bank, n_conv_blocks, subsample, act, dropout_rate):
        super(ContentEncoder, self).__init__()

        self.c_in = c_in
        self.c_h = c_h
        self.c_bank = c_bank
        self.n_conv_blocks = n_conv_blocks
        self.subsample = subsample
        self.act = get_act_func(act)

        # Conv bank
        self.conv_bank = nn.ModuleList([
            nn.Conv1d(c_in, c_bank, kernel_size=k) for k in range(2, 10, 2)
        ])
        in_channels = self.c_bank * 4 + c_in
        self.in_conv_layer = nn.Conv1d(in_channels, c_h, kernel_size=1)

        # Conv layers
        self.first_conv_layers = nn.ModuleList([
            nn.Conv1d(c_h, c_h, kernel_size=kernel_size) for _ in range(n_conv_blocks)
        ])
        self.second_conv_layers = nn.ModuleList([
            nn.Conv1d(c_h, c_h, kernel_size=kernel_size, stride=sub) for sub in subsample
        ])

        # Normalization and output
        self.norm_layer = nn.InstanceNorm1d(c_h, affine=False)
        self.mean_layer = nn.Conv1d(c_h, c_out, kernel_size=1)
        self.std_layer = nn.Conv1d(c_h, c_out, kernel_size=1)
        self.dropout_layer = nn.Dropout(p=dropout_rate)

    def forward(self, inData):
        # Conv bank forward
        outs = [self.act(pad_layer(inData, layer)) for layer in self.conv_bank]
        outData = torch.cat(outs + [inData], dim=1)

        # Dimension reduction
        outData = self.norm_layer(self.act(pad_layer(outData, self.in_conv_layer)))
        outData = self.dropout_layer(outData)

        # Conv blocks
        for l in range(self.n_conv_blocks):
            y = self.act(pad_layer(outData, self.first_conv_layers[l]))
            y = self.norm_layer(y)
            y = self.dropout_layer(y)
            y = self.act(pad_layer(y, self.second_conv_layers[l]))
            y = self.norm_layer(y)
            y = self.dropout_layer(y)
            if self.subsample[l] > 1:
                outData = F.avg_pool1d(outData, kernel_size=self.subsample[l], ceil_mode=True)
            outData = y + outData

        mu = pad_layer(outData, self.mean_layer)
        sigma = pad_layer(outData, self.std_layer)
        return mu, sigma

class Decoder(nn.Module):
    def __init__(self, c_in, c_cond, c_h, c_out, kernel_size, n_conv_blocks, upsample, act, sn, dropout_rate):
        super(Decoder, self).__init__()

        self.n_conv_blocks = n_conv_blocks
        self.upsample = upsample
        self.act = get_act_func(act)
        f = nn.utils.spectral_norm if sn else lambda x: x
        self.in_conv_layer = f(nn.Conv1d(c_in, c_h, kernel_size=1))

        self.first_conv_layers = nn.ModuleList([
            f(nn.Conv1d(c_h, c_h, kernel_size=kernel_size)) for _ in range(n_conv_blocks)
        ])
        self.second_conv_layers = nn.ModuleList([
            f(nn.Conv1d(c_h, c_h * up, kernel_size=kernel_size)) for up in upsample
        ])
        self.norm_layer = nn.InstanceNorm1d(c_h, affine=False)
        self.conv_affine_layers = nn.ModuleList([
            f(nn.Linear(c_cond, c_h * 2)) for _ in range(n_conv_blocks * 2)
        ])
        self.out_conv_layer = f(nn.Conv1d(c_h, c_out, kernel_size=1))
        self.dropout_layer = nn.Dropout(p=dropout_rate)

    def forward(self, z, cond):
        # Input conv
        out = self.norm_layer(self.act(pad_layer(z, self.in_conv_layer)))
        out = self.dropout_layer(out)

        # Conv blocks with AdaIN and upsampling
        for l in range(self.n_conv_blocks):
            y = self.act(pad_layer(out, self.first_conv_layers[l]))
            y = self.dropout_layer(y)
            y = self.apply_adain(y, cond, l)
            y = self.act(pad_layer(y, self.second_conv_layers[l]))
            y = self.dropout_layer(y)
            y = F.pixel_shuffle(y, self.upsample[l])
            y = self.apply_adain(y, cond, l + self.n_conv_blocks)
            out = y + F.interpolate(out, scale_factor=self.upsample[l], mode="nearest")

        return pad_layer(out, self.out_conv_layer)

    def apply_adain(self, x, cond, l):
        mean, std = cond.chunk(2, dim=1)
        mean = self.conv_affine_layers[l](mean)
        std = self.conv_affine_layers[l](std)
        return (x - x.mean(dim=-1, keepdim=True)) / (x.std(dim=-1, keepdim=True) + 1e-8) * std.unsqueeze(2) + mean.unsqueeze(2)

class MAINVC(nn.Module):
    def __init__(self, config):
        super(MAINVC, self).__init__()

        self.speaker_encoder = SpeakerEncoder(
            config["spk"]["c_in"], config["spk"]["c_h"], config["spk"]["c_out"],
            config["spk"]["kernel_size"], config["spk"]["c_bank"], config["spk"]["n_conv_blocks"],
            config["spk"]["n_dense_blocks"], config["spk"]["subsample"], config["spk"]["act"], config["spk"]["dropout_rate"]
        )
        self.content_encoder = ContentEncoder(
            config["cnt"]["c_in"], config["cnt"]["c_h"], config["cnt"]["c_out"],
            config["cnt"]["kernel_size"], config["cnt"]["c_bank"], config["cnt"]["n_conv_blocks"],
            config["cnt"]["subsample"], config["cnt"]["act"], config["cnt"]["dropout_rate"]
        )
        self.decoder = Decoder(
            config["dec"]["c_in"], config["dec"]["c_cond"], config["dec"]["c_h"], config["dec"]["c_out"],
            config["dec"]["kernel_size"], config["dec"]["n_conv_blocks"], config["dec"]["upsample"],
            config["dec"]["act"], config["dec"]["sn"], config["dec"]["dropout_rate"]
        )

    def forward(self, x, x_cond):
        z_mu, z_sigma = self.content_encoder(x)
        z = z_mu + torch.randn_like(z_mu) * z_sigma
        s = self.speaker_encoder(x_cond)
        y = self.decoder(z, s)
        return y


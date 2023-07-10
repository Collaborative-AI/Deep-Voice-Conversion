import torch
import torch.nn as nn
import numpy as np
from models.text.symbols import symbols
import models.Constants as Constants
from models.Modules import Mish, LinearNorm, ConvNorm, Conv1dGLU, \
                    MultiHeadAttention, StyleAdaptiveLayerNorm, get_sinusoid_encoding_table
# from models.VarianceAdaptor import VarianceAdaptor
# from models.Loss import StyleSpeechLoss

def get_mask_from_lengths(input, max_len=None):
    lengths = torch.count_nonzero(input, dim=-1)
    batch_size = lengths.shape[0]
    if max_len is None:
        max_len = torch.max(lengths).item()

    ids = torch.arange(0, max_len).unsqueeze(0).expand(batch_size, -1).cuda()
    mask = (ids >= lengths.unsqueeze(1).expand(-1, max_len))
    return mask


class CrossModalDecoder(nn.Module):
    ''' StyleSpeech '''
    def __init__(self, config):
        super(CrossModalDecoder, self).__init__()
        self.cross_modal_shift = CrossModalShift(config)
        # self.variance_adaptor = VarianceAdaptor(config)
        self.decoder = Decoder(**config.decoder)
        
    # def parse_batch(self, batch):
    #     sid = torch.from_numpy(batch["sid"]).long().cuda()
    #     text = torch.from_numpy(batch["text"]).long().cuda()
    #     mel_target = torch.from_numpy(batch["mel_target"]).float().cuda()
    #     D = torch.from_numpy(batch["D"]).long().cuda()
    #     log_D = torch.from_numpy(batch["log_D"]).float().cuda()
    #     f0 = torch.from_numpy(batch["f0"]).float().cuda()
    #     energy = torch.from_numpy(batch["energy"]).float().cuda()
    #     src_len = torch.from_numpy(batch["src_len"]).long().cuda()
    #     mel_len = torch.from_numpy(batch["mel_len"]).long().cuda()
    #     max_src_len = np.max(batch["src_len"]).astype(np.int32)
    #     max_mel_len = np.max(batch["mel_len"]).astype(np.int32)
    #     return sid, text, mel_target, D, log_D, f0, energy, src_len, mel_len, max_src_len, max_mel_len

    def forward(self, content_embed, style_embed, mel_target):
        # content_embed: bz x 128/2 x 64; bz x time x frequency
        # style_embed: bz x 256 x 1;
        content_mask = get_mask_from_lengths(content_embed)
        
        # Cross Modal Shifting
        encoder_output, content_embed, _ = self.cross_modal_shift(content_embed, style_embed, content_mask) # enc_output: bz x 128/2 x 64; bz x time x frequency
        # # Variance Adaptor; Do I need This?
        # acoustic_adaptor_output, d_prediction, p_prediction, e_prediction, mel_len, mel_mask = self.variance_adaptor(
        #         encoder_output, src_mask, mel_len, mel_mask, 
        #                 d_target, p_target, e_target, max_mel_len)
        # Deocoding
        reconstruction_loss, mel_prediction = self.decoder(encoder_output, mel_target)

        return reconstruction_loss, mel_prediction # src_embedded, src_mask, mel_mask, mel_len

    # def inference(self, style_vector, src_seq, src_len=None, max_src_len=None, return_attn=False):
    #     src_mask = get_mask_from_lengths(src_len, max_src_len)
        
    #     # Encoding
    #     encoder_output, src_embedded, enc_slf_attn = self.encoder(src_seq, style_vector, src_mask)

    #     # Variance Adaptor
    #     acoustic_adaptor_output, d_prediction, p_prediction, e_prediction, \
    #             mel_len, mel_mask = self.variance_adaptor(encoder_output, src_mask)

    #     # Deocoding
    #     mel_output, dec_slf_attn = self.decoder(acoustic_adaptor_output, style_vector, mel_mask)

    #     if return_attn:
    #         return enc_slf_attn, dec_slf_attn

    #     return mel_output, src_embedded, d_prediction, p_prediction, e_prediction, src_mask, mel_mask, mel_len

class Postnet(nn.Module):
    """Postnet
        - Five 1-d convolution with 512 channels and kernel size 5
    """

    def __init__(self):
        super(Postnet, self).__init__()
        self.convolutions = nn.ModuleList()

        self.convolutions.append(
            nn.Sequential(
                ConvNorm(80, 512,
                         kernel_size=5, stride=1,
                         padding=2,
                         dilation=1, w_init_gain='tanh'),
                nn.BatchNorm1d(512))
        )

        for i in range(1, 5 - 1):
            self.convolutions.append(
                nn.Sequential(
                    ConvNorm(512,
                             512,
                             kernel_size=5, stride=1,
                             padding=2,
                             dilation=1, w_init_gain='tanh'),
                    nn.BatchNorm1d(512))
            )

        self.convolutions.append(
            nn.Sequential(
                ConvNorm(512, 80,
                         kernel_size=5, stride=1,
                         padding=2,
                         dilation=1, w_init_gain='linear'),
                nn.BatchNorm1d(80))
            )

    def forward(self, x):
        for i in range(len(self.convolutions) - 1):
            x = torch.tanh(self.convolutions[i](x))

        x = self.convolutions[-1](x)

        return x  

class Decoder(nn.Module):
    def __init__(self, channels, output_channels, z_dim):
        super(Decoder, self).__init__()
        self.conv = nn.ConvTranspose1d(channels, output_channels, 4, 2, 1, bias=False)
        self.decoder = nn.Sequential(
            nn.LayerNorm(z_dim, channels),
            nn.ReLU(True),
            nn.Linear(channels, channels, bias=False),
            nn.LayerNorm(channels),
            nn.ReLU(True),
            nn.Linear(channels, channels, bias=False),
            nn.LayerNorm(channels),
            nn.ReLU(True),
            nn.Linear(channels, channels, bias=False),
            nn.LayerNorm(channels),
            nn.ReLU(True),
            nn.Linear(channels, channels, bias=False),
            nn.LayerNorm(channels),
            nn.ReLU(True),
        )
        self.postnet = Postnet()

    def forward(self, x, mel_target):
        x = self.decoder(x)
        mel_prediction = self.conv(x.transpose(1, 2))
        mel_prediction = self.postnet(mel_prediction)
        return reconstruction_loss, mel_prediction

class CrossModalShift(nn.Module):
    ''' Encoder '''
    def __init__(self, config, n_src_vocab=len(symbols)+1):
        super(CrossModalShift, self).__init__()
        self.max_seq_len = config.max_seq_len
        self.n_layers = config.encoder_layer
        self.c_model = config.content_hidden
        self.d_model = config.encoder_hidden
        self.n_head = config.encoder_head
        self.d_k = config.encoder_hidden // config.encoder_head
        self.d_v = config.encoder_hidden // config.encoder_head
        self.d_inner = config.fft_conv1d_filter_size
        self.fft_conv1d_kernel_size = config.fft_conv1d_kernel_size
        self.d_out = config.decoder_hidden
        self.style_dim = config.style_vector_dim
        self.dropout = config.dropout

        self.src_word_emb = nn.Embedding(n_src_vocab, self.d_model, padding_idx=Constants.PAD)
        self.prenet = Prenet(self.c_model, self.d_model)

        n_position = self.max_seq_len + 1
        self.position_enc = nn.Parameter(
            get_sinusoid_encoding_table(n_position, self.d_model).unsqueeze(0), requires_grad = False)

        self.layer_stack = nn.ModuleList([FFTBlock(
            self.d_model, self.d_inner, self.n_head, self.d_k, self.d_v, 
            self.fft_conv1d_kernel_size, self.style_dim, self.dropout) for _ in range(self.n_layers)])

        self.fc_out = nn.Linear(self.d_model, self.d_out)

    def forward(self, content_embedding, style_embedding, mask):
        batch_size, max_len = content_embedding.shape[0], content_embedding.shape[1]
        
        # -- Prepare masks
        slf_attn_mask = mask.unsqueeze(1).expand(-1, max_len, -1)

        # -- Forward
        # prenet
        content_embedding = self.prenet(content_embedding, mask) # bz x 128/2 x 64 -> bz x 128/2 x 256; bz x time x frequency
        # position encoding
        if content_embedding.shape[1] > self.max_seq_len:
            position_embedded = get_sinusoid_encoding_table(content_embedding.shape[1], self.d_model)[:content_embedding.shape[1], :].unsqueeze(0).expand(batch_size, -1, -1).to(content_embedding.device)
        else:
            position_embedded = self.position_enc[:, :max_len, :].expand(batch_size, -1, -1)
        enc_output = content_embedding + position_embedded
        # fft blocks
        slf_attn = []
        for enc_layer in self.layer_stack:
            enc_output, enc_slf_attn = enc_layer(
                enc_output, style_embedding, 
                mask=mask, 
                slf_attn_mask=slf_attn_mask)
            slf_attn.append(enc_slf_attn)
        # last fc
        enc_output = self.fc_out(enc_output) # bz x 128/2 x 256 -> bz x 128/2 x 64; bz x time x frequency
        return enc_output, content_embedding, slf_attn

class Prenet(nn.Module):
    ''' Prenet '''
    def __init__(self, hidden_dim, out_dim):
        super(Prenet, self).__init__()

        # self.convs = nn.Sequential(
        #     ConvNorm(hidden_dim, hidden_dim, kernel_size=3),
        #     Mish(),
        #     nn.Dropout(dropout),
        #     ConvNorm(hidden_dim, hidden_dim, kernel_size=3),
        #     Mish(),
        #     nn.Dropout(dropout),
        # )
        self.fc = LinearNorm(hidden_dim, out_dim)

    def forward(self, input, mask=None):
        # # convs
        # output = input.transpose(1,2)
        # output = self.convs(output)
        # output = output.transpose(1,2)
        # # fc & residual
        output = self.fc(input)

        if mask is not None:
            output = output.masked_fill(mask.unsqueeze(-1), 0)
        return output
    
class PositionwiseFeedForward(nn.Module):
    ''' A two-feed-forward-layer module '''
    def __init__(self, d_in, d_hid, fft_conv1d_kernel_size, dropout=0.1):
        super().__init__()
        self.w_1 = ConvNorm(d_in, d_hid, kernel_size=fft_conv1d_kernel_size[0])
        self.w_2 =  ConvNorm(d_hid, d_in, kernel_size=fft_conv1d_kernel_size[1])

        self.mish = Mish()
        self.dropout = nn.Dropout(dropout)

    def forward(self, input):
        residual = input

        output = input.transpose(1, 2)
        output = self.w_2(self.dropout(self.mish(self.w_1(output))))
        output = output.transpose(1, 2)

        output = self.dropout(output) + residual
        return output

class FFTBlock(nn.Module):
    ''' FFT Block '''
    def __init__(self, d_model,d_inner,
                    n_head, d_k, d_v, fft_conv1d_kernel_size, style_dim, dropout):
        super(FFTBlock, self).__init__()
        self.slf_attn = MultiHeadAttention(
            n_head, d_model, d_k, d_v, dropout=dropout)
        self.saln_0 = StyleAdaptiveLayerNorm(d_model, style_dim)

        self.pos_ffn = PositionwiseFeedForward(
            d_model, d_inner, fft_conv1d_kernel_size, dropout=dropout)
        self.saln_1 = StyleAdaptiveLayerNorm(d_model, style_dim)

    def forward(self, input, style_vector, mask=None, slf_attn_mask=None):
        # multi-head self attn
        slf_attn_output, slf_attn = self.slf_attn(input, mask=slf_attn_mask)
        slf_attn_output = self.saln_0(slf_attn_output, style_vector)
        if mask is not None:
            slf_attn_output = slf_attn_output.masked_fill(mask.unsqueeze(-1), 0)

        # position wise FF
        output = self.pos_ffn(slf_attn_output)
        output = self.saln_1(output, style_vector)
        if mask is not None:
            output = output.masked_fill(mask.unsqueeze(-1), 0)

        return output, slf_attn
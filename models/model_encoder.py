import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np


class ConvNorm(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1,
                 padding=None, dilation=1, bias=True, w_init_gain='linear'):
        super(ConvNorm, self).__init__()
        if padding is None:
            assert(kernel_size % 2 == 1)
            padding = int(dilation * (kernel_size - 1) / 2)

        self.conv = torch.nn.Conv1d(in_channels, out_channels,
                                    kernel_size=kernel_size, stride=stride,
                                    padding=padding, dilation=dilation,
                                    bias=bias)

        torch.nn.init.xavier_uniform_(
            self.conv.weight, gain=torch.nn.init.calculate_gain(w_init_gain)) # initialize the weights such that the variance of the activations are the same across every layer        

    def forward(self, signal):
        conv_signal = self.conv(signal)
        return conv_signal

def pad_layer(inp, layer, pad_type='reflect'):
    kernel_size = layer.kernel_size[0]
    if kernel_size % 2 == 0:
        pad = (kernel_size//2, kernel_size//2 - 1)
    else:
        pad = (kernel_size//2, kernel_size//2)
    # padding
    inp = F.pad(inp, 
            pad=pad,
            mode=pad_type)
    out = layer(inp)
    return out

def conv_bank(x, module_list, act, pad_type='reflect'):
    outs = []
    for layer in module_list:
        out = act(pad_layer(x, layer, pad_type))
        outs.append(out)
    out = torch.cat(outs + [x], dim=1)
    return out

def get_act(act):
    if act == 'relu':
        return nn.ReLU()
    elif act == 'lrelu':
        return nn.LeakyReLU()
    else:
        return nn.ReLU()

class LinearNorm(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 bias=True, 
                 spectral_norm=False,
                 ):
        super(LinearNorm, self).__init__()
        self.fc = nn.Linear(in_channels, out_channels, bias)
        
        if spectral_norm:
            self.fc = nn.utils.spectral_norm(self.fc)

    def forward(self, input):
        out = self.fc(input)
        return out

class Mish(nn.Module):
    def __init__(self):
        super(Mish, self).__init__()
    def forward(self, x):
        return x * torch.tanh(F.softplus(x))
 
class ScaledDotProductAttention(nn.Module):
    ''' Scaled Dot-Product Attention '''

    def __init__(self, temperature, dropout):
        super().__init__()
        self.temperature = temperature
        self.softmax = nn.Softmax(dim=2)
        self.dropout = nn.Dropout(dropout)

    def forward(self, q, k, v, mask=None):

        attn = torch.bmm(q, k.transpose(1, 2))
        attn = attn / self.temperature

        if mask is not None:
            attn = attn.masked_fill(mask, -np.inf)

        attn = self.softmax(attn)
        p_attn = self.dropout(attn)

        output = torch.bmm(p_attn, v)
        return output, attn
   
class MultiHeadAttention(nn.Module):
    ''' Multi-Head Attention module '''
    def __init__(self, n_head, d_model, d_k, d_v, dropout=0., spectral_norm=False):
        super().__init__()

        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v

        self.w_qs = nn.Linear(d_model, n_head * d_k)
        self.w_ks = nn.Linear(d_model, n_head * d_k)
        self.w_vs = nn.Linear(d_model, n_head * d_v)
        
        self.attention = ScaledDotProductAttention(temperature=np.power(d_model, 0.5), dropout=dropout)

        self.fc = nn.Linear(n_head * d_v, d_model)
        self.dropout = nn.Dropout(dropout)

        if spectral_norm:
            self.w_qs = nn.utils.spectral_norm(self.w_qs)
            self.w_ks = nn.utils.spectral_norm(self.w_ks)
            self.w_vs = nn.utils.spectral_norm(self.w_vs)
            self.fc = nn.utils.spectral_norm(self.fc)

    def forward(self, x, mask=None):
        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head
        sz_b, len_x, _ = x.size()

        residual = x

        q = self.w_qs(x).view(sz_b, len_x, n_head, d_k)
        k = self.w_ks(x).view(sz_b, len_x, n_head, d_k)
        v = self.w_vs(x).view(sz_b, len_x, n_head, d_v)
        q = q.permute(2, 0, 1, 3).contiguous().view(-1,
                                                    len_x, d_k)  # (n*b) x lq x dk
        k = k.permute(2, 0, 1, 3).contiguous().view(-1,
                                                    len_x, d_k)  # (n*b) x lk x dk
        v = v.permute(2, 0, 1, 3).contiguous().view(-1,
                                                    len_x, d_v)  # (n*b) x lv x dv

        if mask is not None:
            slf_mask = mask.repeat(n_head, 1, 1)  # (n*b) x .. x ..
        else:
            slf_mask = None
        output, attn = self.attention(q, k, v, mask=slf_mask)

        output = output.view(n_head, sz_b, len_x, d_v)
        output = output.permute(1, 2, 0, 3).contiguous().view(
                        sz_b, len_x, -1)  # b x lq x (n*dv)

        output = self.fc(output)

        output = self.dropout(output) + residual
        return output, attn

class Conv1dGLU(nn.Module):
    '''
    Conv1d + GLU(Gated Linear Unit) with residual connection.
    For GLU refer to https://arxiv.org/abs/1612.08083 paper.
    '''
    def __init__(self, in_channels, out_channels, kernel_size, dropout):
        super(Conv1dGLU, self).__init__()
        self.out_channels = out_channels
        self.conv1 = ConvNorm(in_channels, 2*out_channels, kernel_size=kernel_size)
        self.dropout = nn.Dropout(dropout)
            
    def forward(self, x):
        residual = x
        x = self.conv1(x)
        x1, x2 = torch.split(x, split_size_or_sections=self.out_channels, dim=1)
        x = x1 * torch.sigmoid(x2)
        x = residual + self.dropout(x)
        return x

# class StyleEncoder(nn.Module):
#     '''
#     reference from speaker-encoder of AdaIN-VC: https://github.com/jjery2243542/adaptive_voice_conversion/blob/master/model.py
#     '''
#     def __init__(self, c_in=80, c_h=128, c_out=1024, kernel_size=5,
#             bank_size=8, bank_scale=1, c_bank=128, 
#             n_conv_blocks=6, n_dense_blocks=6, 
#             subsample=[1, 2, 1, 2, 1, 2], act='relu', dropout_rate=0):
#         super(StyleEncoder, self).__init__()
#         self.c_in = c_in
#         self.c_h = c_h
#         self.c_out = c_out
#         self.kernel_size = kernel_size
#         self.n_conv_blocks = n_conv_blocks
#         self.n_dense_blocks = n_dense_blocks
#         self.subsample = subsample
        
#         self.act = get_act(act)
        
#         self.conv_bank_early = nn.ModuleList(
#                 [nn.Conv1d(c_in, c_bank, kernel_size=k) for k in range(bank_scale, bank_size + 1, bank_scale)])
#         self.conv_bank_late = nn.ModuleList(
#                 [nn.Conv1d(c_h, c_bank, kernel_size=k) for k in range(bank_scale, bank_size, bank_scale)])
        
#         in_channels = c_bank * (bank_size // bank_scale) + c_in
#         self.in_conv_layer = nn.Conv1d(in_channels, c_h, kernel_size=1)
#         self.first_conv_layers = nn.ModuleList([nn.Conv1d(c_h, c_h, kernel_size=kernel_size) for _ \
#                 in range(n_conv_blocks)])
#         self.second_conv_layers = nn.ModuleList([nn.Conv1d(c_h, c_h, kernel_size=kernel_size, stride=sub) 
#             for sub, _ in zip(subsample, range(n_conv_blocks))])
        
#         self.pooling_layer = nn.AdaptiveAvgPool1d(1)
        
#         channels = c_bank * ((bank_size - 1) // bank_scale) + c_h
#         self.first_dense_layers = nn.ModuleList([nn.Linear(channels, channels) for _ in range(n_dense_blocks)])
#         self.second_dense_layers = nn.ModuleList([nn.Linear(channels, channels) for _ in range(n_dense_blocks)])
        
#         self.output_layer = nn.Linear(c_h, c_out)
        
#         self.dropout_layer = nn.Dropout(p=dropout_rate)

#     def conv_blocks(self, inp):
#         out = inp
#         # convolution blocks
#         for l in range(self.n_conv_blocks):
#             y = pad_layer(out, self.first_conv_layers[l])
#             y = self.act(y)
#             y = self.dropout_layer(y)
#             y = pad_layer(y, self.second_conv_layers[l])
#             y = self.act(y)
#             y = self.dropout_layer(y)
#             if self.subsample[l] > 1:
#                 out = F.avg_pool1d(out, kernel_size=self.subsample[l], ceil_mode=True)
#             out = y + out
#         return out

#     def dense_blocks(self, inp):
#         out = inp
#         # dense layers
#         for l in range(self.n_dense_blocks):
#             y = self.first_dense_layers[l](out)
#             y = self.act(y)
#             y = self.dropout_layer(y)
#             y = self.second_dense_layers[l](y)
#             y = self.act(y)
#             y = self.dropout_layer(y)
#             out = y + out
#         return out

#     def forward(self, x):
#         out = conv_bank(x, self.conv_bank_early, act=self.act)
#         # dimension reduction layer
#         out = pad_layer(out, self.in_conv_layer)
#         out = self.act(out)
#         # conv blocks
#         out = self.conv_blocks(out)
#         out = conv_bank(out, self.conv_bank_late, act=self.act)
#         # avg pooling
#         out = self.pooling_layer(out).squeeze(2)
#         # dense blocks
#         out = self.dense_blocks(out)
#         out = self.output_layer(out)
#         return out

class StyleEncoder(nn.Module):
    ''' MelStyleEncoder '''
    def __init__(self, config):
        super(StyleEncoder, self).__init__()
        self.in_dim = config.n_mel_channels 
        self.hidden_dim = config.style_hidden
        self.out_dim = config.style_vector_dim
        self.kernel_size = config.style_kernel_size
        self.n_head = config.style_head
        self.dropout = config.dropout

        self.spectral = nn.Sequential(
            LinearNorm(self.in_dim, self.hidden_dim),
            Mish(),
            nn.Dropout(self.dropout),
            LinearNorm(self.hidden_dim, self.hidden_dim),
            Mish(),
            nn.Dropout(self.dropout)
        )

        self.temporal = nn.Sequential(
            Conv1dGLU(self.hidden_dim, self.hidden_dim, self.kernel_size, self.dropout),
            Conv1dGLU(self.hidden_dim, self.hidden_dim, self.kernel_size, self.dropout),
        )

        self.slf_attn = MultiHeadAttention(self.n_head, self.hidden_dim, 
                                self.hidden_dim//self.n_head, self.hidden_dim//self.n_head, self.dropout) 

        self.fc = LinearNorm(self.hidden_dim, self.out_dim)

    def temporal_avg_pool(self, x, mask=None):
        if mask is None:
            out = torch.mean(x, dim=1)
        else:
            len_ = (~mask).sum(dim=1).unsqueeze(1)
            x = x.masked_fill(mask.unsqueeze(-1), 0)
            x = x.sum(dim=1)
            out = torch.div(x, len_)
        return out

    def forward(self, x, mask=None):
        max_len = x.shape[1]
        slf_attn_mask = mask.unsqueeze(1).expand(-1, max_len, -1) if mask is not None else None
        
        # spectral
        x = self.spectral(x)
        # temporal
        x = x.transpose(1,2)
        x = self.temporal(x)
        x = x.transpose(1,2)
        # self-attention
        if mask is not None:
            x = x.masked_fill(mask.unsqueeze(-1), 0)
        x, _ = self.slf_attn(x, mask=slf_attn_mask)
        # fc
        x = self.fc(x)
        # temoral average pooling
        w = self.temporal_avg_pool(x, mask=mask)

        return w
    
class ContentEncoder(nn.Module):
    '''
    reference from: https://github.com/bshall/VectorQuantizedCPC/blob/master/model.py
    '''
    def __init__(self, in_channels, channels, n_embeddings, z_dim, c_dim):
        super(ContentEncoder, self).__init__()
        self.conv = nn.Conv1d(in_channels, channels, 4, 2, 1, bias=False)
        self.encoder = nn.Sequential(
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
            nn.Linear(channels, channels, bias=False),
            nn.LayerNorm(channels),
            nn.ReLU(True),
            nn.Linear(channels, z_dim),
        )
        self.codebook = VQEmbeddingEMA(n_embeddings, z_dim)
        self.rnn = nn.LSTM(z_dim, c_dim, batch_first=True)

    def encode(self, mel):
        z = self.conv(mel)
        z_beforeVQ = self.encoder(z.transpose(1, 2))
        z, r, indices = self.codebook.encode(z_beforeVQ)
        c, _ = self.rnn(z)
        return z, c, z_beforeVQ, indices

    def forward(self, mels):
        z = self.conv(mels.float()) # (bz, 80, 128) -> (bz, 512, 128/2); (bz, time, mel_frequency)
        z_beforeVQ = self.encoder(z.transpose(1, 2)) # (bz, 512, 128/2) -> (bz, 128/2, 512) -> (bz, 128/2, 64)
        z, r, loss, perplexity = self.codebook(z_beforeVQ) # z: (bz, 128/2, 64)
        c, _ = self.rnn(z) # (64, 140/2, 64) -> (64, 140/2, 256)
        return z, c, z_beforeVQ, loss, perplexity
    
class VQEmbeddingEMA(nn.Module):
    '''
    reference from: https://github.com/bshall/VectorQuantizedCPC/blob/master/model.py
    '''
    def __init__(self, n_embeddings, embedding_dim, commitment_cost=0.25, decay=0.999, epsilon=1e-5):
        super(VQEmbeddingEMA, self).__init__()
        self.commitment_cost = commitment_cost
        self.decay = decay
        self.epsilon = epsilon

        init_bound = 1 / 512
        embedding = torch.Tensor(n_embeddings, embedding_dim)
        embedding.uniform_(-init_bound, init_bound)
        self.register_buffer("embedding", embedding) # only change during forward
        self.register_buffer("ema_count", torch.zeros(n_embeddings))
        self.register_buffer("ema_weight", self.embedding.clone())

    def encode(self, x):
        M, D = self.embedding.size()
        x_flat = x.detach().reshape(-1, D)

        distances = torch.addmm(torch.sum(self.embedding ** 2, dim=1) +
                                torch.sum(x_flat ** 2, dim=1, keepdim=True),
                                x_flat, self.embedding.t(),
                                alpha=-2.0, beta=1.0)

        indices = torch.argmin(distances.float(), dim=-1)
        quantized = F.embedding(indices, self.embedding)
        quantized = quantized.view_as(x)
        residual = x - quantized
        return quantized, residual, indices.view(x.size(0), x.size(1))

    def forward(self, x):
        M, D = self.embedding.size()
        x_flat = x.detach().reshape(-1, D)

        distances = torch.addmm(torch.sum(self.embedding ** 2, dim=1) +
                                torch.sum(x_flat ** 2, dim=1, keepdim=True),
                                x_flat, self.embedding.t(),
                                alpha=-2.0, beta=1.0) # calculate the distance between each ele in embedding and x

        indices = torch.argmin(distances.float(), dim=-1)
        encodings = F.one_hot(indices, M).float()
        quantized = F.embedding(indices, self.embedding)
        quantized = quantized.view_as(x)

        if self.training: # EMA based codebook learning
            self.ema_count = self.decay * self.ema_count + (1 - self.decay) * torch.sum(encodings, dim=0)

            n = torch.sum(self.ema_count)
            self.ema_count = (self.ema_count + self.epsilon) / (n + M * self.epsilon) * n

            dw = torch.matmul(encodings.t(), x_flat)
            self.ema_weight = self.decay * self.ema_weight + (1 - self.decay) * dw

            self.embedding = self.ema_weight / self.ema_count.unsqueeze(-1)

        e_latent_loss = F.mse_loss(x, quantized.detach())
        loss = self.commitment_cost * e_latent_loss
        
        residual = x - quantized
        
        quantized = x + (quantized - x).detach()

        avg_probs = torch.mean(encodings, dim=0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))

        return quantized, residual, loss, perplexity


class CPCLoss(nn.Module):
    '''
    CPC-loss calculation: negative samples are drawn within-speaker
    reference from: https://github.com/bshall/VectorQuantizedCPC/blob/master/model.py
    '''
    def __init__(self, n_speakers_per_batch, n_utterances_per_speaker, n_prediction_steps, n_negatives, z_dim, c_dim):
        super(CPCLoss, self).__init__()
        self.n_speakers_per_batch = n_speakers_per_batch
        self.n_utterances_per_speaker = n_utterances_per_speaker
        self.n_prediction_steps = n_prediction_steps // 2
        self.n_negatives = n_negatives
        self.z_dim = z_dim
        self.c_dim = c_dim
        self.predictors = nn.ModuleList([
            nn.Linear(c_dim, z_dim) for _ in range(n_prediction_steps)
        ])

    def forward(self, z, c): # z:(64, 70, 64), c:(64, 70, 256)
        length = z.size(1) - self.n_prediction_steps # 64

        z = z.reshape(
            self.n_speakers_per_batch,
            self.n_utterances_per_speaker,
            -1,
            self.z_dim
        ) # (64, 70, 64) -> (8, 8, 70, 64)
        c = c[:, :-self.n_prediction_steps, :] # (64, 64, 256)

        losses, accuracies = list(), list()
        for k in range(1, self.n_prediction_steps+1):
            z_shift = z[:, :, k:length + k, :] # (8, 8, 64, 64), positive samples

            Wc = self.predictors[k-1](c) # (64, 64, 256) -> (64, 64, 64)
            Wc = Wc.view(
                self.n_speakers_per_batch,
                self.n_utterances_per_speaker,
                -1,
                self.z_dim
            ) # (64, 64, 64) -> (8, 8, 64, 64)

            batch_index = torch.randint(
                0, self.n_utterances_per_speaker,
                size=(
                    self.n_utterances_per_speaker,
                    self.n_negatives
                ),
                device=z.device
            )
            batch_index = batch_index.view(
                1, self.n_utterances_per_speaker, self.n_negatives, 1
            ) # (1, 8, 17, 1)

            # seq_index: (8, 8, 17, 64)
            seq_index = torch.randint(
                1, length,
                size=(
                    self.n_speakers_per_batch,
                    self.n_utterances_per_speaker,
                    self.n_negatives,
                    length
                ),
                device=z.device
            ) 
            seq_index += torch.arange(length, device=z.device) #(1)
            seq_index = torch.remainder(seq_index, length) #(2) (1)+(2) ensures that the current positive frame will not be selected as negative sample...
            
            speaker_index = torch.arange(self.n_speakers_per_batch, device=z.device) # within-speaker sampling
            speaker_index = speaker_index.view(-1, 1, 1, 1)
            
            # z_negatives: (8,8,17,64,64); z_negatives[0,0,:,0,:] is (17, 64) that is negative samples for first frame of first utterance of first speaker...
            z_negatives = z_shift[speaker_index, batch_index, seq_index, :] # speaker_index has the original order (within-speaker sampling)
                                                                            # batch_index is randomly sampled from 0~7, each point has 17 negative samples
                                                                            # seq_index is randomly sampled from 0~115
                                                                        # so for each positive frame with time-id as t, the negative samples will be selected from 
                                                                        # another or the current utterance and the seq-index (frame-index) will not conclude t  

            zs = torch.cat((z_shift.unsqueeze(2), z_negatives), dim=2) # (8, 8, 1+17, 64, 64)

            f = torch.sum(zs * Wc.unsqueeze(2) / math.sqrt(self.z_dim), dim=-1) # (8, 8, 1+17, 64), vector product in fact...
            f = f.view(
                self.n_speakers_per_batch * self.n_utterances_per_speaker,
                self.n_negatives + 1,
                -1
            ) # (64, 1+17, 64)

            labels = torch.zeros(
                self.n_speakers_per_batch * self.n_utterances_per_speaker, length,
                dtype=torch.long, device=z.device
            ) # (64, 64)

            loss = F.cross_entropy(f, labels)

            accuracy = f.argmax(dim=1) == labels # (64, 116)
            accuracy = torch.mean(accuracy.float())

            losses.append(loss)
            accuracies.append(accuracy.item())

        loss = torch.stack(losses).mean()
        return loss, accuracies


class CPCLoss_sameSeq(nn.Module):
    '''
    CPC-loss calculation: negative samples are drawn within-sequence/utterance
    '''
    def __init__(self, n_speakers_per_batch, n_utterances_per_speaker, n_prediction_steps, n_negatives, z_dim, c_dim):
        super(CPCLoss_sameSeq, self).__init__()
        self.n_speakers_per_batch = n_speakers_per_batch
        self.n_utterances_per_speaker = n_utterances_per_speaker
        self.n_prediction_steps = n_prediction_steps 
        self.n_negatives = n_negatives
        self.z_dim = z_dim
        self.c_dim = c_dim
        self.predictors = nn.ModuleList([
            nn.Linear(c_dim, z_dim) for _ in range(n_prediction_steps)
        ])

    def forward(self, z, c): # z:(256, 64, 64), c:(256, 64, 256)
        length = z.size(1) - self.n_prediction_steps # 64-6=58, length is the total time-steps of each utterance used for calculated cpc loss
        n_speakers_per_batch = z.shape[0] # each utterance is treated as a speaker
        c = c[:, :-self.n_prediction_steps, :] # (256, 58, 256)

        losses, accuracies = list(), list()
        for k in range(1, self.n_prediction_steps+1):
            z_shift = z[:, k:length + k, :] # (256, 58, 64), positive samples

            Wc = self.predictors[k-1](c) # (256, 58, 256) -> (256, 58, 64)

            # seq_index: (256, 10, 58)
            seq_index = torch.randint(
                1, length,
                size=(
                    n_speakers_per_batch,
                    self.n_negatives,
                    length
                ),
                device=z.device
            ) 
            seq_index += torch.arange(length, device=z.device) #(1)
            seq_index = torch.remainder(seq_index, length) #(2) (1)+(2) ensures that the current positive frame will not be selected as negative sample...
            
            speaker_index = torch.arange(n_speakers_per_batch, device=z.device) # within-utterance sampling
            speaker_index = speaker_index.view(-1, 1, 1)
            
            
            z_negatives = z_shift[speaker_index, seq_index, :] # (256,10,58,64), z_negatives[i,:,j,:] is the negative samples set for ith utterance and jth time-step

            zs = torch.cat((z_shift.unsqueeze(1), z_negatives), dim=1) # (256,11,58,64) 

            f = torch.sum(zs * Wc.unsqueeze(1) / math.sqrt(self.z_dim), dim=-1) # (256,11,58), vector product in fact...
            
            labels = torch.zeros(
                n_speakers_per_batch, length,
                dtype=torch.long, device=z.device
            ) 

            loss = F.cross_entropy(f, labels)

            accuracy = f.argmax(dim=1) == labels # (256, 58)
            accuracy = torch.mean(accuracy.float())

            losses.append(loss)
            accuracies.append(accuracy.item())

        loss = torch.stack(losses).mean()
        return loss, accuracies
    



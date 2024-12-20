import math
import torchaudio.transforms
from model.model import *
from torchaudio.functional import lfilter
from .mi import MI


class VC(nn.Module):
    def __init__(self, core, model_name, sample_rate, mel, regularization, mi_cfg):
        super().__init__()
        self.core = core
        self.model_name = model_name
        self.sample_rate = sample_rate
        self.mel = mel
        self.mel_transform = self.make_mel_transform()
        self.regularization = regularization
        self.mi = MI(mi_cfg)

    def time_shuffle(self, data):
        seg_list = torch.split(data, 20, dim=2)  # TODO: shuffle size can be adjusted
        indices = torch.randperm(len(seg_list))
        shuffled_seg_list = [seg_list[i] for i in indices]
        data_shuffled = torch.cat(shuffled_seg_list, dim=2)
        return data_shuffled

    def make_mel_transform(self):
        transform = []
        for i in range(len(self.mel['win_length'])):
            transform_i = torchaudio.transforms.MelSpectrogram(sample_rate=self.sample_rate,
                                                               n_fft=self.mel['n_fft'][i],
                                                               win_length=self.mel['win_length'][i],
                                                               hop_length=self.mel['hop_length'][i],
                                                               n_mels=self.mel['n_mels'][i],
                                                               f_min=self.mel['f_min'],
                                                               f_max=self.mel['f_max'],
                                                               power=1,
                                                               center=True)
            transform.append(transform_i)
        transform = nn.ModuleList(transform)
        return transform

    def make_mel(self, audio):
        if 'preemph' in self.mel and self.mel['preemph'] > 0:
            audio = lfilter(audio, torch.tensor([1, 0], device=audio.device),
                            torch.tensor([1, -self.mel['preemph']], device=audio.device))
        mel = []
        for i in range(len(self.mel_transform)):
            mel_i = self.mel_transform[i](audio)
            mel.append(mel_i)
        return mel

    def forward(self, input):
        output = {}
        print(input.keys())
        if self.model_name == 'mainvc':
            print(input['audio'].size())
            mel = self.make_mel(input['audio'])[0]  # N, S, T
            ref_mel = self.make_mel(input['ref_audio'])[0]
            log_mel = (mel + 1e-9).log()
            # log_mel[mel == 0] = 0
            log_ref_mel = (ref_mel + 1e-9).log()
            # log_ref_mel[ref_mel == 0] = 0
            log_mel_shuffled = self.time_shuffle(log_mel)
            mu, log_sigma, emb, emb_, dec = self.core(log_mel, log_mel_shuffled, log_ref_mel)

            # loss_flag = torch.ones([emb.shape[0]]).to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
            # emb = emb.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
            # emb_ = emb_.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))

            # reconstruction loss
            # print(dec.size(), log_mel.size(), log_mel_shuffled.size(), log_ref_mel.size())
            # exit()
            # print(log_mel.isnan().any())
            # print(dec.isnan().any())
            # exit()
            # loss_rec = F.l1_loss(dec[mel != 0], log_mel[mel != 0])
            loss_rec = F.l1_loss(dec, log_mel)
            # KL loss
            if self.regularization['kl'] > 0:
                loss_kl = 0.5 * torch.mean(torch.exp(log_sigma) + mu ** 2 - 1 - log_sigma)
            else:
                loss_kl = 0
            # siamese loss
            if self.regularization['sia'] > 0:
                # cos = nn.CosineEmbeddingLoss(reduction="mean")
                loss_flag = emb.new_ones([emb.shape[0]])
                loss_sia = F.cosine_embedding_loss(emb, emb_, loss_flag)
            else:
                loss_sia = 0
            # CMI first forward
            if self.regularization['mi'] > 0:
                loss_mi = self.mi(mu, emb, self.training)
            else:
                loss_mi = 0

            # total loss
            print(loss_rec)
            print(loss_kl)
            print(loss_sia)
            print(loss_mi)
            loss = (self.regularization['rec'] * loss_rec + self.regularization['kl'] * loss_kl +
                    self.regularization['sia'] * loss_sia + self.regularization['mi'] * loss_mi)
            output['pred'] = dec
            output['loss'] = loss
            input['target'] = log_mel
            print(loss)
        # exit()
        return output


def vc(core, cfg):
    model = VC(core, cfg['model_name'], cfg['sample_rate'], cfg[cfg['model_name']]['mel'], cfg['regularization'],
               cfg['mi'])
    return model

# def log_mel_spectrogram(
#         y: np.ndarray,
#         preemph: float,
#         sample_rate: int,
#         n_mels: int,
#         n_fft: int,
#         hop_length: int,
#         win_length: int,
#         fmin: int,
#         fmax: int,
# ) -> np.ndarray:
#     """Create a log Mel spectrogram from a raw audio signal."""
#     if preemph > 0:
#         y = lfilter([1, -preemph], [1], y)
#     magnitude = np.abs(
#         librosa.stft(y=y, n_fft=n_fft, hop_length=hop_length, win_length=win_length)
#     )
#     mel_fb = librosa.filters.mel(sr=sample_rate, n_fft=n_fft, n_mels=n_mels, fmin=fmin, fmax=fmax)
#     mel_spec = np.dot(mel_fb, magnitude)
#     log_mel_spec = np.log(mel_spec + 1e-9).T
#     return log_mel_spec  # shape(T, n_mels)

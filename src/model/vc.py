import math
import torchaudio.transforms
from model.model import *
from torchaudio.functional import lfilter


class VC(nn.Module):
    def __init__(self, core, model_name, sample_rate, mel):
        super().__init__()
        self.core = core
        self.model_name = model_name
        self.sample_rate = sample_rate
        self.mel = mel
        self.mel_transform = self.make_mel_transform()

    # def time_shuffle(self, data):
    #     seg_list = list(torch.split(data, 20, dim=2))
    #     random.shuffle(seg_list)
    #     return torch.cat(seg_list, dim=2)

    def time_shuffle(self, data):
        seg_list = torch.split(data, 20, dim=2)
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
                                                               hop_length=int(self.mel['win_length'][i] * \
                                                                              self.mel['hop_ratio']),
                                                               n_mels=self.mel['n_mels'][i],
                                                               f_min=self.mel['f_min'],
                                                               f_max=self.mel['f_max'],
                                                               power=1)
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
        print(input.keys())
        if self.model_name == 'mainvc':
            mel = self.make_mel(input['audio'])[0]
            log_mel = mel.log()
            print(log_mel.size())
            log_mel_shuffled = self.time_shuffle(log_mel)
            ref_mel = self.make_mel(input['ref_audio'])[0]
            log_ref_mel = ref_mel.log()
            mu, log_sigma, emb, emb_, dec = self.core(log_mel, log_mel_shuffled, log_ref_mel)
            print(mu.size(), log_sigma.size(), emb.size(), emb_.size(), dec.size())
            exit()
            # loss
            criterion = nn.L1Loss()
            cos = nn.CosineEmbeddingLoss(reduction="mean")
            loss_flag = torch.ones([emb.shape[0]]).to(
                torch.device("cuda" if torch.cuda.is_available() else "cpu")
            )
            emb = emb.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
            emb_ = emb_.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
            # reconstruction loss
            loss_rec = criterion(dec, x)
            # KL loss
            loss_kl = 0.5 * torch.mean(torch.exp(log_sigma) + mu ** 2 - 1 - log_sigma)
            # siamese loss
            loss_sia = cos(emb, emb_, loss_flag)

            # CMI first forward
            if self.cmi_activate:
                for _ in range(self.cmi_steps):
                    self.mi_opt.zero_grad()
                    mu_tmp = mu.transpose(1, 2)
                    emb_tmp = emb
                    mu_tmp = mu_tmp.detach()
                    emb_tmp = emb_tmp.detach()
                    self.mi_club.train()
                    self.mi_mine.train()
                    # jointly train CLUB and MINE
                    self.club_loss = -self.mi_club.loglikeli(emb_tmp, mu_tmp)
                    self.mine_loss = self.mi_mine.learning_loss(emb_tmp, mu_tmp)
                    delta = self.mi_club.mi_est(emb_tmp, mu_tmp) - self.mi_mine(
                        emb_tmp, mu_tmp
                    )
                    gap_loss = delta if delta > 0 else 0
                    mimodule_loss = self.club_loss + self.mine_loss + gap_loss
                    mimodule_loss.backward(retain_graph=True)
                    self.mi_opt.step()

            # CMI second forward
            # MI loss
            loss_mi = self.mi_club.mi_est(emb, mu.transpose(1, 2))

            # total loss
            lambda_sia = self.config["lambda"]["lambda_sia"] if self.sia_activate else 0
            lambda_mi = lambda_mi if self.cmi_activate else 0
            loss = (
                    self.config["lambda"]["lambda_rec"] * loss_rec
                    + lambda_kl * loss_kl
                    + lambda_sia * loss_sia
                    + lambda_mi * loss_mi
            )

        output = {}
        output['data'], output['loss'] = self.core(x_0, t, cond, training)
        return output


def vc(core, cfg):
    model = VC(core, cfg['model_name'], cfg['sample_rate'], cfg[cfg['model_name']]['mel'])
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

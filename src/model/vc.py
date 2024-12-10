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
            mel_i = self.mel_transform[i](audio[i])
            mel.append(mel_i)
        return mel

    def forward(self, input):
        print(input.keys())
        if self.model_name == 'mainvc':
            mel = self.make_mel(input['audio'])[0]
            log_mel = mel.log().T
            ref_mel = self.make_mel(input['ref_audio'])[0]
            log_ref_mel = ref_mel.log().T
            print(log_ref_mel.size())
            exit()
            output = self.core(log_mel, log_ref_mel)
            exit()
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

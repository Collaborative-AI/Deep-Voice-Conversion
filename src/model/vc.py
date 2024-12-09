import math
import torchaudio.transforms
from model.model import *
from scipy.signal import lfilter


class VC(nn.Module):
    def __init__(self, core, model_name, sample_rate, mel):
        super().__init__()
        self.core = core
        self.model_name = model_name
        self.sample_rate = sample_rate
        self.mel = mel

    def make_mel(self, audio):
        mel = []
        if 'preemph' in self.mel and self.mel['preemph'] > 0:
            audio = lfilter([1, -self.mel['preemph']], [1], audio)
        for i in range(len(self.mel['win_length'])):
            transform_i = torchaudio.transforms.MelSpectrogram(sample_rate=self.sample_rate,
                                                               n_fft=self.mel['n_fft'][i],
                                                               win_length=self.mel['win_length'][i],
                                                               hop_length=self.mel['win_length'][i] // 4,
                                                               n_mels=self.mel['n_mels'][i],
                                                               f_min=self.mel['f_min'],
                                                               f_max=self.mel['f_max'],
                                                               power=1)
            mel_i = transform_i(audio)
            mel.append(mel_i)
        return mel

    def forward(self, input):
        print(input.keys())
        if self.model_name == 'mainvc':
            input['mel'] = self.make_mel(input['audio'])
            input['ref_mel'] = self.make_mel(input['ref_audio'])
            self.core(input['mel'][0], input['ref_mel'][0])
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

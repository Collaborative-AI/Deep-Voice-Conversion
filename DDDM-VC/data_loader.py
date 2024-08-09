import numpy as np
import torch
import torchaudio
from torchaudio.transforms import MelSpectrogram
from utils import parse_filelist

np.random.seed(1234)
class AudioDataset(torch.utils.data.Dataset):
    """
    Provides dataset management for given filelist.
    """
    def __init__(self, config, training=True):
        super(AudioDataset, self).__init__()
        self.config = config
        self.hop_length = config.data.hop_length
        self.training = training
        self.mel_length = config.train.segment_size // config.data.hop_length
        if self.training:
            self.segment_length = config.train.segment_size
        self.sample_rate = config.data.sampling_rate

        self.filelist_path = config.data.train_filelist_path \
            if self.training else config.data.test_filelist_path
        self.audio_paths = parse_filelist(self.filelist_path)
        self.f0_norm_paths = parse_filelist(self.filelist_path.replace('_wav', '_f0'))

    def load_audio_to_torch(self, audio_path):
        audio, sample_rate = torchaudio.load(audio_path)

        if not self.training:
            p = (audio.shape[-1] // 1280 + 1) * 1280 - audio.shape[-1]
            audio = torch.nn.functional.pad(audio, (0, p), mode='constant').data
        return audio.squeeze(), sample_rate

    def __getitem__(self, index):
        audio_path = self.audio_paths[index]
        f0_norm_path = self.f0_norm_paths[index]
      
        audio, sample_rate = self.load_audio_to_torch(audio_path)
        f0_norm = torch.load(f0_norm_path)

        assert sample_rate == self.sample_rate, \
            f"""Got path to audio of sampling rate {sample_rate}, \
                but required {self.sample_rate} according config."""

        if not self.training:  
            return audio, f0_norm

        if audio.shape[-1] > self.segment_length:
            max_f0_start = f0_norm.shape[-1] - self.segment_length//80
            
            f0_start = np.random.randint(0, max_f0_start)
            f0_segment = f0_norm[:, f0_start:f0_start + self.segment_length//80]    
            audio_start = f0_start*80
            segment = audio[audio_start:audio_start + self.segment_length]
            if segment.shape[-1] < self.segment_length:
                segment = torch.nn.functional.pad(
                    segment, (0, self.segment_length - segment.shape[-1]), 'constant'
                ).data
            length = torch.LongTensor([self.mel_length])

        else:
            segment = torch.nn.functional.pad(
                audio, (0, self.segment_length - audio.shape[-1]), 'constant'
            ).data
            length = torch.LongTensor([audio.shape[-1] // self.hop_length])
            f0_segment = torch.nn.functional.pad(
                f0_norm, (0, self.segment_length // 80 - f0_norm.shape[-1]), 'constant'
            ).data

        return segment, f0_segment, length

    def __len__(self):
        return len(self.audio_paths)

    def sample_test_batch(self, size):
        idx = np.random.choice(range(len(self)), size=size, replace=False)
        test_batch = []
        for index in idx:
            test_batch.append(self.__getitem__(index))
        return test_batch

class MelSpectrogramFixed(torch.nn.Module):
    """In order to remove padding of torchaudio package + add log10 scale."""

    def __init__(self, **kwargs):
        super(MelSpectrogramFixed, self).__init__()
        self.torchaudio_backend = MelSpectrogram(**kwargs)

    def forward(self, x):
        outputs = torch.log(self.torchaudio_backend(x) + 0.001)

        return outputs[..., :-1]

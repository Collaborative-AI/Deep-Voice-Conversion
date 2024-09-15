import requests
import zipfile
import librosa
import soundfile as sf
import pandas as pd
import numpy as np
import os
import torch
import torch.nn.functional as F
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
from module import check_exists, makedir_exist_ok, save, load
from .utils import download_url, extract_file, make_classes_counts
from config import cfg


class VCTK(Dataset):
    def __init__(self, root, split, transform=None):
        assert split in ['train', 'test', 'val']

        self.root = root
        self.file = ('https://datashare.is.ed.ac.uk/bitstream/handle/10283/3443/VCTK-Corpus-0.92.zip', None)
        self.split = split
        self.transform = transform
        self.sample_rate = cfg['sample_rate']
        self.sr_int = str(int(self.sample_rate // 1e3))

        if not check_exists(self.processed_folder):
            self.process()
        self.data = load(os.path.join(self.processed_folder, self.sr_int, self.split))
        self.other = {}
        self.meta = load(os.path.join(self.processed_folder, self.sr_int, 'meta'))

    def __len__(self):
        return len(self.data)

    @property
    def processed_folder(self):
        return os.path.join(self.root, 'processed')

    @property
    def raw_folder(self):
        return os.path.join(self.root, 'raw')

    def process(self):
        if not check_exists(os.path.join(self.raw_folder, 'wav48_silence_trimmed')):
            self.download()
        if not check_exists(os.path.join(self.raw_folder, 'wav' + self.sr_int)):
            self.downsample()
        train_set, test_set, meta = self.make_data()
        save(train_set, os.path.join(self.processed_folder, self.sr_int, 'train'))
        save(test_set, os.path.join(self.processed_folder, self.sr_int, 'test'))
        save(meta, os.path.join(self.processed_folder, self.sr_int, 'meta'))
        return

    def download(self):
        url, md5 = self.file
        filename = os.path.basename(url)
        if not os.path.exists(os.path.join(self.raw_folder, filename)):
            download_url(url, os.path.join(self.raw_folder, filename), md5)
        extract_file(os.path.join(self.raw_folder, filename))
        return

    def __repr__(self):
        fmt_str = 'Dataset {}\nSize: {}\nRoot: {}\nSplit: {}\nTransforms: {}'.format(
            self.__class__.__name__, self.__len__(), self.root, self.split, self.transform.__repr__())
        return fmt_str

    def downsample(self):
        # Define paths and target sample rate
        input_dir = os.path.join(self.raw_folder, 'wav48_silence_trimmed')
        output_dir = os.path.join(self.raw_folder, 'wav' + self.sr_int)

        # Ensure the output directory exists
        os.makedirs(output_dir, exist_ok=True)

        # Collect all files to process
        files_to_process = []
        for root, dirs, files in os.walk(input_dir):
            for file in files:
                if file.endswith("_mic1.flac"):
                    files_to_process.append((root, file))

        # Process files with a progress bar
        for root, file in tqdm(files_to_process, desc="Processing files", unit="file"):
            # Construct full file path
            file_path = os.path.join(root, file)

            # Load the audio file using librosa
            audio, sr = librosa.load(file_path, sr=None)

            # Downsample the audio file to the target sample rate
            audio_resampled = librosa.resample(audio, orig_sr=sr, target_sr=self.sample_rate)

            # Remove '_mic1' from the file name and change extension to .wav
            new_file_name = file.replace('_mic1.flac', '.wav')

            # Construct the output file path
            relative_path = os.path.relpath(file_path, input_dir)
            relative_dir = os.path.dirname(relative_path)
            output_file_path = os.path.join(output_dir, relative_dir, new_file_name)
            output_file_dir = os.path.dirname(output_file_path)
            os.makedirs(output_file_dir, exist_ok=True)

            # Export the downsampled audio file as a .wav file using soundfile
            sf.write(output_file_path, audio_resampled, self.sample_rate)
        return

    def read_speaker_info(self):
        speaker_info_path = os.path.join(self.raw_folder, 'speaker-info.txt')
        speaker_info = {}
        with open(speaker_info_path, 'r') as file:
            lines = file.readlines()[1:]  # Skip the header
            for line in lines:
                parts = line.strip().split()
                speaker_id = parts[0]
                age = parts[1]
                gender = parts[2]
                accent = parts[3]
                region = parts[4] if len(parts) > 4 else ""
                comment = " ".join(parts[5:]) if len(parts) > 5 else ""
                speaker_info[speaker_id] = {
                    "age": age,
                    "gender": gender,
                    "accent": accent,
                    "region": region,
                    "comment": comment,
                }
        return speaker_info

    def make_data(self):
        # Define paths
        wav_dir = os.path.join(self.raw_folder, 'wav' + self.sr_int)
        speaker_info = self.read_speaker_info()

        files_to_process = []
        for root, dirs, files in os.walk(wav_dir):
            for file in files:
                if file.endswith(".wav"):
                    files_to_process.append((root, file))

        dataset = []
        id = []
        for root, file in tqdm(files_to_process, desc="Creating dataset", unit="file"):
            file_path = os.path.join(root, file)
            file_name = os.path.basename(file)
            speaker_id = file_name.split("_")[0]
            id.append(speaker_id)
            speaker_meta = speaker_info.get(speaker_id, {})
            entry = {
                "id": speaker_id,
                "path": file_path,
                "meta": speaker_meta
            }
            dataset.append(entry)

        df = pd.DataFrame(dataset)
        unique_speakers = np.unique(id)
        print(unique_speakers)
        exit()
        speaker_to_idx = {speaker: idx for idx, speaker in enumerate(unique_speakers)}

        train_df, test_df = self.split_dataset(df)
        return train_df, test_df, speaker_to_idx

    def split_dataset(self, df, train_size=0.8, random_state=42):
        speakers = df['id'].unique()
        train_speakers, test_speakers = train_test_split(speakers, train_size=train_size, random_state=random_state)
        train_df = df[df['id'].isin(train_speakers)]
        test_df = df[df['id'].isin(test_speakers)]
        return train_df, test_df


class VCTKMel(VCTK):
    data_name = 'VCTKMel'

    def __init__(self, root, split, transform=None):
        super().__init__(root, split, transform)

    def __getitem__(self, idx):
        """
        What i want the code to do is as follows:
        
        - select target information
        - clip target audio to MAX_WAVE_LEN with random start position
        - clip reference audio to MAX_WAVE_LEN with another random start position
        """

        # get target information
        row = self.data.iloc[idx]

        og_audio, sr = librosa.load(row['path'], sr=cfg['sample_rate'])
        og_audio = torch.from_numpy(og_audio)

        speaker_id = self.speaker_to_idx[row['speaker_id']]

        # get reference information b-vae way --- ADHERE TO THESE COMMENTS
        ## get max wave len random section of audio 
        ## get another max wav len random section of audio -- this is ref audio
        if len(og_audio) > cfg['wav_length']:
            start_idx = np.random.randint(0, len(og_audio) - cfg['wav_length'])
            audio = og_audio[start_idx:start_idx + cfg['wav_length']]
            ref_start_idx = np.random.randint(0, len(og_audio) - cfg['wav_length'])
            ref_audio = og_audio[ref_start_idx:ref_start_idx + cfg['wav_length']]
            length = cfg['wav_length']
        else:
            audio = F.pad(og_audio, (0, cfg['wav_length'] - og_audio.shape[-1]), 'constant', cfg['audio_pad_id'])
            ref_audio = F.pad(og_audio, (0, cfg['wav_length'] - og_audio.shape[-1]), 'constant', cfg['audio_pad_id'])
            length = og_audio.shape[-1]

        # get reference information styletts way --- IGNORE THIS SECTION
        # ref_row = self.data[self.data['speaker_id'] == row['speaker_id']].sample(1).iloc[0]

        # ref_audio, sr = librosa.load(ref_row['path'], sr=SR)
        # ref_audio = torch.from_numpy(ref_audio)

        # ref_speaker_id = speaker_to_idx[ref_row['speaker_id']]

        mel_audio = librosa.feature.melspectrogram(y=audio.numpy(),
                                                   sr=cfg['sample_rate'],
                                                   n_fft=cfg['filter_length'],
                                                   hop_length=cfg["hop_length"],
                                                   win_length=cfg["win_length"],
                                                   fmin=cfg['mel_fmin'],
                                                   fmax=cfg['mel_fmax'],
                                                   n_mels=cfg["n_mel_channels"],
                                                   window=torch.hann_window)

        ref_mel_audio = librosa.feature.melspectrogram(y=ref_audio.numpy(),
                                                       sr=cfg['sample_rate'],
                                                       n_fft=cfg['filter_length'],
                                                       hop_length=cfg["hop_length"],
                                                       win_length=cfg["win_length"],
                                                       fmin=cfg['mel_fmin'],
                                                       fmax=cfg['mel_fmax'],
                                                       n_mels=cfg["n_mel_channels"],
                                                       window=torch.hann_window)

        mel_audio = torch.from_numpy(mel_audio)  # REVIEW HERE: switches between torch and np may be inefficient
        ref_mel_audio = torch.from_numpy(ref_mel_audio)
        mel_audio = mel_audio.unsqueeze(0)

        sample = {
            'data': mel_audio,
            'wav': audio,
            'target': torch.flatten(mel_audio),
            # 'length': length,
            'speaker_id': torch.tensor(speaker_id, dtype=torch.long),  # Numeric speaker ID
            'ref_audio': ref_mel_audio,
            # 'ref_speaker_id': torch.tensor(ref_speaker_id, dtype=torch.long),
        }

        if self.transform is not None:
            sample = self.transform(sample)

        return sample


class VCTKTime(VCTK):
    data_name = 'VCTKTime'

    def __init__(self, root, split, transform=None):
        super().__init__(root, split, transform)

    def __getitem__(self, idx):
        """
        What i want the code to do is as follows:
        
        - select target information
        - clip target audio to MAX_WAVE_LEN with random start position
        - clip refrence audio to MAX_WAVE_LEN with another random start position
        """

        # get target information
        row = self.data.iloc[idx]

        og_audio, sr = librosa.load(row['path'], sr=cfg['sample_rate'])
        og_audio = torch.from_numpy(og_audio)

        speaker_id = self.speaker_to_idx[row['speaker_id']]

        # get reference information b-vae way --- ADHERE TO THESE COMMENTS
        ## get max wave len random section of audio 
        ## get another max wav len random section of audio -- this is ref audio
        if len(og_audio) > cfg['wav_length']:
            start_idx = np.random.randint(0, len(og_audio) - cfg['wav_length'])
            audio = og_audio[start_idx:start_idx + cfg['wav_length']]
            ref_start_idx = np.random.randint(0, len(og_audio) - cfg['wav_length'])
            ref_audio = og_audio[ref_start_idx:ref_start_idx + cfg['wav_length']]
            length = cfg['wav_length']
        else:
            audio = F.pad(og_audio, (0, cfg['wav_length'] - og_audio.shape[-1]), 'constant', cfg['audio_pad_id'])
            ref_audio = F.pad(og_audio, (0, cfg['wav_length'] - og_audio.shape[-1]), 'constant', cfg['audio_pad_id'])
            length = og_audio.shape[-1]

        # get reference information styletts way --- IGNORE THIS SECTION
        # ref_row = self.data[self.data['speaker_id'] == row['speaker_id']].sample(1).iloc[0]

        # ref_audio, sr = librosa.load(ref_row['path'], sr=SR)
        # ref_audio = torch.from_numpy(ref_audio)

        # ref_speaker_id = speaker_to_idx[ref_row['speaker_id']]

        sample = {
            'data': audio,
            'target': audio,
            'length': length,
            'speaker_id': torch.tensor(speaker_id, dtype=torch.long),  # Numeric speaker ID
            'ref_audio': ref_audio,
            # 'ref_speaker_id': torch.tensor(ref_speaker_id, dtype=torch.long),
        }

        if self.transform is not None:
            sample = self.transform(sample)

        return sample

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
from scipy.signal import lfilter
from module import check_exists, save, load
from .utils import download_url, extract_file
from config import cfg


class VCTK(Dataset):
    def __init__(self, root, split, transform=None):
        self.root = root
        self.file = ('https://datashare.is.ed.ac.uk/bitstream/handle/10283/3443/VCTK-Corpus-0.92.zip', None)
        self.split = split
        self.transform = transform
        self.sample_rate = cfg['sample_rate']
        self.segment_seconds = cfg['segment_seconds']
        self.sr_int = str(int(self.sample_rate // 1e3))
        self.train_ratio = 0.9
        self.num_test_out = 10
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
        train_set, test_in_set, test_out_set, meta = self.make_data()
        save(train_set, os.path.join(self.processed_folder, self.sr_int, 'train'))
        save(test_in_set, os.path.join(self.processed_folder, self.sr_int, 'test_in'))
        save(test_out_set, os.path.join(self.processed_folder, self.sr_int, 'test_out'))
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

    # def speaker_file_paths(self, root_dir):
    #     speaker2filenames = defaultdict(lambda: [])
    #     for path in sorted(glob.glob(os.path.join(root_dir, "*/*"))):
    #         filename = path.strip().split("\\")[-1]  # "\\" for Windows, "/" for Linux
    #         speaker_id = get_speaker_id(filename)
    #         speaker2filenames[speaker_id].append(path)
    #     return speaker2filenames
    #
    # def get_speaker_id(self, filename):
    #     pattern = r'^p\d{3}_(\d{3})\.wav$'
    #     match = re.search(pattern, filename)
    #     if match:
    #         speaker_id = filename.split('_')[0]
    #     return speaker_id

    def make_data(self):
        wav_dir = os.path.join(self.raw_folder, 'wav' + self.sr_int)
        speaker_info = self.read_speaker_info()
        speaker_list = list(speaker_info.keys())
        train_speakers = speaker_list[:-self.num_test_out]
        test_out_speakers = speaker_list[-self.num_test_out:]
        speaker_to_idx = {speaker: idx for idx, speaker in enumerate(speaker_list)}
        speaker_split = {'train': train_speakers, 'test_out': test_out_speakers}

        def make_entry(path_, speaker_):
            speaker_id = speaker_to_idx.get(speaker_)
            speaker_meta = speaker_info.get(speaker_)
            entry = {
                "path": path_,
                "speaker_id": speaker_id,
                "speaker_meta": speaker_meta
            }
            return entry

        train_dataset = []
        test_in_dataset = []
        test_out_dataset = []
        for speaker in tqdm(os.listdir(wav_dir), desc="Creating dataset", unit="speaker"):
            files = os.listdir(os.path.join(wav_dir, speaker))
            valid_files = []
            for file in files:
                wav = load_wav(os.path.join(wav_dir, speaker, file), self.sample_rate)
                if len(wav) > self.sample_rate * self.segment_seconds:
                    valid_files.append(file)
            if speaker in train_speakers:
                num_train = int(len(valid_files) * self.train_ratio)
                train_files = valid_files[:num_train]
                test_in_files = valid_files[num_train:]
                for path in train_files:
                    train_dataset.append(make_entry(path, speaker))
                for path in test_in_files:
                    test_in_dataset.append(make_entry(path, speaker))
            else:
                for path in valid_files:
                    test_out_dataset.append(make_entry(path, speaker))
        # print(train_dataset)
        # print(test_in_dataset)
        # print(test_out_dataset)
        # exit()

        # for root, dirs, files in os.walk(wav_dir):
        #     for file in files:
        #         if file.endswith(".wav"):
        #             path = os.path.join(root, file)
        #             speaker = file.split('_')[0]
        #             if speaker in train_speakers:
        #                 train_files.append((file, speaker))

        # train_dataset = []
        # test_in_dataset = []
        # test_out_dataset = []
        # for root, speaker, file in tqdm(files_to_process, desc="Creating dataset", unit="file"):
        #     if speaker in train_speakers:
        #         file_path = os.path.join(root, file)
        #
        #     train_dataset.append(entry)

        # df = pd.DataFrame(dataset)
        # unique_speakers = np.unique(id)
        # df['idx'] = df['id'].map(speaker_to_idx)
        meta = {'speaker_to_idx': speaker_to_idx, 'speaker_split': speaker_split}
        return train_dataset, test_in_dataset, test_out_dataset, meta

    # def split_dataset(self, df, train_size=90, random_state=42):
    #     speakers = df['id'].unique()
    #     train_speakers, test_speakers = train_test_split(speakers, train_size=train_size, random_state=random_state)
    #     train_df = df[df['id'].isin(train_speakers)]
    #     test_df = df[df['id'].isin(test_speakers)]
    #     return train_df, test_df


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
        og_audio = load_wav(row['path'], self.sample_rate)
        og_audio = torch.from_numpy(og_audio)
        speaker_id = row['idx']

        if len(og_audio) > cfg['wav_length']:
            start_idx = np.random.randint(0, len(og_audio) - cfg['wav_length'])
            audio = og_audio[start_idx:start_idx + cfg['wav_length']]
            # length = cfg['wav_length']
        else:
            audio = F.pad(og_audio, (0, cfg['wav_length'] - og_audio.shape[-1]), 'constant', cfg['audio_pad_id'])
            # ref_audio = F.pad(og_audio, (0, cfg['wav_length'] - og_audio.shape[-1]), 'constant', cfg['audio_pad_id'])
            # length = og_audio.shape[-1]

        # # get reference information b-vae way --- ADHERE TO THESE COMMENTS
        # ## get max wave len random section of audio
        # ## get another max wav len random section of audio -- this is ref audio
        # if len(og_audio) > cfg['wav_length']:
        #     start_idx = np.random.randint(0, len(og_audio) - cfg['wav_length'])
        #     audio = og_audio[start_idx:start_idx + cfg['wav_length']]
        #     ref_start_idx = np.random.randint(0, len(og_audio) - cfg['wav_length'])
        #     ref_audio = og_audio[ref_start_idx:ref_start_idx + cfg['wav_length']]
        #     length = cfg['wav_length']
        # else:
        #     audio = F.pad(og_audio, (0, cfg['wav_length'] - og_audio.shape[-1]), 'constant', cfg['audio_pad_id'])
        #     ref_audio = F.pad(og_audio, (0, cfg['wav_length'] - og_audio.shape[-1]), 'constant', cfg['audio_pad_id'])
        #     length = og_audio.shape[-1]

        # get reference information styletts way --- IGNORE THIS SECTION
        # ref_row = self.data[self.data['speaker_id'] == row['speaker_id']].sample(1).iloc[0]
        # ref_audio, sr = librosa.load(ref_row['path'], sr=SR)
        # ref_audio = torch.from_numpy(ref_audio)
        # ref_speaker_id = speaker_to_idx[ref_row['speaker_id']]

        # mel_audio = librosa.feature.melspectrogram(y=audio.numpy(),
        #                                            sr=cfg['sample_rate'],
        #                                            n_fft=cfg['n_fft'],
        #                                            hop_length=cfg["hop_length"],
        #                                            win_length=cfg["win_length"],
        #                                            fmin=cfg['fmin'],
        #                                            fmax=cfg['fmax'],
        #                                            n_mels=cfg["n_mels"],
        #                                            window=torch.hann_window)

        mel_audio = log_mel_spectrogram(y=audio.numpy(),
                                        preemph=cfg['preemph'],
                                        sample_rate=cfg['sample_rate'],
                                        n_mels=cfg['n_mels'],
                                        n_fft=cfg['n_fft'],
                                        hop_length=cfg['hop_length'],
                                        win_length=cfg['win_length'],
                                        fmin=cfg['fmin'],
                                        fmax=cfg['fmax'])
        # ref_mel_audio = librosa.feature.melspectrogram(y=ref_audio.numpy(),
        #                                                sr=cfg['sample_rate'],
        #                                                n_fft=cfg['n_fft'],
        #                                                hop_length=cfg["hop_length"],
        #                                                win_length=cfg["win_length"],
        #                                                fmin=cfg['fmin'],
        #                                                fmax=cfg['fmax'],
        #                                                n_mels=cfg["n_mels"],
        #                                                window=torch.hann_window)

        mel_audio = torch.from_numpy(mel_audio)  # REVIEW HERE: switches between torch and np may be inefficient
        # ref_mel_audio = torch.from_numpy(ref_mel_audio)
        # mel_audio = mel_audio.unsqueeze(0)

        sample = {
            'data': mel_audio,
            # 'wav': audio,
            # 'target': torch.flatten(mel_audio),
            'target': torch.flatten(mel_audio),
            # 'length': length,
            'speaker_id': torch.tensor(speaker_id, dtype=torch.long),  # Numeric speaker ID
            # 'ref_audio': ref_mel_audio,
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
        og_audio = load_wav(row['path'], cfg['sample_rate'])
        og_audio = torch.from_numpy(og_audio)
        speaker_id = row['idx']

        if len(og_audio) > cfg['wav_length']:
            start_idx = np.random.randint(0, len(og_audio) - cfg['wav_length'])
            audio = og_audio[start_idx:start_idx + cfg['wav_length']]
            # length = cfg['wav_length']
        else:
            audio = F.pad(og_audio, (0, cfg['wav_length'] - og_audio.shape[-1]), 'constant', cfg['audio_pad_id'])
            # ref_audio = F.pad(og_audio, (0, cfg['wav_length'] - og_audio.shape[-1]), 'constant', cfg['audio_pad_id'])
            # length = og_audio.shape[-1]

        # get reference information b-vae way --- ADHERE TO THESE COMMENTS # TODO: this nees check
        ## get max wave len random section of audio 
        ## get another max wav len random section of audio -- this is ref audio
        # if len(og_audio) > cfg['wav_length']:
        #     start_idx = np.random.randint(0, len(og_audio) - cfg['wav_length'])
        #     audio = og_audio[start_idx:start_idx + cfg['wav_length']]
        #     ref_start_idx = np.random.randint(0, len(og_audio) - cfg['wav_length'])
        #     ref_audio = og_audio[ref_start_idx:ref_start_idx + cfg['wav_length']]
        #     length = cfg['wav_length']
        # else:
        #     audio = F.pad(og_audio, (0, cfg['wav_length'] - og_audio.shape[-1]), 'constant', cfg['audio_pad_id'])
        #     ref_audio = F.pad(og_audio, (0, cfg['wav_length'] - og_audio.shape[-1]), 'constant', cfg['audio_pad_id'])
        #     length = og_audio.shape[-1]

        # get reference information styletts way --- IGNORE THIS SECTION
        # ref_row = self.data[self.data['speaker_id'] == row['speaker_id']].sample(1).iloc[0]
        # ref_audio, sr = librosa.load(ref_row['path'], sr=SR)
        # ref_audio = torch.from_numpy(ref_audio)
        # ref_speaker_id = speaker_to_idx[ref_row['speaker_id']]

        sample = {
            'data': audio,
            'target': audio,
            # 'length': length,
            'speaker_id': torch.tensor(speaker_id, dtype=torch.long),  # Numeric speaker ID
            # 'ref_audio': ref_audio,
            # 'ref_speaker_id': torch.tensor(ref_speaker_id, dtype=torch.long),
        }
        if self.transform is not None:
            sample = self.transform(sample)

        return sample


def load_wav(audio_path, sample_rate, trim=False):
    """Load and preprocess waveform."""
    wav, _ = librosa.load(audio_path, sr=sample_rate)
    # wav = wav / (np.abs(wav).max() + 1e-6)  # normalization is not needed here
    if trim:
        _, (start_frame, end_frame) = librosa.effects.trim(
            wav, top_db=25, frame_length=512, hop_length=128
        )
        start_frame = max(0, start_frame - 0.1 * sample_rate)
        end_frame = min(len(wav), end_frame + 0.1 * sample_rate)

        start = int(start_frame)
        end = int(end_frame)
        if end - start > 1000:
            wav = wav[start:end]
    return wav


def log_mel_spectrogram(
        y: np.ndarray,
        preemph: float,
        sample_rate: int,
        n_mels: int,
        n_fft: int,
        hop_length: int,
        win_length: int,
        fmin: int,
        fmax: int,
) -> np.ndarray:
    """Create a log Mel spectrogram from a raw audio signal."""
    if preemph > 0:
        y = lfilter([1, -preemph], [1], y)
    magnitude = np.abs(
        librosa.stft(y=y, n_fft=n_fft, hop_length=hop_length, win_length=win_length)
    )
    mel_fb = librosa.filters.mel(sr=sample_rate, n_fft=n_fft, n_mels=n_mels, fmin=fmin, fmax=fmax)
    mel_spec = np.dot(mel_fb, magnitude)
    log_mel_spec = np.log(mel_spec + 1e-9).T
    return log_mel_spec  # shape(T, n_mels)

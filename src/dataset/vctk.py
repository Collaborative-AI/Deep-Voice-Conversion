### IMPORTS ###
import os
import requests
import zipfile
import os
from tqdm import tqdm
import librosa
import soundfile as sf
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
import torch
from torch.utils.data import Dataset
import torch.nn.functional as F
import numpy as np
import os
import torch
from torch.utils.data import Dataset
from module import check_exists, makedir_exist_ok, save, load
from config import cfg

class VCTKTime(Dataset):
    data_name = 'VCTKTime'

    def __init__(self, root, split, transform=None):
        
        assert split in ['train', 'test', 'val']
        
        self.root = os.path.expanduser(root)
        self.file = 'https://datashare.is.ed.ac.uk/bitstream/handle/10283/3443/VCTK-Corpus-0.92.zip'
        self.split = split
        self.transform = transform
        self.sample_rate = cfg['sample_rate']
        self.sr_int = str(int(self.sample_rate // 1e3))
        
        ### CREATE TOKENIZER
        if not check_exists(self.processed_folder):
            print("Processing...")
            self.process()
        self.data = pd.read_pickle(os.path.join(self.processed_folder, self.split + self.sr_int))
        self.other = {}
        
        
        self.speaker_to_idx = load(os.path.join(self.processed_folder, 'meta' + self.sr_int))
            
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
        
        # get refrence information b-vae way --- ADHERE TO THESE COMMENTS
        ## get max wave len random section of audio 
        ## get another max wav len random section of audio -- this is ref audio
        if len(og_audio) > cfg['segment_length']:
            start_idx = np.random.randint(0, len(og_audio) - cfg['segment_length'])
            audio = og_audio[start_idx:start_idx + cfg['segment_length']]
            ref_start_idx = np.random.randint(0, len(og_audio) - cfg['segment_length'])
            ref_audio = og_audio[ref_start_idx:ref_start_idx + cfg['segment_length']]
            length = cfg['segment_length']
        else:
            audio = F.pad(og_audio, (0, cfg['segment_length'] - og_audio.shape[-1]), 'constant', cfg['audio_pad_id'])
            ref_audio = F.pad(og_audio, (0, cfg['segment_length'] - og_audio.shape[-1]), 'constant', cfg['audio_pad_id'])
            length = og_audio.shape[-1]
        
        # get refrence information styletts way --- IGNORE THIS SECTION
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
        
        os.makedirs(self.processed_folder, exist_ok=True)
        
        train_set, val_set, test_set = self.make_data()
        train_set.to_pickle(os.path.join(self.processed_folder, 'train'+self.sr_int))
        val_set.to_pickle(os.path.join(self.processed_folder, 'val'+self.sr_int))
        test_set.to_pickle(os.path.join(self.processed_folder, 'test'+self.sr_int))
        
        unique_speakers = (train_set['speaker_id'].unique().tolist() + 
                            val_set['speaker_id'].unique().tolist() + 
                            test_set['speaker_id'].unique().tolist())

        speaker_to_idx = {speaker: idx for idx, speaker in enumerate(unique_speakers)}
        save(speaker_to_idx, os.path.join(self.processed_folder, 'meta'+self.sr_int))
        return
        

    def download(self):
        makedir_exist_ok(self.raw_folder)
        # Define the URL and the target paths
        url = self.file
        data_dir = self.raw_folder
        download_path = os.path.join(data_dir, 'VCTK-Corpus-0.92.zip')
        extract_path = os.path.join(data_dir, 'VCTK')

        # Ensure the data directory exists
        os.makedirs(data_dir, exist_ok=True)

        # Download the dataset
        print(f"Downloading VCTK dataset from {url}. This may take a long time...")
        response = requests.get(url, stream=True)
        with open(download_path, 'wb') as file:
            for chunk in response.iter_content(chunk_size=8192):
                file.write(chunk)
        print("Download complete.")

        # Unzip the file
        print(f"Extracting {download_path} to {data_dir}...")
        with zipfile.ZipFile(download_path, 'r') as zip_ref:
            zip_ref.extractall(data_dir)
        print("Extraction complete.")

        # Find the extracted folder and rename it to "VCTK"
        extracted_folder_name = 'VCTK-Corpus-0.92'
        original_extract_path = os.path.join(data_dir, extracted_folder_name)

        if os.path.exists(original_extract_path):
            os.rename(original_extract_path, extract_path)
            print(f"Renamed {original_extract_path} to {extract_path}")
        else:
            print(f"Expected extracted folder {original_extract_path} not found")

        print(f"VCTK dataset is ready at {extract_path}")
        return

    def __repr__(self):
        fmt_str = 'Dataset {}\nSize: {}\nRoot: {}\nSplit: {}\nTransforms: {}'.format(
            self.__class__.__name__, self.__len__(), self.root, self.split, self.transform.__repr__())
        return fmt_str
        
    def downsample(self):
        # Define paths and target sample rate
        input_dir = os.path.join(self.raw_folder, 'wav48_silence_trimmed')
        output_dir = os.path.join(self.raw_folder, 'wav'+self.sr_int)
        
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
        wav_dir = os.path.join(self.raw_folder, 'wav'+self.sr_int)
        speaker_info = self.read_speaker_info()
        
        dataset = []

        files_to_process = []
        for root, dirs, files in os.walk(wav_dir):
            for file in files:
                if file.endswith(".wav"):
                    files_to_process.append((root, file))
        

        for root, file in tqdm(files_to_process, desc="Creating dataset", unit="file"):
            file_path = os.path.join(root, file)
            file_name = os.path.basename(file)
            speaker_id = file_name.split("_")[0]
            
            speaker_meta = speaker_info.get(speaker_id, {})
            entry = {
                "speaker_id": speaker_id,
                "path": file_path
            }
            dataset.append(entry)
            
        df = pd.DataFrame(dataset)
        train_df, val_df, test_df = self.split_dataset(df)        
        
        return train_df, val_df, test_df
    
    def split_dataset(self, df, train_size=0.7, val_size=0.15, test_size=0.15, random_state=42):
        # Ensure the split proportions sum to 1
        assert train_size + val_size + test_size == 1.0, "Train, validation, and test sizes must sum to 1.0"
        
        # Get unique speakers
        speakers = df['speaker_id'].unique()
        
        # Split speakers into train and temp (val + test)
        train_speakers, temp_speakers = train_test_split(speakers, train_size=train_size, random_state=random_state)
        
        # Calculate the proportion for validation in the temp split
        val_proportion = val_size / (val_size + test_size)
        
        # Split temp_speakers into validation and test sets
        val_speakers, test_speakers = train_test_split(temp_speakers, train_size=val_proportion, random_state=random_state)
        
        # Assign entries to the respective sets
        train_df = df[df['speaker_id'].isin(train_speakers)]
        val_df = df[df['speaker_id'].isin(val_speakers)]
        test_df = df[df['speaker_id'].isin(test_speakers)]
        
        return train_df, val_df, test_df
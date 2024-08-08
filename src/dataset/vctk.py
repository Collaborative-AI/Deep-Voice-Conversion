### IMPORTS ###
import os
import requests
import zipfile
import shutil
import os
from tqdm import tqdm
import librosa
import soundfile as sf
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
import numpy as np
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer
import gc
from torch.utils.data import DataLoader
from torch.utils.data.dataloader import default_collate
from torch.nn.utils.rnn import pad_sequence
import torch.nn as nn
import torch.nn.functional as F
import torchaudio.transforms as T
# import codecs
import numpy as np
import os
import torch
from PIL import Image
from torch.utils.data import Dataset
from module import check_exists, makedir_exist_ok, save, load
from .utils import download_url, extract_file, make_classes_counts

SR = int(8e3)
SRkHz = int(SR//1e3)
MAX_CLIP_SECS = 2
MAX_WAV_LEN = int(SR*MAX_CLIP_SECS)

AUDIO_PAD_ID = 2.0

class VCTK(Dataset):
    data_name = 'VCTK'

    def __init__(self, root, split, transform=None):
        
        assert split in ['train', 'test', 'val']
        
        self.root = os.path.expanduser(root)
        self.file = 'https://datashare.is.ed.ac.uk/bitstream/handle/10283/3443/VCTK-Corpus-0.92.zip'
        self.split = split
        self.transform = transform
        self.sample_rate = SR
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
        
        og_audio, sr = librosa.load(row['path'], sr=SR)
        og_audio = torch.from_numpy(og_audio)
        
        speaker_id = self.speaker_to_idx[row['speaker_id']]
        
        # get refrence information b-vae way --- ADHERE TO THESE COMMENTS
        ## get max wave len random section of audio 
        ## get another max wav len random section of audio -- this is ref audio
        if len(og_audio) > MAX_WAV_LEN:
            start_idx = np.random.randint(0, len(og_audio) - MAX_WAV_LEN)
            audio = og_audio[start_idx:start_idx + MAX_WAV_LEN]
            ref_start_idx = np.random.randint(0, len(og_audio) - MAX_WAV_LEN)
            ref_audio = og_audio[ref_start_idx:ref_start_idx + MAX_WAV_LEN]
            length = MAX_WAV_LEN
        else:
            audio = torch.nn.functional.pad(og_audio, (0, MAX_WAV_LEN - og_audio.shape[-1]), 'constant', AUDIO_PAD_ID)
            ref_audio = torch.nn.functional.pad(og_audio, (0, MAX_WAV_LEN - og_audio.shape[-1]), 'constant', AUDIO_PAD_ID)
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

    # def make_data(self):
    #     train_data = read_image_file(os.path.join(self.raw_folder, 'train-images-idx3-ubyte'))
    #     test_data = read_image_file(os.path.join(self.raw_folder, 't10k-images-idx3-ubyte'))
    #     train_target = read_label_file(os.path.join(self.raw_folder, 'train-labels-idx1-ubyte'))
    #     test_target = read_label_file(os.path.join(self.raw_folder, 't10k-labels-idx1-ubyte'))
    #     train_id, test_id = np.arange(len(train_data)).astype(np.int64), np.arange(len(test_data)).astype(np.int64)
    #     classes = list(map(str, list(range(10))))
    #     classes_to_labels = {classes[i]: i for i in range(len(classes))}
    #     target_size = len(classes)
    #     return (train_id, train_data, train_target), (test_id, test_data, test_target), (classes_to_labels, target_size)
    
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
        txt_dir = os.path.join(self.raw_folder, 'txt')
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

        # train_df.to_csv('train_{}.csv'.format(int(self.sample_rate//1e3)))
        # val_df.to_csv('val_{}.csv'.format(int(self.sample_rate//1e3)))
        # test_df.to_csv('test_{}.csv'.format(int(self.sample_rate//1e3)))
        
        
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


# import codecs
# import numpy as np
# import os
# import torch
# from PIL import Image
# from torch.utils.data import Dataset
# from module import check_exists, makedir_exist_ok, save, load
# from .utils import download_url, extract_file, make_classes_counts, get_data_path_list
# import torchaudio
# import random
# import re
# import librosa
# import soundfile as sf
# import random
# from config import cfg

# _pad = "$"
# _punctuation = ';:,.!?¡¿—…"«»“” '
# _letters = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz'
# _letters_ipa = "ɑɐɒæɓʙβɔɕçɗɖðʤəɘɚɛɜɝɞɟʄɡɠɢʛɦɧħɥʜɨɪʝɭɬɫɮʟɱɯɰŋɳɲɴøɵɸθœɶʘɹɺɾɻʀʁɽʂʃʈʧʉʊʋⱱʌɣɤʍχʎʏʑʐʒʔʡʕʢǀǁǂǃˈˌːˑʼʴʰʱʲʷˠˤ˞↓↑→↗↘'̩'ᵻ"

# # Export all symbols:
# symbols = [_pad] + list(_punctuation) + list(_letters) + list(_letters_ipa)

# fixed_wave_length = int(4e4)
# cfg['wave_len'] = fixed_wave_length

# dicts = {}
# for i in range(len((symbols))):
#     dicts[symbols[i]] = i

# class TextCleaner:
#     def __init__(self, dummy=None):
#         self.word_index_dictionary = dicts
#     def __call__(self, text):
#         indexes = []
#         for char in text:
#             try:
#                 indexes.append(self.word_index_dictionary[char])
#             except KeyError:
#                 print(text)
#         return indexes

# np.random.seed(1)
# SPECT_PARAMS = {
#     "n_fft": 2048,
#     "win_length": 1200,
#     "hop_length": 300
# }
# MEL_PARAMS = {
#     "n_mels": 80,
# }

# to_mel = torchaudio.transforms.MelSpectrogram(
#     n_mels=80, n_fft=2048, win_length=1200, hop_length=300)
# mean, std = -4, 4

# def preprocess(wave):
#     wave_tensor = torch.from_numpy(wave).float()
#     mel_tensor = to_mel(wave_tensor)
#     mel_tensor = (torch.log(1e-5 + mel_tensor.unsqueeze(0)) - mean) / std
#     return mel_tensor

# # NEEDS TO BE ADJUSTED 
# class VCTK(Dataset):
#     data_name = 'VCTK'
#     file = [('https://datashare.ed.ac.uk/download/DS_10283_3443.zip')]
    
#     def __init__(self,
#                  root,
#                  split,
#                  transform=None,
#                  sr=8000,
#                  data_augmentation=False,
#                  validation=False,
#                  ):

#         spect_params = SPECT_PARAMS
#         mel_params = MEL_PARAMS

#         self.root = os.path.expanduser(root)
#         self.split = split
#         self.transform = transform
#         if not check_exists(self.raw_folder):
#             raise Exception("No data found") # update this to auto process/download raw data later
#             self.process()
        
#         if sr/1000 != sr//1000:
#             raise Exception("signal rate is not a multiple of 1000")
#         if split not in ['train', 'test', 'val']:
#             raise Exception(f"Expected value in ['train', 'test', 'val']. Got '{split}'")
#         data_root = os.path.join(self.processed_folder, f"{split}_list{sr//1000}.txt")
        
        
#         data_list = get_data_path_list(data_root) 
        
#         _data_list = [l[:-1].split('|') for l in data_list]
#         for l in _data_list:
#             l[0] = "data/VCTK/" + l[0]
#         self.data_list = [data if len(data) == 3 else (*data, 0) for data in _data_list]
#         self.text_cleaner = TextCleaner()
#         self.sr = sr

#         self.to_melspec = torchaudio.transforms.MelSpectrogram(**MEL_PARAMS)

#         self.mean, self.std = -4, 4
#         self.data_augmentation = data_augmentation and (not validation)
#         self.max_mel_length = 192
#         self.id = [int(''.join(re.search(r'(\d+)_(\d+)\.wav$', data[0]).groups())) for data in self.data_list] # double check this for errors
#         self.data = self.data_list
#         self.other = {}

#     # def __init__(self, root, split, transform=None):
#     #     self.root = os.path.expanduser(root)
#     #     self.split = split
#     #     self.transform = transform
#     #     # MAKE FILE PATH DATALOADER
#     #     if not check_exists(self.processed_folder):
#     #         raise Exception("No data found") # update this to auto process/download later
#     #         self.process()
#     #     self.id, self.data, self.target = load(os.path.join(self.processed_folder, self.split))
#     #     self.other = {}
#     #     # NO TARGETS NECESSARY
#     #     self.classes_counts = make_classes_counts(self.target)
#     #     self.classes_to_labels, self.target_size = load(os.path.join(self.processed_folder, 'meta'))

#     # def __getitem__(self, index):
#     #     # CAN USE STYLETTS GET ITEM FROM MELDATASET.PY
#     #     id, data, target = torch.tensor(self.id[index]), Image.fromarray(self.data[index], mode='L'), torch.tensor(
#     #         self.target[index])
#     #     input = {'id': id, 'data': data, 'target': target}
#     #     other = {k: torch.tensor(self.other[k][index]) for k in self.other}
#     #     input = {**input, **other}
#         # if self.transform is not None:
#         #     input = self.transform(input)
#         # return input
    
#     def __getitem__(self, idx):
#         data = self.data_list[idx]
#         path = data[0]
        
#         wave, text_tensor, speaker_id = self._load_tensor(data)
        
#         # mel_tensor = preprocess(wave).squeeze() ****
        
#         # mel_tensor = mel_tensor.squeeze()     MEL STUFF. LEAVE HERE
#         # length_feature = mel_tensor.size(1)   ****
#         # mel_tensor = mel_tensor[:, :(length_feature - length_feature % 2)] ****
        
#         ref_data = random.choice(self.data_list)
#         # ref_wave, ref_mel_tensor, ref_label = self._load_data(ref_data) ****
#         ref_wave, ref_label = self._load_data(ref_data)
        
        
#         # return wave, speaker_id, acoustic_feature, text_tensor, ref_mel_tensor, ref_label, path # RETURN A DICTIONARY CONTAINING EACH VALUE AT A KEY
        
#         id = torch.tensor(self.id[idx])
#         # data = (wave, speaker_id, mel_tensor, text_tensor, ref_mel_tensor, ref_label, path)
#         # input = {
#         #         'id':  id, 'wave': wave, 'speaker_id': target, 
#         #         "mel_tensor": mel_tensor, "text_tensor": text_tensor,
#         #         "ref_wave": ref_wave, "ref_mel_tensor": ref_mel_tensor, 
#         #         "ref_speaker_id": torch.tensor(ref_label)
#         #     } 
#         input = {
#                 'id':  id, 'data': wave, 
#                 "text_tensor": text_tensor,
#                 'target': wave, "ref_wave": ref_wave, 
#                 "ref_speaker_id": torch.tensor(ref_label)
#             } 
#         other = {k: torch.tensor(self.other[k][idx]) for k in self.other}
#         input = {**input, **other}
#         # if self.transform is not None:
#         #     input = self.transform(input['data'])
#         return input
            
#     # def _load_data(self, data):     ****
#     #     wave, text_tensor, speaker_id = self._load_tensor(data)
#     #     mel_tensor = preprocess(wave).squeeze()
#     #     mel_length = mel_tensor.size(1) 
#     #     if mel_length > self.max_mel_length:
#     #         random_start = np.random.randint(0, mel_length - self.max_mel_length)
#     #         mel_tensor = mel_tensor[:, random_start:random_start + self.max_mel_length]

#     #     return wave, mel_tensor, speaker_id ****
    
#     def _load_data(self, data):
#         wave, text_tensor, speaker_id = self._load_tensor(data)
#         return wave, speaker_id

#     def __len__(self):
#         return len(self.data)

#     @property
#     def processed_folder(self):
#         return os.path.join(self.root, 'processed')

#     @property
#     def raw_folder(self):
#         return os.path.join(self.root, 'raw')

#     def process(self): # need to implement for processing
#         raise NotImplementedError("Process function has not yet been implemented")
#         if not check_exists(self.raw_folder):
#             self.download()
#         train_set, test_set = self.make_data()
#         save(train_set, os.path.join(self.processed_folder, 'train'))
#         save(test_set, os.path.join(self.processed_folder, 'test'))
#         return

#     def _load_tensor(self, data):
#         wave_path, text, speaker_id = data
#         speaker_id = int(speaker_id)
#         wave, sr = sf.read(wave_path)
#         if wave.shape[-1] == 2:
#             wave = wave[:, 0].squeeze()
#         # if sr != 24000:
#         #     wave = librosa.resample(wave, sr, 24000)
#         #     print(wave_path, sr)
           
#         wave, index = librosa.effects.trim(wave, top_db=30)
#         # wave = np.concatenate([np.zeros([5000]), wave, np.zeros([5000])], axis=0) # is this necessary??? 
#         # fixed_wave_length = int(3e4)
        
#         size = wave.size
#         if size > fixed_wave_length:
#             start = random.randint(0, size - fixed_wave_length-1) # minus one for indexing from zeros
#             end =  start + fixed_wave_length
#             wave = wave[start:end]
#         elif size < fixed_wave_length:
#             wave = pad_waveform(wave, fixed_wave_length)
        
#         # print()
#         # print("og size", size)
#         # print("new size", (wave.size))
#         # print()
                
#         text = self.text_cleaner(text)
        
#         text.insert(0, 0)
#         text.append(0)
        
#         text = torch.LongTensor(text)
#         wave = torch.tensor(wave, dtype=torch.float32)

#         return wave, text, speaker_id

#     def download(self):
#         raise NotImplementedError
#         makedir_exist_ok(self.raw_folder)
#         for (url, md5) in self.file:
#             filename = os.path.basename(url)
#             download_url(url, os.path.join(self.raw_folder, filename), md5)
#             extract_file(os.path.join(self.raw_folder, filename))
#         return
#     def download(self):  
#         # Define the URL and the target paths
#         makedir_exist_ok(self.raw_folder)
#         url = 'https://datashare.is.ed.ac.uk/bitstream/handle/10283/3443/VCTK-Corpus-0.92.zip'
#         data_dir = './data/VCTK/raw'
#         download_path = os.path.join(data_dir, 'VCTK-Corpus-0.92.zip')
#         extract_path = os.path.join(data_dir, 'VCTK')

#         # Ensure the data directory exists
#         os.makedirs(data_dir, exist_ok=True)

#         # Download the dataset
#         print(f"Downloading VCTK dataset from {url}...")
#         response = requests.get(url, stream=True)
#         with open(download_path, 'wb') as file:
#             for chunk in response.iter_content(chunk_size=8192):
#                 file.write(chunk)
#         print("Download complete.")

#         # Unzip the file
#         print(f"Extracting {download_path} to {data_dir}...")
#         with zipfile.ZipFile(download_path, 'r') as zip_ref:
#             zip_ref.extractall(data_dir)
#         print("Extraction complete.")

#         # Find the extracted folder and rename it to "VCTK"
#         extracted_folder_name = 'VCTK-Corpus-0.92'
#         original_extract_path = os.path.join(data_dir, extracted_folder_name)

#         if os.path.exists(original_extract_path):
#             os.rename(original_extract_path, extract_path)
#             print(f"Renamed {original_extract_path} to {extract_path}")
#         else:
#             print(f"Expected extracted folder {original_extract_path} not found")

#         print(f"VCTK dataset is ready at {extract_path}")

#         def __repr__(self):
#             fmt_str = 'Dataset {}\nSize: {}\nRoot: {}\nSplit: {}\nTransforms: {}'.format(
#                 self.__class__.__name__, self.__len__(), self.root, self.split, self.transform.__repr__())
#             return fmt_str

#     def make_data(self):
#         raise NotImplementedError
#         train_data = read_image_file(os.path.join(self.raw_folder, 'train-images-idx3-ubyte'))
#         test_data = read_image_file(os.path.join(self.raw_folder, 't10k-images-idx3-ubyte'))
#         train_target = read_label_file(os.path.join(self.raw_folder, 'train-labels-idx1-ubyte'))
#         test_target = read_label_file(os.path.join(self.raw_folder, 't10k-labels-idx1-ubyte'))
#         train_id, test_id = np.arange(len(train_data)).astype(np.int64), np.arange(len(test_data)).astype(np.int64)
#         # classes = list(map(str, list(range(10))))
#         # classes_to_labels = {classes[i]: i for i in range(len(classes))}
#         # target_size = len(classes)
#         return (train_id, train_data, train_target), (test_id, test_data, test_target)

# def get_int(b):
#     return int(codecs.encode(b, 'hex'), 16)


# # def read_image_file(path):
# #     with open(path, 'rb') as f:
# #         data = f.read()
# #         assert get_int(data[:4]) == 2051
# #         length = get_int(data[4:8])
# #         num_rows = get_int(data[8:12])
# #         num_cols = get_int(data[12:16])
# #         parsed = np.frombuffer(data, dtype=np.uint8, offset=16).reshape((length, num_rows, num_cols))
# #         return parsed


# # def read_label_file(path):
# #     with open(path, 'rb') as f:
# #         data = f.read()
# #         assert get_int(data[:4]) == 2049
# #         length = get_int(data[4:8])
# #         parsed = np.frombuffer(data, dtype=np.uint8, offset=8).reshape(length).astype(np.int64)
# #         return parsed

# def pad_waveform(waveform, target_length):
#     pad_length = target_length-len(waveform)
#     waveform = np.concatenate((waveform, np.zeros(pad_length)), axis=0)
#     return waveform


# # def pad_mel_spectrogram(mel_spectrogram, target_length):
# #     # Pad the Mel spectrogram to the target time length
# #     current_time_frames = mel_spectrogram.shape[-1]
# #     if current_time_frames < target_length:
# #         pad_amount = target_length - current_time_frames
# #         mel_spectrogram = torch.nn.functional.pad(mel_spectrogram, (0, pad_amount), mode='constant', value=0)
# #     return mel_spectrogram

# def pad_mel_spectrogram(mel_spectrogram, target_length):
#     # Calculate the padding amount
#     current_length = mel_spectrogram.shape[1]
#     pad_width = target_length - current_length
#     if pad_width > 0:
#         # Pad Mel spectrogram with zeros at the end along the time axis
#         mel_spectrogram = np.pad(mel_spectrogram, ((0, 0), (0, pad_width)), mode='constant')
#     return mel_spectrogram

# def create_padding_mask(array):
#     mask = array != 0
#     return mask
### IMPORTS ###
# import os
# import requests
# import zipfile
# import shutil
# import os
# from tqdm import tqdm
# import librosa
# import soundfile as sf
# import pandas as pd
# import pickle
# from sklearn.model_selection import train_test_split
# import numpy as np
# import torch
# from torch.utils.data import Dataset
# from transformers import AutoTokenizer
# import sentencepiece
# import gc
# from torch.utils.data import DataLoader
# from torch.utils.data.dataloader import default_collate
# from torch.nn.utils.rnn import pad_sequence
# import torch.nn as nn
# import torch.nn.functional as F
# import torchaudio.transforms as T
# import codecs
# import numpy as np
# import os
# import torch
# from PIL import Image
# from torch.utils.data import Dataset
# from module import check_exists, makedir_exist_ok, save, load
# from .utils import download_url, extract_file, make_classes_counts

# SR = int(8e3)
# SRkHz = int(SR//1e3)
# VOCAB_SIZE = int(4e3)
# MAX_CLIP_SECS = 2
# MAX_WAV_LEN = int(SR*MAX_CLIP_SECS)

# PAD_ID = 0
# BOS_ID = 1
# EOS_ID = 2
# UNK_ID = 3

# AUDIO_PAD_ID = -2.0

# class VCTK(Dataset):
#     data_name = 'VCTK'

#     def __init__(self, root, split, transform=None):
#         self.root = os.path.expanduser(root)
#         file = 'https://datashare.is.ed.ac.uk/bitstream/handle/10283/3443/VCTK-Corpus-0.92.zip'
#         self.split = split
#         self.transform = transform
#         self.sample_rate = SR
        
#         ### CREATE TOKENIZER

        
#         if not check_exists(self.processed_folder):
#             self.process()
#         self.data = pd.read_pickle(os.path.join(self.processed_folder, self.split))
#         self.other = {}
        
#         # set up tokenizer
#         args = {
#             "pad_id": PAD_ID,
#             "bos_id": BOS_ID,
#             "eos_id": EOS_ID,
#             "unk_id": UNK_ID,
#             "input": os.path.join(self.raw_folder, 'text.txt'),
#             "vocab_size": VOCAB_SIZE,
#             "model_prefix": "Multi30k",
#             # "model_type": "word",
#         }
#         combined_args = " ".join(
#             "--{}={}".format(key, value) for key, value in args.items())
#         sentencepiece.SentencePieceTrainer.Train(combined_args)

#         self.vocab = sentencepiece.SentencePieceProcessor()
#         self.vocab.Load("Multi30k.model")
        
#         self.speaker_to_idx = load(os.path.join(self.processed_folder, 'meta'))
    
#     def __getitem__(self, idx):
#         """
#         What i want the code to do is as follows:
        
#         - select target information
#         - clip target audio to MAX_WAVE_LEN with random start position
#         - clip refrence audio to MAX_WAVE_LEN with another random start position
#         """
        
#         # get target information
#         row = self.df.iloc[idx]
        
#         og_audio, sr = librosa.load(row['path'], sr=SR)
#         og_audio = torch.from_numpy(og_audio)
        
#         text = row['text']
#         tokens = torch.tensor(self.vocab.EncodeAsIds(text))
        
#         speaker_id = self.speaker_to_idx[row['speaker_id']]
        
#         # get refrence information b-vae way --- ADHERE TO THESE COMMENTS
#         ## get max wave len random section of audio 
#         ## get another max wav len random section of audio -- this is ref audio
#         if len(og_audio) > MAX_WAV_LEN:
#             start_idx = np.random.randint(0, len(og_audio) - MAX_WAV_LEN)
#             audio = og_audio[start_idx:start_idx + MAX_WAV_LEN]
#             ref_start_idx = np.random.randint(0, len(og_audio) - MAX_WAV_LEN)
#             ref_audio = og_audio[ref_start_idx:ref_start_idx + MAX_WAV_LEN]
#         else:
#             audio = og_audio
#             ref_audio = og_audio
        
#         # get refrence information styletts way --- IGNORE THIS SECTION
#         # ref_row = self.df[self.df['speaker_id'] == row['speaker_id']].sample(1).iloc[0]
        
#         # ref_audio, sr = librosa.load(ref_row['path'], sr=SR)
#         # ref_audio = torch.from_numpy(ref_audio)
        
#         # ref_text = ref_row['text']
#         # ref_tokens = torch.tensor(vocab.EncodeAsIds(ref_text))
        
#         # ref_speaker_id = speaker_to_idx[ref_row['speaker_id']]
        
        
        
        
#         sample = {
#             'data': audio,
#             'target': audio,
#             'tokens': tokens,  # Token IDs
#             'speaker_id': torch.tensor(speaker_id, dtype=torch.long),  # Numeric speaker ID
#             'ref_audio': ref_audio,
#             # 'ref_tokens': ref_tokens,  # Token IDs
#             # 'ref_speaker_id': torch.tensor(ref_speaker_id, dtype=torch.long),
#         }
        
#         if self.transform is not None:
#             sample = self.transform(sample)
        
#         return sample

#     def __len__(self):
#         return len(self.data)

#     @property
#     def processed_folder(self):
#         return os.path.join(self.root, 'processed')

#     @property
#     def raw_folder(self):
#         return os.path.join(self.root, 'raw')

#     def process(self):
#         if not check_exists(self.raw_folder):
#             self.download()
#         train_set, val_set, test_set = self.make_data()
#         train_set.to_pickle(os.path.join(self.processed_folder, 'train'))
#         val_set.to_pickle(os.path.join(self.processed_folder, 'val'))
#         test_set.to_pickle(os.path.join(self.processed_folder, 'test'))
        
#         unique_speakers = (train_set['speaker_id'].unique().tolist() + 
#                             val_set['speaker_id'].unique().tolist() + 
#                             test_set['speaker_id'].unique().tolist())

#         speaker_to_idx = {speaker: idx for idx, speaker in enumerate(unique_speakers)}
#         save(speaker_to_idx, os.path.join(self.processed_folder, 'meta'))
#         return
        

#     def download(self):
#         makedir_exist_ok(self.raw_folder)
#         # Define the URL and the target paths
#         url = self.file
#         data_dir = self.raw_folder
#         download_path = os.path.join(data_dir, 'VCTK-Corpus-0.92.zip')
#         extract_path = os.path.join(data_dir, 'VCTK')

#         # Ensure the data directory exists
#         os.makedirs(data_dir, exist_ok=True)

#         # Download the dataset
#         print(f"Downloading VCTK dataset from {url}...")
#         response = requests.get(url, stream=True)
#         with open(download_path, 'wb') as file:
#             for chunk in response.iter_content(chunk_size=8192):
#                 file.write(chunk)
#         print("Download complete.")

#         # Unzip the file
#         print(f"Extracting {download_path} to {data_dir}...")
#         with zipfile.ZipFile(download_path, 'r') as zip_ref:
#             zip_ref.extractall(data_dir)
#         print("Extraction complete.")

#         # Find the extracted folder and rename it to "VCTK"
#         extracted_folder_name = 'VCTK-Corpus-0.92'
#         original_extract_path = os.path.join(data_dir, extracted_folder_name)

#         if os.path.exists(original_extract_path):
#             os.rename(original_extract_path, extract_path)
#             print(f"Renamed {original_extract_path} to {extract_path}")
#         else:
#             print(f"Expected extracted folder {original_extract_path} not found")

#         print(f"VCTK dataset is ready at {extract_path}")
#         return

#     def __repr__(self):
#         fmt_str = 'Dataset {}\nSize: {}\nRoot: {}\nSplit: {}\nTransforms: {}'.format(
#             self.__class__.__name__, self.__len__(), self.root, self.split, self.transform.__repr__())
#         return fmt_str

#     # def make_data(self):
#     #     train_data = read_image_file(os.path.join(self.raw_folder, 'train-images-idx3-ubyte'))
#     #     test_data = read_image_file(os.path.join(self.raw_folder, 't10k-images-idx3-ubyte'))
#     #     train_target = read_label_file(os.path.join(self.raw_folder, 'train-labels-idx1-ubyte'))
#     #     test_target = read_label_file(os.path.join(self.raw_folder, 't10k-labels-idx1-ubyte'))
#     #     train_id, test_id = np.arange(len(train_data)).astype(np.int64), np.arange(len(test_data)).astype(np.int64)
#     #     classes = list(map(str, list(range(10))))
#     #     classes_to_labels = {classes[i]: i for i in range(len(classes))}
#     #     target_size = len(classes)
#     #     return (train_id, train_data, train_target), (test_id, test_data, test_target), (classes_to_labels, target_size)
    
#     def downsample(self):
#         # Define paths and target sample rate
#         input_dir = os.path.join(self.raw_folder, 'wav48_silence_trimmed')
#         output_dir = os.path.join(self.raw_folder, 'wav{}'.format(self.sr_int))
        
#         # Ensure the output directory exists
#         os.makedirs(output_dir, exist_ok=True)

#         # Collect all files to process
#         files_to_process = []
#         for root, dirs, files in os.walk(input_dir):
#             for file in files:
#                 if file.endswith("_mic1.flac"):
#                     files_to_process.append((root, file))

#         # Process files with a progress bar
#         for root, file in tqdm(files_to_process, desc="Processing files", unit="file"):
#             # Construct full file path
#             file_path = os.path.join(root, file)

#             # Load the audio file using librosa
#             audio, sr = librosa.load(file_path, sr=None)

#             # Downsample the audio file to the target sample rate
#             audio_resampled = librosa.resample(audio, orig_sr=sr, target_sr=self.sample_rate)

#             # Remove '_mic1' from the file name and change extension to .wav
#             new_file_name = file.replace('_mic1.flac', '.wav')

#             # Construct the output file path
#             relative_path = os.path.relpath(file_path, input_dir)
#             relative_dir = os.path.dirname(relative_path)
#             output_file_path = os.path.join(output_dir, relative_dir, new_file_name)
#             output_file_dir = os.path.dirname(output_file_path)
#             os.makedirs(output_file_dir, exist_ok=True)

#             # Export the downsampled audio file as a .wav file using soundfile
#             sf.write(output_file_path, audio_resampled, self.sample_rate)

#     def read_speaker_info(self):
#         speaker_info_path = os.path.join(self.raw_folder, 'speaker-info.txt')
#         speaker_info = {}
#         with open(speaker_info_path, 'r') as file:
#             lines = file.readlines()[1:]  # Skip the header
#             for line in lines:
#                 parts = line.strip().split()
#                 speaker_id = parts[0]
#                 age = parts[1]
#                 gender = parts[2]
#                 accent = parts[3]
#                 region = parts[4] if len(parts) > 4 else ""
#                 comment = " ".join(parts[5:]) if len(parts) > 5 else ""
#                 speaker_info[speaker_id] = {
#                     "age": age,
#                     "gender": gender,
#                     "accent": accent,
#                     "region": region,
#                     "comment": comment,
#                 }
#         return speaker_info
    
#     def make_data(self):
#         # Define paths
#         wav_dir = os.path.join(self.raw_folder, 'wav{}'.format(self.sr_int))
#         txt_dir = os.path.join(self.raw_folder, 'txt')
#         speaker_info = self.read_speaker_info()
        
#         dataset = []

#         files_to_process = []
#         for root, dirs, files in os.walk(wav_dir):
#             for file in files:
#                 if file.endswith(".wav"):
#                     files_to_process.append((root, file))

#         for root, file in tqdm(files_to_process, desc="Creating dataset", unit="file"):
#             file_path = os.path.join(root, file)
#             file_name = os.path.basename(file)
#             speaker_id, text_id = file_name.split("_")[0], file_name.split("_")[1].split(".")[0]
#             text_file_path = os.path.join(txt_dir, speaker_id, "{}_{}.txt".format(speaker_id, text_id))
            
#             # Check if the text file exists
#             if not os.path.exists(text_file_path):
#                 print(f"Text file not found for {file_name}, skipping...")
#                 continue
            
#             with open(text_file_path, 'r') as text_file:
#                 text = text_file.read().strip()
            
#             speaker_meta = speaker_info.get(speaker_id, {})
#             entry = {
#                 "speaker_id": speaker_id,
#                 "text": text,
#                 "path": file_path
#             }
#             dataset.append(entry)
            
#         df = pd.DataFrame(dataset)
#         train_df, val_df, test_df = self.split_dataset(df)

#         # train_df.to_csv('train_{}.csv'.format(int(self.sample_rate//1e3)))
#         # val_df.to_csv('val_{}.csv'.format(int(self.sample_rate//1e3)))
#         # test_df.to_csv('test_{}.csv'.format(int(self.sample_rate//1e3)))
        
        
#         return train_df, val_df, test_df
    
#     def split_dataset(df, train_size=0.7, val_size=0.15, test_size=0.15, random_state=42):
#         # Ensure the split proportions sum to 1
#         assert train_size + val_size + test_size == 1.0, "Train, validation, and test sizes must sum to 1.0"
        
#         # Get unique speakers
#         speakers = df['speaker_id'].unique()
        
#         # Split speakers into train and temp (val + test)
#         train_speakers, temp_speakers = train_test_split(speakers, train_size=train_size, random_state=random_state)
        
#         # Calculate the proportion for validation in the temp split
#         val_proportion = val_size / (val_size + test_size)
        
#         # Split temp_speakers into validation and test sets
#         val_speakers, test_speakers = train_test_split(temp_speakers, train_size=val_proportion, random_state=random_state)
        
#         # Assign entries to the respective sets
#         train_df = df[df['speaker_id'].isin(train_speakers)]
#         val_df = df[df['speaker_id'].isin(val_speakers)]
#         test_df = df[df['speaker_id'].isin(test_speakers)]
        
#         return train_df, val_df, test_df


# import codecs
# import numpy as np
# import os
# import torch
# from PIL import Image
# from torch.utils.data import Dataset
# from module import check_exists, makedir_exist_ok, save, load
# from .utils import download_url, extract_file, make_classes_counts, get_data_path_list
# import torchaudio
# import random
# import re
# import librosa
# import soundfile as sf
# import random
# from config import cfg

# _pad = "$"
# _punctuation = ';:,.!?¡¿—…"«»“” '
# _letters = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz'
# _letters_ipa = "ɑɐɒæɓʙβɔɕçɗɖðʤəɘɚɛɜɝɞɟʄɡɠɢʛɦɧħɥʜɨɪʝɭɬɫɮʟɱɯɰŋɳɲɴøɵɸθœɶʘɹɺɾɻʀʁɽʂʃʈʧʉʊʋⱱʌɣɤʍχʎʏʑʐʒʔʡʕʢǀǁǂǃˈˌːˑʼʴʰʱʲʷˠˤ˞↓↑→↗↘'̩'ᵻ"

# # Export all symbols:
# symbols = [_pad] + list(_punctuation) + list(_letters) + list(_letters_ipa)

# fixed_wave_length = int(4e4)
# cfg['wave_len'] = fixed_wave_length

# dicts = {}
# for i in range(len((symbols))):
#     dicts[symbols[i]] = i

# class TextCleaner:
#     def __init__(self, dummy=None):
#         self.word_index_dictionary = dicts
#     def __call__(self, text):
#         indexes = []
#         for char in text:
#             try:
#                 indexes.append(self.word_index_dictionary[char])
#             except KeyError:
#                 print(text)
#         return indexes

# np.random.seed(1)
# SPECT_PARAMS = {
#     "n_fft": 2048,
#     "win_length": 1200,
#     "hop_length": 300
# }
# MEL_PARAMS = {
#     "n_mels": 80,
# }

# to_mel = torchaudio.transforms.MelSpectrogram(
#     n_mels=80, n_fft=2048, win_length=1200, hop_length=300)
# mean, std = -4, 4

# def preprocess(wave):
#     wave_tensor = torch.from_numpy(wave).float()
#     mel_tensor = to_mel(wave_tensor)
#     mel_tensor = (torch.log(1e-5 + mel_tensor.unsqueeze(0)) - mean) / std
#     return mel_tensor

# # NEEDS TO BE ADJUSTED 
# class VCTK(Dataset):
#     data_name = 'VCTK'
#     file = [('https://datashare.ed.ac.uk/download/DS_10283_3443.zip')]
    
#     def __init__(self,
#                  root,
#                  split,
#                  transform=None,
#                  sr=8000,
#                  data_augmentation=False,
#                  validation=False,
#                  ):

#         spect_params = SPECT_PARAMS
#         mel_params = MEL_PARAMS

#         self.root = os.path.expanduser(root)
#         self.split = split
#         self.transform = transform
#         if not check_exists(self.raw_folder):
#             raise Exception("No data found") # update this to auto process/download raw data later
#             self.process()
        
#         if sr/1000 != sr//1000:
#             raise Exception("signal rate is not a multiple of 1000")
#         if split not in ['train', 'test', 'val']:
#             raise Exception(f"Expected value in ['train', 'test', 'val']. Got '{split}'")
#         data_root = os.path.join(self.processed_folder, f"{split}_list{sr//1000}.txt")
        
        
#         data_list = get_data_path_list(data_root) 
        
#         _data_list = [l[:-1].split('|') for l in data_list]
#         for l in _data_list:
#             l[0] = "data/VCTK/" + l[0]
#         self.data_list = [data if len(data) == 3 else (*data, 0) for data in _data_list]
#         self.text_cleaner = TextCleaner()
#         self.sr = sr

#         self.to_melspec = torchaudio.transforms.MelSpectrogram(**MEL_PARAMS)

#         self.mean, self.std = -4, 4
#         self.data_augmentation = data_augmentation and (not validation)
#         self.max_mel_length = 192
#         self.id = [int(''.join(re.search(r'(\d+)_(\d+)\.wav$', data[0]).groups())) for data in self.data_list] # double check this for errors
#         self.data = self.data_list
#         self.other = {}

#     # def __init__(self, root, split, transform=None):
#     #     self.root = os.path.expanduser(root)
#     #     self.split = split
#     #     self.transform = transform
#     #     # MAKE FILE PATH DATALOADER
#     #     if not check_exists(self.processed_folder):
#     #         raise Exception("No data found") # update this to auto process/download later
#     #         self.process()
#     #     self.id, self.data, self.target = load(os.path.join(self.processed_folder, self.split))
#     #     self.other = {}
#     #     # NO TARGETS NECESSARY
#     #     self.classes_counts = make_classes_counts(self.target)
#     #     self.classes_to_labels, self.target_size = load(os.path.join(self.processed_folder, 'meta'))

#     # def __getitem__(self, index):
#     #     # CAN USE STYLETTS GET ITEM FROM MELDATASET.PY
#     #     id, data, target = torch.tensor(self.id[index]), Image.fromarray(self.data[index], mode='L'), torch.tensor(
#     #         self.target[index])
#     #     input = {'id': id, 'data': data, 'target': target}
#     #     other = {k: torch.tensor(self.other[k][index]) for k in self.other}
#     #     input = {**input, **other}
#         # if self.transform is not None:
#         #     input = self.transform(input)
#         # return input
    
#     def __getitem__(self, idx):
#         data = self.data_list[idx]
#         path = data[0]
        
#         wave, text_tensor, speaker_id = self._load_tensor(data)
        
#         # mel_tensor = preprocess(wave).squeeze() ****
        
#         # mel_tensor = mel_tensor.squeeze()     MEL STUFF. LEAVE HERE
#         # length_feature = mel_tensor.size(1)   ****
#         # mel_tensor = mel_tensor[:, :(length_feature - length_feature % 2)] ****
        
#         ref_data = random.choice(self.data_list)
#         # ref_wave, ref_mel_tensor, ref_label = self._load_data(ref_data) ****
#         ref_wave, ref_label = self._load_data(ref_data)
        
        
#         # return wave, speaker_id, acoustic_feature, text_tensor, ref_mel_tensor, ref_label, path # RETURN A DICTIONARY CONTAINING EACH VALUE AT A KEY
        
#         id = torch.tensor(self.id[idx])
#         # data = (wave, speaker_id, mel_tensor, text_tensor, ref_mel_tensor, ref_label, path)
#         # input = {
#         #         'id':  id, 'wave': wave, 'speaker_id': target, 
#         #         "mel_tensor": mel_tensor, "text_tensor": text_tensor,
#         #         "ref_wave": ref_wave, "ref_mel_tensor": ref_mel_tensor, 
#         #         "ref_speaker_id": torch.tensor(ref_label)
#         #     } 
#         input = {
#                 'id':  id, 'data': wave, 
#                 "text_tensor": text_tensor,
#                 'target': wave, "ref_wave": ref_wave, 
#                 "ref_speaker_id": torch.tensor(ref_label)
#             } 
#         other = {k: torch.tensor(self.other[k][idx]) for k in self.other}
#         input = {**input, **other}
#         # if self.transform is not None:
#         #     input = self.transform(input['data'])
#         return input
            
#     # def _load_data(self, data):     ****
#     #     wave, text_tensor, speaker_id = self._load_tensor(data)
#     #     mel_tensor = preprocess(wave).squeeze()
#     #     mel_length = mel_tensor.size(1) 
#     #     if mel_length > self.max_mel_length:
#     #         random_start = np.random.randint(0, mel_length - self.max_mel_length)
#     #         mel_tensor = mel_tensor[:, random_start:random_start + self.max_mel_length]

#     #     return wave, mel_tensor, speaker_id ****
    
#     def _load_data(self, data):
#         wave, text_tensor, speaker_id = self._load_tensor(data)
#         return wave, speaker_id

#     def __len__(self):
#         return len(self.data)

#     @property
#     def processed_folder(self):
#         return os.path.join(self.root, 'processed')

#     @property
#     def raw_folder(self):
#         return os.path.join(self.root, 'raw')

#     def process(self): # need to implement for processing
#         raise NotImplementedError("Process function has not yet been implemented")
#         if not check_exists(self.raw_folder):
#             self.download()
#         train_set, test_set = self.make_data()
#         save(train_set, os.path.join(self.processed_folder, 'train'))
#         save(test_set, os.path.join(self.processed_folder, 'test'))
#         return

#     def _load_tensor(self, data):
#         wave_path, text, speaker_id = data
#         speaker_id = int(speaker_id)
#         wave, sr = sf.read(wave_path)
#         if wave.shape[-1] == 2:
#             wave = wave[:, 0].squeeze()
#         # if sr != 24000:
#         #     wave = librosa.resample(wave, sr, 24000)
#         #     print(wave_path, sr)
           
#         wave, index = librosa.effects.trim(wave, top_db=30)
#         # wave = np.concatenate([np.zeros([5000]), wave, np.zeros([5000])], axis=0) # is this necessary??? 
#         # fixed_wave_length = int(3e4)
        
#         size = wave.size
#         if size > fixed_wave_length:
#             start = random.randint(0, size - fixed_wave_length-1) # minus one for indexing from zeros
#             end =  start + fixed_wave_length
#             wave = wave[start:end]
#         elif size < fixed_wave_length:
#             wave = pad_waveform(wave, fixed_wave_length)
        
#         # print()
#         # print("og size", size)
#         # print("new size", (wave.size))
#         # print()
                
#         text = self.text_cleaner(text)
        
#         text.insert(0, 0)
#         text.append(0)
        
#         text = torch.LongTensor(text)
#         wave = torch.tensor(wave, dtype=torch.float32)

#         return wave, text, speaker_id

#     def download(self):
#         raise NotImplementedError
#         makedir_exist_ok(self.raw_folder)
#         for (url, md5) in self.file:
#             filename = os.path.basename(url)
#             download_url(url, os.path.join(self.raw_folder, filename), md5)
#             extract_file(os.path.join(self.raw_folder, filename))
#         return
#     def download(self):  
#         # Define the URL and the target paths
#         makedir_exist_ok(self.raw_folder)
#         url = 'https://datashare.is.ed.ac.uk/bitstream/handle/10283/3443/VCTK-Corpus-0.92.zip'
#         data_dir = './data/VCTK/raw'
#         download_path = os.path.join(data_dir, 'VCTK-Corpus-0.92.zip')
#         extract_path = os.path.join(data_dir, 'VCTK')

#         # Ensure the data directory exists
#         os.makedirs(data_dir, exist_ok=True)

#         # Download the dataset
#         print(f"Downloading VCTK dataset from {url}...")
#         response = requests.get(url, stream=True)
#         with open(download_path, 'wb') as file:
#             for chunk in response.iter_content(chunk_size=8192):
#                 file.write(chunk)
#         print("Download complete.")

#         # Unzip the file
#         print(f"Extracting {download_path} to {data_dir}...")
#         with zipfile.ZipFile(download_path, 'r') as zip_ref:
#             zip_ref.extractall(data_dir)
#         print("Extraction complete.")

#         # Find the extracted folder and rename it to "VCTK"
#         extracted_folder_name = 'VCTK-Corpus-0.92'
#         original_extract_path = os.path.join(data_dir, extracted_folder_name)

#         if os.path.exists(original_extract_path):
#             os.rename(original_extract_path, extract_path)
#             print(f"Renamed {original_extract_path} to {extract_path}")
#         else:
#             print(f"Expected extracted folder {original_extract_path} not found")

#         print(f"VCTK dataset is ready at {extract_path}")

#         def __repr__(self):
#             fmt_str = 'Dataset {}\nSize: {}\nRoot: {}\nSplit: {}\nTransforms: {}'.format(
#                 self.__class__.__name__, self.__len__(), self.root, self.split, self.transform.__repr__())
#             return fmt_str

#     def make_data(self):
#         raise NotImplementedError
#         train_data = read_image_file(os.path.join(self.raw_folder, 'train-images-idx3-ubyte'))
#         test_data = read_image_file(os.path.join(self.raw_folder, 't10k-images-idx3-ubyte'))
#         train_target = read_label_file(os.path.join(self.raw_folder, 'train-labels-idx1-ubyte'))
#         test_target = read_label_file(os.path.join(self.raw_folder, 't10k-labels-idx1-ubyte'))
#         train_id, test_id = np.arange(len(train_data)).astype(np.int64), np.arange(len(test_data)).astype(np.int64)
#         # classes = list(map(str, list(range(10))))
#         # classes_to_labels = {classes[i]: i for i in range(len(classes))}
#         # target_size = len(classes)
#         return (train_id, train_data, train_target), (test_id, test_data, test_target)

# def get_int(b):
#     return int(codecs.encode(b, 'hex'), 16)


# # def read_image_file(path):
# #     with open(path, 'rb') as f:
# #         data = f.read()
# #         assert get_int(data[:4]) == 2051
# #         length = get_int(data[4:8])
# #         num_rows = get_int(data[8:12])
# #         num_cols = get_int(data[12:16])
# #         parsed = np.frombuffer(data, dtype=np.uint8, offset=16).reshape((length, num_rows, num_cols))
# #         return parsed


# # def read_label_file(path):
# #     with open(path, 'rb') as f:
# #         data = f.read()
# #         assert get_int(data[:4]) == 2049
# #         length = get_int(data[4:8])
# #         parsed = np.frombuffer(data, dtype=np.uint8, offset=8).reshape(length).astype(np.int64)
# #         return parsed

# def pad_waveform(waveform, target_length):
#     pad_length = target_length-len(waveform)
#     waveform = np.concatenate((waveform, np.zeros(pad_length)), axis=0)
#     return waveform


# # def pad_mel_spectrogram(mel_spectrogram, target_length):
# #     # Pad the Mel spectrogram to the target time length
# #     current_time_frames = mel_spectrogram.shape[-1]
# #     if current_time_frames < target_length:
# #         pad_amount = target_length - current_time_frames
# #         mel_spectrogram = torch.nn.functional.pad(mel_spectrogram, (0, pad_amount), mode='constant', value=0)
# #     return mel_spectrogram

# def pad_mel_spectrogram(mel_spectrogram, target_length):
#     # Calculate the padding amount
#     current_length = mel_spectrogram.shape[1]
#     pad_width = target_length - current_length
#     if pad_width > 0:
#         # Pad Mel spectrogram with zeros at the end along the time axis
#         mel_spectrogram = np.pad(mel_spectrogram, ((0, 0), (0, pad_width)), mode='constant')
#     return mel_spectrogram

# def create_padding_mask(array):
#     mask = array != 0
#     return mask

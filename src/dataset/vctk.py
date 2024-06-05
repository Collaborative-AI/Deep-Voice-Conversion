import codecs
import numpy as np
import os
import torch
from PIL import Image
from torch.utils.data import Dataset
from module import check_exists, makedir_exist_ok, save, load
from .utils import download_url, extract_file, make_classes_counts, get_data_path_list
import torchaudio
import random
import re
import librosa
import soundfile as sf
import random
from config import cfg

_pad = "$"
_punctuation = ';:,.!?¡¿—…"«»“” '
_letters = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz'
_letters_ipa = "ɑɐɒæɓʙβɔɕçɗɖðʤəɘɚɛɜɝɞɟʄɡɠɢʛɦɧħɥʜɨɪʝɭɬɫɮʟɱɯɰŋɳɲɴøɵɸθœɶʘɹɺɾɻʀʁɽʂʃʈʧʉʊʋⱱʌɣɤʍχʎʏʑʐʒʔʡʕʢǀǁǂǃˈˌːˑʼʴʰʱʲʷˠˤ˞↓↑→↗↘'̩'ᵻ"

# Export all symbols:
symbols = [_pad] + list(_punctuation) + list(_letters) + list(_letters_ipa)

fixed_wave_length = int(4e4)
cfg['wave_len'] = fixed_wave_length

dicts = {}
for i in range(len((symbols))):
    dicts[symbols[i]] = i

class TextCleaner:
    def __init__(self, dummy=None):
        self.word_index_dictionary = dicts
    def __call__(self, text):
        indexes = []
        for char in text:
            try:
                indexes.append(self.word_index_dictionary[char])
            except KeyError:
                print(text)
        return indexes

np.random.seed(1)
SPECT_PARAMS = {
    "n_fft": 2048,
    "win_length": 1200,
    "hop_length": 300
}
MEL_PARAMS = {
    "n_mels": 80,
}

to_mel = torchaudio.transforms.MelSpectrogram(
    n_mels=80, n_fft=2048, win_length=1200, hop_length=300)
mean, std = -4, 4

def preprocess(wave):
    wave_tensor = torch.from_numpy(wave).float()
    mel_tensor = to_mel(wave_tensor)
    mel_tensor = (torch.log(1e-5 + mel_tensor.unsqueeze(0)) - mean) / std
    return mel_tensor

# NEEDS TO BE ADJUSTED 
class VCTK(Dataset):
    data_name = 'VCTK'
    file = [('https://datashare.ed.ac.uk/download/DS_10283_3443.zip')]
    
    def __init__(self,
                 root,
                 split,
                 transform=None,
                 sr=8000,
                 data_augmentation=False,
                 validation=False,
                 ):

        spect_params = SPECT_PARAMS
        mel_params = MEL_PARAMS

        self.root = os.path.expanduser(root)
        self.split = split
        self.transform = transform
        if not check_exists(self.raw_folder):
            raise Exception("No data found") # update this to auto process/download raw data later
            self.process()
        
        if sr/1000 != sr//1000:
            raise Exception("signal rate is not a multiple of 1000")
        if split not in ['train', 'test', 'val']:
            raise Exception(f"Expected value in ['train', 'test', 'val']. Got '{split}'")
        data_root = os.path.join(self.processed_folder, f"{split}_list{sr//1000}.txt")
        
        
        data_list = get_data_path_list(data_root) 
        
        _data_list = [l[:-1].split('|') for l in data_list]
        for l in _data_list:
            l[0] = "data/VCTK/" + l[0]
        self.data_list = [data if len(data) == 3 else (*data, 0) for data in _data_list]
        self.text_cleaner = TextCleaner()
        self.sr = sr

        self.to_melspec = torchaudio.transforms.MelSpectrogram(**MEL_PARAMS)

        self.mean, self.std = -4, 4
        self.data_augmentation = data_augmentation and (not validation)
        self.max_mel_length = 192
        self.id = [int(''.join(re.search(r'(\d+)_(\d+)\.wav$', data[0]).groups())) for data in self.data_list] # double check this for errors
        self.data = self.data_list
        self.other = {}

    # def __init__(self, root, split, transform=None):
    #     self.root = os.path.expanduser(root)
    #     self.split = split
    #     self.transform = transform
    #     # MAKE FILE PATH DATALOADER
    #     if not check_exists(self.processed_folder):
    #         raise Exception("No data found") # update this to auto process/download later
    #         self.process()
    #     self.id, self.data, self.target = load(os.path.join(self.processed_folder, self.split))
    #     self.other = {}
    #     # NO TARGETS NECESSARY
    #     self.classes_counts = make_classes_counts(self.target)
    #     self.classes_to_labels, self.target_size = load(os.path.join(self.processed_folder, 'meta'))

    # def __getitem__(self, index):
    #     # CAN USE STYLETTS GET ITEM FROM MELDATASET.PY
    #     id, data, target = torch.tensor(self.id[index]), Image.fromarray(self.data[index], mode='L'), torch.tensor(
    #         self.target[index])
    #     input = {'id': id, 'data': data, 'target': target}
    #     other = {k: torch.tensor(self.other[k][index]) for k in self.other}
    #     input = {**input, **other}
        # if self.transform is not None:
        #     input = self.transform(input)
        # return input
    
    def __getitem__(self, idx):
        data = self.data_list[idx]
        path = data[0]
        
        wave, text_tensor, speaker_id = self._load_tensor(data)
        
        # mel_tensor = preprocess(wave).squeeze() ****
        
        # mel_tensor = mel_tensor.squeeze()     MEL STUFF. LEAVE HERE
        # length_feature = mel_tensor.size(1)   ****
        # mel_tensor = mel_tensor[:, :(length_feature - length_feature % 2)] ****
        
        ref_data = random.choice(self.data_list)
        # ref_wave, ref_mel_tensor, ref_label = self._load_data(ref_data) ****
        ref_wave, ref_label = self._load_data(ref_data)
        
        
        # return wave, speaker_id, acoustic_feature, text_tensor, ref_mel_tensor, ref_label, path # RETURN A DICTIONARY CONTAINING EACH VALUE AT A KEY
        
        id = torch.tensor(self.id[idx])
        # data = (wave, speaker_id, mel_tensor, text_tensor, ref_mel_tensor, ref_label, path)
        # input = {
        #         'id':  id, 'wave': wave, 'speaker_id': target, 
        #         "mel_tensor": mel_tensor, "text_tensor": text_tensor,
        #         "ref_wave": ref_wave, "ref_mel_tensor": ref_mel_tensor, 
        #         "ref_speaker_id": torch.tensor(ref_label)
        #     } 
        input = {
                'id':  id, 'data': wave, 
                "text_tensor": text_tensor,
                'target': wave, "ref_wave": ref_wave, 
                "ref_speaker_id": torch.tensor(ref_label)
            } 
        other = {k: torch.tensor(self.other[k][idx]) for k in self.other}
        input = {**input, **other}
        # if self.transform is not None:
        #     input = self.transform(input['data'])
        return input
            
    # def _load_data(self, data):     ****
    #     wave, text_tensor, speaker_id = self._load_tensor(data)
    #     mel_tensor = preprocess(wave).squeeze()
    #     mel_length = mel_tensor.size(1) 
    #     if mel_length > self.max_mel_length:
    #         random_start = np.random.randint(0, mel_length - self.max_mel_length)
    #         mel_tensor = mel_tensor[:, random_start:random_start + self.max_mel_length]

    #     return wave, mel_tensor, speaker_id ****
    
    def _load_data(self, data):
        wave, text_tensor, speaker_id = self._load_tensor(data)
        return wave, speaker_id

    def __len__(self):
        return len(self.data)

    @property
    def processed_folder(self):
        return os.path.join(self.root, 'processed')

    @property
    def raw_folder(self):
        return os.path.join(self.root, 'raw')

    def process(self): # need to implement for processing
        raise NotImplementedError("Process function has not yet been implemented")
        if not check_exists(self.raw_folder):
            self.download()
        train_set, test_set = self.make_data()
        save(train_set, os.path.join(self.processed_folder, 'train'))
        save(test_set, os.path.join(self.processed_folder, 'test'))
        return

    def _load_tensor(self, data):
        wave_path, text, speaker_id = data
        speaker_id = int(speaker_id)
        wave, sr = sf.read(wave_path)
        if wave.shape[-1] == 2:
            wave = wave[:, 0].squeeze()
        # if sr != 24000:
        #     wave = librosa.resample(wave, sr, 24000)
        #     print(wave_path, sr)
           
        wave, index = librosa.effects.trim(wave, top_db=30)
        # wave = np.concatenate([np.zeros([5000]), wave, np.zeros([5000])], axis=0) # is this necessary??? 
        # fixed_wave_length = int(3e4)
        
        size = wave.size
        if size > fixed_wave_length:
            start = random.randint(0, size - fixed_wave_length-1) # minus one for indexing from zeros
            end =  start + fixed_wave_length
            wave = wave[start:end]
        elif size < fixed_wave_length:
            wave = pad_waveform(wave, fixed_wave_length)
        
        # print()
        # print("og size", size)
        # print("new size", (wave.size))
        # print()
                
        text = self.text_cleaner(text)
        
        text.insert(0, 0)
        text.append(0)
        
        text = torch.LongTensor(text)
        wave = torch.tensor(wave, dtype=torch.float32)

        return wave, text, speaker_id

    def download(self):
        raise NotImplementedError
        makedir_exist_ok(self.raw_folder)
        for (url, md5) in self.file:
            filename = os.path.basename(url)
            download_url(url, os.path.join(self.raw_folder, filename), md5)
            extract_file(os.path.join(self.raw_folder, filename))
        return

    def __repr__(self):
        fmt_str = 'Dataset {}\nSize: {}\nRoot: {}\nSplit: {}\nTransforms: {}'.format(
            self.__class__.__name__, self.__len__(), self.root, self.split, self.transform.__repr__())
        return fmt_str

    def make_data(self):
        raise NotImplementedError
        train_data = read_image_file(os.path.join(self.raw_folder, 'train-images-idx3-ubyte'))
        test_data = read_image_file(os.path.join(self.raw_folder, 't10k-images-idx3-ubyte'))
        train_target = read_label_file(os.path.join(self.raw_folder, 'train-labels-idx1-ubyte'))
        test_target = read_label_file(os.path.join(self.raw_folder, 't10k-labels-idx1-ubyte'))
        train_id, test_id = np.arange(len(train_data)).astype(np.int64), np.arange(len(test_data)).astype(np.int64)
        # classes = list(map(str, list(range(10))))
        # classes_to_labels = {classes[i]: i for i in range(len(classes))}
        # target_size = len(classes)
        return (train_id, train_data, train_target), (test_id, test_data, test_target)

def get_int(b):
    return int(codecs.encode(b, 'hex'), 16)


# def read_image_file(path):
#     with open(path, 'rb') as f:
#         data = f.read()
#         assert get_int(data[:4]) == 2051
#         length = get_int(data[4:8])
#         num_rows = get_int(data[8:12])
#         num_cols = get_int(data[12:16])
#         parsed = np.frombuffer(data, dtype=np.uint8, offset=16).reshape((length, num_rows, num_cols))
#         return parsed


# def read_label_file(path):
#     with open(path, 'rb') as f:
#         data = f.read()
#         assert get_int(data[:4]) == 2049
#         length = get_int(data[4:8])
#         parsed = np.frombuffer(data, dtype=np.uint8, offset=8).reshape(length).astype(np.int64)
#         return parsed

def pad_waveform(waveform, target_length):
    pad_length = target_length-len(waveform)
    waveform = np.concatenate((waveform, np.zeros(pad_length)), axis=0)
    return waveform


# def pad_mel_spectrogram(mel_spectrogram, target_length):
#     # Pad the Mel spectrogram to the target time length
#     current_time_frames = mel_spectrogram.shape[-1]
#     if current_time_frames < target_length:
#         pad_amount = target_length - current_time_frames
#         mel_spectrogram = torch.nn.functional.pad(mel_spectrogram, (0, pad_amount), mode='constant', value=0)
#     return mel_spectrogram

def pad_mel_spectrogram(mel_spectrogram, target_length):
    # Calculate the padding amount
    current_length = mel_spectrogram.shape[1]
    pad_width = target_length - current_length
    if pad_width > 0:
        # Pad Mel spectrogram with zeros at the end along the time axis
        mel_spectrogram = np.pad(mel_spectrogram, ((0, 0), (0, pad_width)), mode='constant')
    return mel_spectrogram

def create_padding_mask(array):
    mask = array != 0
    return mask
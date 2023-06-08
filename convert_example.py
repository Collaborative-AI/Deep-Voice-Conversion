import torch
import numpy as np

import soundfile as sf

from models.model_encoder import ContentEncoder, StyleEncoder
from models.model_decoder import Decoder_ac_without_lf0
from models.model_encoder_contrastive import ASE
import os

import subprocess
from spectrogram import logmelspectrogram
import kaldiio

import resampy
import pyworld as pw

import argparse


def extract_logmel(wav_path, mean, std, sr=16000):
    # wav, fs = librosa.load(wav_path, sr=sr)
    wav, fs = sf.read(wav_path)
    if fs != sr:
        wav = resampy.resample(wav, fs, sr, axis=0)
        fs = sr
    #wav, _ = librosa.effects.trim(wav, top_db=15)
    # duration = len(wav)/fs
    assert fs == 16000
    peak = np.abs(wav).max()
    if peak > 1.0:
        wav /= peak
    mel = logmelspectrogram(
                x=wav,
                fs=fs,
                n_mels=80,
                n_fft=400,
                n_shift=160,
                win_length=400,
                window='hann',
                fmin=80,
                fmax=7600,
            )
    # compute mel-spectrogram feature from an audio waveform
    #result: 2D array
    
    #pitch analysis
    mel = (mel - mean) / (std + 1e-8) #normalizing the mel-spectrogram feature
    tlen = mel.shape[0]
    frame_period = 160/fs*1000 
    f0, timeaxis = pw.dio(wav.astype('float64'), fs, frame_period=frame_period) #raw pitch extractor
    f0 = pw.stonemask(wav.astype('float64'), f0, timeaxis, fs) #pitch refinement
    f0 = f0[:tlen].reshape(-1).astype('float32') #match the length of tlen, result: 1D array
    nonzeros_indices = np.nonzero(f0)
    lf0 = f0.copy()
    lf0[nonzeros_indices] = np.log(f0[nonzeros_indices]) # for f0(Hz), lf0 > 0 when f0 != 0
    mean, std = np.mean(lf0[nonzeros_indices]), np.std(lf0[nonzeros_indices])
    lf0[nonzeros_indices] = (lf0[nonzeros_indices] - mean) / (std + 1e-8) #normalization of lf0 values
    return mel, lf0


def convert(args):
    src_wav_path = args.source_wav
    ref_wav_path = args.reference_wav
    
    out_dir = args.converted_wav_path
    os.makedirs(out_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#initiate neural network models
    encoder = ContentEncoder(in_channels=80, channels=512, n_embeddings=512, z_dim=64, c_dim=256)
    encoder_style = StyleEncoder()
    encoder_ase = ASE()
    decoder = Decoder_ac_without_lf0(dim_neck=64)
    encoder.to(device)
    encoder_style.to(device)
    encoder_ase.to(device)
    decoder.to(device)
#loads the checkpoint of the trained models from the given model path
    checkpoint_path = args.model_path
    checkpoint = torch.load(checkpoint_path, map_location=lambda storage, loc: storage)
    encoder.load_state_dict(checkpoint["encoder"])
    encoder_style.load_state_dict(checkpoint["encoder_style"])
    decoder.load_state_dict(checkpoint["decoder"])
#set to evaluation mode
    encoder.eval()
    encoder_style.eval()
    decoder.eval()
    
    mel_stats = np.load('./mel_stats/stats.npy')
    mean = mel_stats[0]
    std = mel_stats[1]
    feat_writer = kaldiio.WriteHelper("ark,scp:{o}.ark,{o}.scp".format(o=str(out_dir)+'/feats.1')) #write the converted source and ref features to kaldi-compatible files
    #extract mel and lfo from the source and reference WAV files
    src_mel, src_lf0 = extract_logmel(src_wav_path, mean, std)
    ref_mel, _ = extract_logmel(ref_wav_path, mean, std)
    #converts to tensor and moves to device
    src_mel = torch.FloatTensor(src_mel.T).unsqueeze(0).to(device)
    src_lf0 = torch.FloatTensor(src_lf0).unsqueeze(0).to(device)
    ref_mel = torch.FloatTensor(ref_mel.T).unsqueeze(0).to(device)
    out_filename = os.path.basename(src_wav_path).split('.')[0] 
    with torch.no_grad():
        z, _, _, _ = encoder.encode(src_mel)
        style_embs = encoder_style(src_lf0)
        style_emb = encoder_style(ref_mel)
        output = decoder(z, style_embs, style_emb)
        
        feat_writer[out_filename+'_converted'] = output.squeeze(0).cpu().numpy()
        feat_writer[out_filename+'_source'] = src_mel.squeeze(0).cpu().numpy().T
        feat_writer[out_filename+'_reference'] = ref_mel.squeeze(0).cpu().numpy().T
    
    feat_writer.close()
    print('synthesize waveform...')
    cmd = ['parallel-wavegan-decode', '--checkpoint', \
           './vocoder/checkpoint-3000000steps.pkl', \
           '--feats-scp', f'{str(out_dir)}/feats.1.scp', '--outdir', str(out_dir)]
    subprocess.call(cmd)

if __name__ == "__main__":
    mode = 1
    # 0: both unseen; 1: both seen
    parser = argparse.ArgumentParser()
    parser = argparse.ArgumentParser(conflict_handler='resolve')
    parser.add_argument('--model_path', '-m', type=str, required=False, default="checkpoints/useCSMITrue_useCPMITrue_usePSMITrue_useAmpFalse/model.ckpt-500.pt") 
    
    if mode == 0:
        speaker_names = []
        folder_path = 'data/test/lf0'
        data_root = "Dataset/VCTK-Corpus/wav48/"
        
        for file in os.listdir(folder_path):
            speaker_names.append(str(file))
        
        # reference is the speaker (pair[0]), source is the content (pair[1]) 
        style_content_pairs = [(speaker_names[i], speaker_names[i+1]) for i in range(0, len(speaker_names), 2)]
        
        for pair in style_content_pairs:
            try:
                reference_root = data_root + pair[0] + "/" + pair[0] + "_002.wav"
                parser.add_argument('--reference_wav', '-r', type=str, required=False, default=reference_root)
            except Exception as e:
                reference_root = data_root + pair[0] + "/" + pair[0] + "_003.wav"
                parser.add_argument('--reference_wav', '-r', type=str, required=False, default=reference_root)            
                continue
            for i in range(10, 20):
                try:
                    source_root = data_root + pair[1] + "/" + pair[1] + "_0" + str(i) + ".wav"
                    parser.add_argument('--source_wav', '-s', type=str, required=False, default=source_root)
                    converted_root = 'converted/unseen_content_unseen_speaker/' + pair[0] + "_" + pair[1] + "_test_" + str(i-9)
                    parser.add_argument('--converted_wav_path', '-c', type=str, default=converted_root)
                    args = parser.parse_args()
                    convert(args)
                except Exception as e:
                    print("Error:", e)
                    continue
    
    if mode == 1:
        style_names = []
        folder_path = 'data/train/lf0'
        data_root = "Dataset/VCTK-Corpus/wav48/"
        
        for file in os.listdir(folder_path):
            style_names.append(str(file))
        
        style_names = style_names[:20]
        print(len(style_names))
        
        # reference is the speaker (pair[0]), source is the content (pair[1]) 
        style_content_pairs = [(style_names[i], style_names[i+1]) for i in range(0, len(style_names), 2)]
        
        for pair in style_content_pairs:
            try:
                reference_root = data_root + pair[0] + "/" + pair[0] + "_002.wav"
                parser.add_argument('--reference_wav', '-r', type=str, required=False, default=reference_root)
            except Exception as e:
                reference_root = data_root + pair[0] + "/" + pair[0] + "_003.wav"
                parser.add_argument('--reference_wav', '-r', type=str, required=False, default=reference_root)            
                continue
            for i in range(10, 20):
                try:
                    source_root = data_root + pair[1] + "/" + pair[1] + "_0" + str(i) + ".wav"
                    parser.add_argument('--source_wav', '-s', type=str, required=False, default=source_root)
                    converted_root = 'converted/seen_content_seen_speaker/' + pair[0] + "_" + pair[1] + "_test_" + str(i-9)
                    parser.add_argument('--converted_wav_path', '-c', type=str, default=converted_root)
                    args = parser.parse_args()
                    convert(args)
                except Exception as e:
                    print("Error:", e)
                    continue

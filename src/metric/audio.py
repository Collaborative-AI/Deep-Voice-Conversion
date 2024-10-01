import os
import jiwer
import torch
import torchaudio
from transformers import Wav2Vec2ForCTC, Wav2Vec2Tokenizer, Wav2Vec2Processor, WavLMForCTC
from scipy.spatial.distance import cosine


class ASR:

    def __init__(self, device=None):
        self.model_name_or_path = "facebook/wav2vec2-large-960h-lv60-self"
        self.cache_dir = os.path.join('output', 'asr')
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device
        self.model, self.tokenizer = self.make_model()
        self.reset()

    def reset(self):
        self.output_target = []
        self.input_target = []
        return

    def add(self, input, output):
        with torch.no_grad():
            self.output_target.append(self.wav_to_text(output['target']))
            self.input_target.append(self.wav_to_text(input['target']))
        return

    def __call__(self, input=None, output=None):
        with torch.no_grad():
            wer = self.wer(self.output_target, self.input_target)
            cer = self.cer(self.output_target, self.input_target)
        self.reset()
        return wer, cer

    def make_model(self):
        model = Wav2Vec2ForCTC.from_pretrained(self.model_name_or_path, cache_dir=self.cache_dir).to(self.device)
        tokenizer = Wav2Vec2Tokenizer.from_pretrained(self.model_name_or_path, cache_dir=self.cache_dir)
        return model, tokenizer

    def wav_to_text(self, wav):
        # Tokenize the input
        inputs = self.tokenizer(wav, sampling_rate=16000, return_tensors="pt", padding="longest")

        # Fix input shape if necessary
        input_values = inputs.input_values.squeeze(1)  # Squeeze out the extra dimension

        # Move tensors to the GPU if available
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        input_values = input_values.to(device)

        # Get the model predictions (logits)
        logits = self.model(input_values).logits

        # Get the predicted IDs
        predicted_ids = torch.argmax(logits, dim=-1)

        # Decode the predicted IDs into the text (batch_decode returns a list, so we take [0])
        result = self.tokenizer.batch_decode(predicted_ids)[0]

        return result

    def wer(self, output, target):
        # perform calcute for word error rate
        wer = jiwer.wer(output, target)
        return wer

    def cer(self, output, target):
        # perform calcute for character error rate
        cer = jiwer.cer(output, target)
        return cer


class SSIM:

    def __init__(self, device=None):
        self.model_name_or_path = "facebook/wav2vec2-large-960h-lv60-self"
        self.cache_dir = os.path.join('output', 'asr')
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device
        self.model, self.processor = self.make_model()
        self.reset()

    def reset(self):
        self.output_target = []
        self.input_target = []
        return

    def add(self, input, output):
        return

    def __call__(self, input, output):
        with torch.no_grad():
            output_target = self.extract_embedding(output['target'])
            input_target = self.extract_embedding(input['target'])
            ssim = (output_target, input_target)
        self.reset()
        return ssim

    def make_model(self):
        processor = Wav2Vec2Processor.from_pretrained('patrickvonplaten/wavlm-libri-clean-100h-base-plus',
                                                      cache_dir=self.cache_dir)
        model = WavLMForCTC.from_pretrained('patrickvonplaten/wavlm-libri-clean-100h-base-plus',
                                            cache_dir=self.cache_dir).to(self.device)
        return model, processor

    # Extracting embedded value of a wav file (input can be modified later to meet our need)
    def extract_embedding(self, waveform, sample_rate=16000):
        # Resample to 16kHz if necessary
        if sample_rate != 16000:
            resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)
            waveform = resampler(waveform)
        waveform = waveform.squeeze()

        # Process the waveform using the feature extractor
        inputs = self.processor(waveform, sampling_rate=16000, return_tensors="pt", padding=True)
        input_values = inputs.input_values.to(self.device)

        # Extract WavLM embeddings
        with torch.no_grad():
            outputs = self.model(**input_values)
            logits = outputs.logits
            # Average over time steps to create a speaker embedding
            embedding = logits.mean(dim=1).squeeze()  # Average over time steps

        return embedding

    def ssim(self, embedding1, embedding2):
        with torch.no_grad():
            ssim = 1 - cosine(embedding1.cpu().numpy(), embedding2.cpu().numpy())
        return ssim

import jiwer
import torch
import torchaudio
from transformers import AutoProcessor, AutoModelForCTC, Wav2Vec2Processor, WavLMForCTC
from scipy.spatial.distance import cosine


ASR_PRETRAINED_MODEL = "facebook/hubert-large-ls960-ft"

def load_asr():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AutoModelForCTC.from_pretrained(ASR_PRETRAINED_MODEL).to(device)
    tokenizer = AutoProcessor.from_pretrained(ASR_PRETRAINED_MODEL)
    models = {"model": model, "tokenizer": tokenizer}
    return models


def wav_to_text(model, wav):
    # Tokenize the input
    inputs = model["tokenizer"](wav, sampling_rate=16000, return_tensors="pt", padding="longest")

    # Fix input shape if necessary
    input_values = inputs.input_values.squeeze(1)  # Squeeze out the extra dimension

    # Move tensors to the GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    input_values = input_values.to(device)

    # Get the model predictions (logits)
    logits = model["model"](input_values).logits
    
    # Get the predicted IDs
    predicted_ids = torch.argmax(logits, dim=-1)
    
    # Decode the predicted IDs into the text (batch_decode returns a list, so we take [0])
    result = model["tokenizer"].batch_decode(predicted_ids)[0]

    return result

def calculate_error(converted, source):
    #perform calcute for word/character error rate
    wer = jiwer.wer(converted, source)
    cer = jiwer.cer(converted, source)

    return wer, cer


def load_wavLM():
    # Load pre-trained WavLM model and processor
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    processor = Wav2Vec2Processor.from_pretrained('patrickvonplaten/wavlm-libri-clean-100h-base-plus')
    model = WavLMForCTC.from_pretrained('patrickvonplaten/wavlm-libri-clean-100h-base-plus').to(device)

    return model, processor

#Extracting embedded value of a wav file (input can be modify later to meet our need)
def extract_embedding(wav_path, processor, model):
    #load wav file
    waveform, sample_rate = torchaudio.load(wav_path)
    # Resample to 16kHz if necessary
    if sample_rate != 16000:
        resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)
        waveform = resampler(waveform)

    # Squeeze the waveform to remove any extra dimensions
    waveform = waveform.squeeze()

    # Process the waveform using the feature extractor
    inputs = processor(waveform, sampling_rate=16000, return_tensors="pt", padding=True)

    # Move inputs to GPU if available
    input_values = inputs.input_values.to("cuda")

    # Extract WavLM embeddings
    with torch.no_grad():
        outputs = model(input_values, output_hidden_states=True)  # Enable hidden states output

        # Use the last hidden state as the embedding
        hidden_states = outputs.hidden_states[-1]  # Last layer's hidden state
        embedding = hidden_states.mean(dim=1).squeeze()  # Average over time steps

    return embedding.cpu()  

def speaker_sim(embedding1, embedding2):
    return 1 - cosine(embedding1.numpy(), embedding2.numpy())


    

import jiwer
import torch
from transformers import Wav2Vec2ForCTC, Wav2Vec2Tokenizer


ASR_PRETRAINED_MODEL = "facebook/wav2vec2-large-960h-lv60-self"

def load_asr():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Wav2Vec2ForCTC.from_pretrained(ASR_PRETRAINED_MODEL).to(device)
    tokenizer = Wav2Vec2Tokenizer.from_pretrained(ASR_PRETRAINED_MODEL)
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
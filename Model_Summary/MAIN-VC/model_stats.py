import torch
from torchinfo import summary
import yaml
import time
from models.model import MAINVC, SpeakerEncoder, ContentEncoder, Decoder  # Import models

# Load config
with open("./config.yaml") as f:
    config = yaml.load(f, Loader=yaml.FullLoader)

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Initialize model modules
speaker_encoder = SpeakerEncoder(**config["SpeakerEncoder"]).to(device)
content_encoder = ContentEncoder(**config["ContentEncoder"]).to(device)
decoder = Decoder(**config["Decoder"]).to(device)
main_model = MAINVC(config).to(device)

# Define input tensors
x = torch.randn(1, 80, 128).to(device)  # Speech feature
x_sf = torch.randn(1, 80, 128).to(device)  # Speaker feature
x_ = torch.randn(1, 80, 128).to(device)  # Speaker reference feature

print("\n===== Speaker Encoder Summary =====")
summary(speaker_encoder, input_data=[x_sf], depth=5)

print("\n===== Content Encoder Summary =====")
summary(content_encoder, input_data=[x], depth=5)

print("\n===== Decoder Summary =====")
mu = content_encoder(x)[0]  # Get mean for decoder input
emb = speaker_encoder(x_sf)  # Get speaker embedding for conditioning
summary(decoder, input_data=[mu, emb], depth=5)

print("\n===== MAINVC Model Summary =====")
summary(main_model, input_data=[x, x_sf, x_], depth=5)

# Measure execution time
start_time = time.time()
mu, log_sigma, emb, emb_, dec = main_model(x, x_sf, x_)
end_time = time.time()

print(f"\nScript execution time: {end_time - start_time:.4f} seconds")


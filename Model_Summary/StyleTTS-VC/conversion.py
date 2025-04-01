import os
from pydub import AudioSegment

# Input and output directories
input_dir = "wav48_silence_trimmed"
output_dir = "wav24_silence_trimmed"

# Create the output directory if it doesn't exist
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Walk through the directory structure
for root, dirs, files in os.walk(input_dir):
    for file in files:
        if file.endswith(".flac"):
            # Construct full input path
            input_path = os.path.join(root, file)

            # Create corresponding output directory
            relative_path = os.path.relpath(root, input_dir)
            output_subdir = os.path.join(output_dir, relative_path)
            if not os.path.exists(output_subdir):
                os.makedirs(output_subdir)

            # Construct full output path (convert .flac to .wav)
            output_path = os.path.join(output_subdir, os.path.splitext(file)[0] + ".wav")

            # Load the audio file
            audio = AudioSegment.from_file(input_path, format="flac")

            # Downsample to 24 kHz
            audio = audio.set_frame_rate(24000)

            # Export the downsampled audio as .wav
            audio.export(output_path, format="wav")
            print(f"Processed: {input_path} -> {output_path}")

print("All files have been downsampled to 24 kHz and saved.")

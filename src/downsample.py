import os
import librosa
import soundfile as sf
import argparse

### RUN WITH: python downsample.py -s [source folder] -k [target kHz]
### MOVE RESULTING DATA TO ./data/VCTK/raw

def downsample_audio(source_folder, target_khz):
    # Define the target folders
    target_folder = source_folder.replace("48", str(target_khz))  # Adjust the folder name based on the target kHz
    
    # Ensure the target base directory exists
    if not os.path.exists(target_folder):
        os.makedirs(target_folder)
    
    # Iterate over each subfolder in the base directory
    for subfolder in os.listdir(source_folder):
        subfolder_path = os.path.join(source_folder, subfolder)
        
        # Skip if not a directory
        if not os.path.isdir(subfolder_path):
            continue
        
        # Create corresponding subfolder in target directory
        target_subfolder_path = os.path.join(target_folder, subfolder)
        if not os.path.exists(target_subfolder_path):
            os.makedirs(target_subfolder_path)
        
        # Process each audio file in the subfolder
        for file in os.listdir(subfolder_path):
            if file.endswith("mic1.flac"):
                file_path = os.path.join(subfolder_path, file)
                
                # Load the audio file
                audio, sr = librosa.load(file_path, sr=None)
                
                # Downsample the audio
                audio_downsampled = librosa.resample(audio, orig_sr=sr, target_sr=target_khz*1000)
                
                # Generate the target file path with .wav extension
                target_file_name = file.replace("_mic1.flac", ".wav")  # Remove '_mic1' and change extension
                target_file_path = os.path.join(target_subfolder_path, target_file_name)
                
                # Save the downsampled audio to the target directory
                sf.write(target_file_path, audio_downsampled, target_khz*1000, format='WAV')


if __name__ == "__main__":
    # Setup command line argument parsing
    parser = argparse.ArgumentParser(description="Downsample FLAC files to a specified sample rate and save as WAV.")
    
    # Optional arguments with flags
    parser.add_argument("-s", "--source_folder", type=str, default="./dataset/DS_10283_3443/VCTK-Corpus-0.92/wav48_silence_trimmed", help="Path to the folder containing the original FLAC files. Default is '/path/to/wav48_silence_trimmed'.")
    parser.add_argument("-k", "--target_khz", type=int, default=24, help="Target sample rate in kHz. Default is 24 kHz.")

    # Parse arguments
    args = parser.parse_args()

    # Call the downsample function with provided arguments
    downsample_audio(args.source_folder, args.target_khz)


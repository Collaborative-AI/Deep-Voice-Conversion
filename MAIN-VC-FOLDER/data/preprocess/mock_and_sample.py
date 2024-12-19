import os
import random
import pickle
import numpy as np
import json  # Import json for handling JSON data

# Mock HyperParameters class
# class HyperParameters:
#     sr = 16000  # Sample rate
#     preemph = 0.97
#     n_fft = 1024
#     n_mels = 80
#     hop_len = 256
#     win_len = 1024
#     f_min = 0

class HyperParameters:
    sr = 16000  # Sample rate
    preemph = 0.97
    n_fft = 2048
    n_mels = 80
    hop_len = 300
    win_len = 1200
    f_min = 80
    
hp = HyperParameters()

# Mock function to read speaker information
def read_speaker_info(speaker_info_path):
    return [f"speaker_{i}" for i in range(1, 11)]  # Simulate 10 speakers

# Mock function to simulate loading a wav file
def load_wav(path, sample_rate):
    return np.random.rand(sample_rate)  # Simulating 1 second of audio

# Mock function to simulate log mel spectrogram
def log_mel_spectrogram(wav, preemph, sample_rate, n_mels, n_fft, hop_len, win_len, f_min):
    T = len(wav) // hop_len  # Number of time steps
    return np.random.rand(T, n_mels)

# Mock function to simulate speaker file paths
def speaker_file_paths(data_dir):
    speaker2filepaths = {}
    for speaker_id in range(1, 11):  # 10 speakers
        speaker = f"speaker_{speaker_id}"
        file_paths = [os.path.join(data_dir, f"{speaker}/audio_{i}.wav") for i in range(1, 6)]  # 5 files per speaker
        speaker2filepaths[speaker] = file_paths
    return speaker2filepaths

def sample_from_mock_data(speaker2filepaths, n_samples=5):
    selected_speakers = random.sample(list(speaker2filepaths.keys()), k=3)  # Sample 3 speakers
    print(f"Selected speakers: {selected_speakers}")

    sampled_files = []

    for speaker_id in selected_speakers:
        if speaker_id in speaker2filepaths:
            # Get available file paths for the speaker
            filepath_list = speaker2filepaths[speaker_id]
            print(f"[DEBUG] File paths for {speaker_id}: {filepath_list}")

            if len(filepath_list) == 0:
                print(f"[MAIN-VC](sample_dataset) No files available for speaker {speaker_id}.")
                continue  # Skip this speaker if there are no files

            # Randomly sample files from the available paths
            sample_utt_index_list = random.choices(range(len(filepath_list)), k=n_samples)
            for index in sample_utt_index_list:
                sampled_files.append(filepath_list[index])
                print(f"Sampled file: {filepath_list[index]}")
        else:
            print(f"Speaker ID {speaker_id} not found in speaker2filepaths.")

    if len(sampled_files) == 0:
        print("[MAIN-VC](sample_dataset) No utterances available for sampling. Exiting.")
    else:
        print(f"[MAIN-VC](sample_dataset) Sampled files: {sampled_files}")

def main():
    # Mock parameters
    data_dir = "mock_data_dir"
    speaker_info_path = "mock_speaker_info.txt"
    output_dir = "output"
    log_dir = os.path.join(output_dir, "log")  # Define log directory
    test_speakers = 3  # Number of test speakers
    test_proportion = 0.2  # 20% test data
    n_utts_attr = 5  # Number of attributes to consider

    # Create the output and log directories if they don't exist
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)  # Ensure log directory exists

    # Logging example
    def log_message(message):
        with open(os.path.join(log_dir, "training.log"), "a") as log_file:
            log_file.write(message + "\n")

    # Log the start of the process
    log_message("[MAIN-VC] Process started.")

    # Read speaker IDs
    speaker_ids = read_speaker_info(speaker_info_path)
    print(f"[MAIN-VC](make_datasets) got {len(speaker_ids)} speakers' ids")
    random.shuffle(speaker_ids)

    train_speaker_ids = speaker_ids[:-test_speakers]
    test_speaker_ids = speaker_ids[-test_speakers:]

    # Output unseen and seen speaker IDs
    with open(os.path.join(output_dir, "unseen_speaker_ids.txt"), "w") as f:
        for id in test_speaker_ids:
            f.write(f"{id}\n")
    with open(os.path.join(output_dir, "seen_speaker_ids.txt"), "w") as f:
        for id in train_speaker_ids:
            f.write(f"{id}\n")

    log_message(f"[MAIN-VC](make_datasets) {len(train_speaker_ids)} train speakers, {len(test_speaker_ids)} test speakers")

    speaker2filepaths = speaker_file_paths(data_dir)

    train_path_list, in_test_path_list, out_test_path_list = [], [], []
    train_speaker2filenames = {}

    # Divide the data from train_speaker into training and test data
    for speaker in train_speaker_ids:
        path_list = speaker2filepaths[speaker]
        random.shuffle(path_list)
        test_data_size = int(len(path_list) * test_proportion)
        train_speaker2filenames[speaker] = path_list[:-test_data_size]
        train_path_list += path_list[:-test_data_size]
        in_test_path_list += path_list[-test_data_size:]

    # Output in-test files
    with open(os.path.join(output_dir, "in_test_files.txt"), "w") as f:
        for path in in_test_path_list:
            f.write(f"{path}\n")

    # Output speaker to filenames
    with open(os.path.join(output_dir, "speaker2filenames.pkl"), "wb") as f:
        pickle.dump(train_speaker2filenames, f)

    # Generate train_samples_128.json
    train_samples_data = {"train_samples": []}
    for speaker in train_speaker_ids:
        for filepath in train_speaker2filenames[speaker]:
            train_samples_data["train_samples"].append(filepath)

    with open(os.path.join(output_dir, "train_samples_128.json"), "w") as json_file:
        json.dump(train_samples_data, json_file)

    # Add paths of test_speakers' speech to out_test
    for speaker in test_speaker_ids:
        path_list = speaker2filepaths[speaker]
        out_test_path_list += path_list

    # Output out-test files
    with open(os.path.join(output_dir, "out_test_files.txt"), "w") as f:
        for path in out_test_path_list:
            f.write(f"{path}\n")

    for dataset_type, path_list in zip(
        ["train", "in_test", "out_test"],
        [train_path_list, in_test_path_list, out_test_path_list],
    ):
        print(f"[MAIN-VC](make_datasets) processed {dataset_type} set, {len(path_list)} files")
        data = {}
        output_path = os.path.join(output_dir, f"{dataset_type}.pkl")
        all_train_data = []

        for i, path in enumerate(sorted(path_list)):
            if i % 1000 == 0 or i == len(path_list) - 1:
                print(f"[MAIN-VC](make_datasets) processed {i} file of {dataset_type} set")
            filename = os.path.basename(path)
            wav = load_wav(path, hp.sr)
            print(wav.shape)
            print(hp.preemph)
            print(hp.sr)
            print(hp.n_mels)
            print(hp.n_fft)
            print(hp.hop_len)
            print(hp.win_len)
            print(hp.f_min)
            mel = log_mel_spectrogram(
                wav,
                hp.preemph,
                hp.sr,
                hp.n_mels,
                hp.n_fft,
                hp.hop_len,
                hp.win_len,
                hp.f_min,
            )
            print(mel.shape)
            exit()
            data[filename] = mel

            if dataset_type == "train" and i < n_utts_attr:
                all_train_data.append(mel)

        # Get mean and std of train set
        if dataset_type == "train":
            all_train_data = np.concatenate(all_train_data)
            mean = np.mean(all_train_data, axis=0)
            std = np.std(all_train_data, axis=0)
            attr = {"mean": mean, "std": std}
            with open(os.path.join(output_dir, "attr.pkl"), "wb") as f:
                pickle.dump(attr, f)

        # Normalization
        for key, val in data.items():
            val = (val - mean) / std
            data[key] = val

        with open(output_path, "wb") as f:
            pickle.dump(data, f)

    # Now you can sample from the mock data
    sample_from_mock_data(speaker2filepaths)  # Pass speaker2filepaths as an argument

if __name__ == "__main__":
    main()

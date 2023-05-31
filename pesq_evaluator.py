import pesq
import os
import librosa
import pypesq

# Specify the root directory containing the original and degraded WAV files to evaluate
root_dir = 'converted/unseen_content_unseen_speaker/'
root_dir = 'converted/seen_content_seen_speaker/'

# Initialize a list to store the PESQ scores
pesq_scores = []

# Loop through the directories containing the WAV files
for subdir, dirs, files in os.walk(root_dir):
    for dir_name in dirs:
        dir_path = os.path.join(subdir, dir_name)
        print('Evaluating files in directory:', dir_path)

        # Find the original and degraded WAV files in the current directory
        orig_file = None
        deg_file = None
        for file_name in os.listdir(dir_path):
            if file_name.endswith('source_gen.wav'):
                orig_file = dir_path + "/" + file_name
            elif file_name.endswith('converted_gen.wav'):
                deg_file = dir_path + "/" + file_name

        # If both original and degraded files are found, compute the PESQ score
        if orig_file and deg_file:
            # Compute the PESQ score for the WAV files
            orig_file, _ = librosa.load(orig_file, sr=16000)
            deg_file, _ = librosa.load(deg_file, sr=16000)

            pesq_score = pesq.pesq(16000, orig_file, deg_file, 'wb')
            
            # Print the PESQ score for the current pair of files
            print('PESQ score for', orig_file, 'and', deg_file, ':', pesq_score)

            # Append the PESQ score to the list of scores
            pesq_scores.append(pesq_score)

# Compute the average PESQ score for all the WAV files
avg_pesq_score = sum(pesq_scores) / len(pesq_scores)

# Print the average PESQ score
print('Average PESQ score:', avg_pesq_score)
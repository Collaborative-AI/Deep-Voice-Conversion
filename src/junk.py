def replace_wav24_to_wav8(source_file_path, output_file_path):
    with open(source_file_path, 'r', encoding='utf-8') as source_file:
        with open(output_file_path, 'w', encoding='utf-8') as output_file:
            for line in source_file:
                modified_line = "raw/"+ line.replace("wav24", "wav8")
                output_file.write(modified_line)

# Example usage
file = 'val'
source_file_path = f'./data/VCTK/processed/{file}_list8.txt'  # Update this to your source file path
output_file_path = f'./data/VCTK/processed/{file}_list8_changed.txt'  # Update this to your desired output file path

replace_wav24_to_wav8(source_file_path, output_file_path)
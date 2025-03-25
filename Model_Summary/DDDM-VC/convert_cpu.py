import os

def replace_cuda_with_cpu(folder_path):
    """
    Recursively replace .to("cpu") with .to('cpu') in all .py files in the specified folder.
    """
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.endswith(".py"):
                file_path = os.path.join(root, file)
                print(f"Processing file: {file_path}")

                # Read the content of the file
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()

                # Replace .to("cpu") with .to('cpu')
                new_content = content.replace('.to("cpu")', '.to("cpu")')

                # Write the modified content back to the file
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(new_content)

                print(f"Replaced .to("cpu") with .to('cpu') in {file_path}")

if __name__ == "__main__":
    folder_path = input("Enter the path to the folder containing Python files: ")
    replace_cuda_with_cpu(folder_path)
    print("All occurrences of .to("cpu") have been replaced with .to('cpu').")

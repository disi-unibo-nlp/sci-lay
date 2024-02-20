import os
import shutil
import argparse

def get_folders_in_path(directory):
    folders = []
    for entry in os.scandir(directory):
        if entry.is_dir():
            folders.append(entry.name)
    return folders

def remove_file(file_path):
    try:
        os.remove(file_path)
        print("File removed successfully.")
    except FileNotFoundError:
        print(f"File not found: {file_path}")
    except PermissionError:
        print(f"Permission denied: {file_path}")
    except Exception as e:
        print(f"Error occurred while removing file: {str(e)}")

def remove_folder(folder_path):
    try:
        shutil.rmtree(folder_path)
        print(f"Folder '{folder_path}' removed successfully.")
    except OSError as e:
        print(f"Error: {folder_path} : {e.strerror}")


def main():
    # Specify the path for which you want to retrieve the folder list
    path = "output"

    folder_list = get_folders_in_path(path)

    for folder in folder_list:

        try:
            if args.remove == "file":
                # Specify the file path you want to delete
                file_path = os.path.join(path, folder, "pytorch_model.bin")
                remove_file(file_path)
            
            else:
                folder_path = os.path.join(path, folder)
                final_folder_path = os.path.join(folder_path, get_folders_in_path(folder_path)[0])
                remove_folder(final_folder_path)
        except:
            pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--remove", default="file")

    args = parser.parse_args()
    main()
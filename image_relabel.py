import os

def relabel_images(folder_path, start_number=263):
    # Get all PNG files (case insensitive)
    files = [f for f in os.listdir(folder_path) if f.lower().endswith('.png')]
    
    # Sort to keep consistent order
    files.sort()

    current_number = start_number

    for filename in files:
        old_path = os.path.join(folder_path, filename)
        new_filename = f"{current_number}.PNG"
        new_path = os.path.join(folder_path, new_filename)

        os.rename(old_path, new_path)
        current_number += 1

    print(f"Renamed {len(files)} files starting from {start_number}")

# ==== USE LIKE THIS ====
folder_path = "normal11"
relabel_images(folder_path)

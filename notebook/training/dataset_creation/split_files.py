import os
import random

# === Parametri ===
input_folder = "C:\\Users\\Andrea\\Desktop\\segmentation-pytorch\\notebook\\training\\dataset_creation\\images"
train_file = "train.txt"
val_file = "val.txt"
test_file = "test.txt"

# === 1️⃣ Ottieni tutti i nomi dei file ===
all_files = os.listdir(input_folder)
all_files = [f for f in all_files if os.path.isfile(os.path.join(input_folder, f))]

# === 2️⃣ Mescola ===
random.shuffle(all_files)

# === 3️⃣ Calcola split ===
total = len(all_files)
train_size = int(0.7 * total)
val_size = int(0.2 * total)
test_size = total - train_size - val_size  # per avere 100%

train_files = all_files[:train_size]
val_files = all_files[train_size:train_size + val_size]
test_files = all_files[train_size + val_size:]

# === 4️⃣ Scrivi i file ===
with open(train_file, "w") as f:
    for name in train_files:
        f.write(os.path.join(input_folder, f"{name}\n"))

with open(val_file, "w") as f:
    for name in val_files:
        f.write(os.path.join(input_folder, f"{name}\n"))

with open(test_file, "w") as f:
    for name in test_files:
        f.write(os.path.join(input_folder, f"{name}\n"))

print(f"Fatto! {train_file}: {len(train_files)}, {val_file}: {len(val_files)}, {test_file}: {len(test_files)}")

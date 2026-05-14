"""
CelebA Preprocessing Script
============================
Crops all 178x218 CelebA images into perfect 256x256 squares.
Uses all available CPU cores for maximum speed.

Usage:
  1. Extract the Kaggle CelebA zip into this folder
  2. Run: python preprocess.py
  3. Upload the output zip to Google Drive
"""
import os
import sys
from pathlib import Path
from PIL import Image
import concurrent.futures
from tqdm import tqdm
import zipfile

# Auto-detect the input folder (Kaggle nests it differently)
POSSIBLE_INPUTS = [
    Path('img_align_celeba'),
    Path('img_align_celeba/img_align_celeba'),
    Path('celeba-dataset/img_align_celeba/img_align_celeba'),
    Path('archive/img_align_celeba'),
    Path('archive/img_align_celeba/img_align_celeba'),
]
OUTPUT_DIR = Path('celeba_256')
ZIP_PATH = Path('celeba_256.zip')


def find_input_dir():
    for p in POSSIBLE_INPUTS:
        if p.exists() and any(f for f in p.iterdir() if f.suffix.lower() == '.jpg'):
            return p
    return None


def process_image(args):
    filename, input_dir = args
    try:
        input_path = input_dir / filename
        output_path = OUTPUT_DIR / filename

        img = Image.open(input_path).convert('RGB')
        # CelebA images are 178x218. Crop 20px off top & bottom -> 178x178 square
        # Then resize to 256x256 (power of 2, required by the model)
        img = img.crop((0, 20, 178, 198)).resize((256, 256), Image.Resampling.LANCZOS)
        img.save(output_path, quality=95)
        return True
    except Exception as e:
        return f"Error: {filename} - {e}"


def main():
    input_dir = find_input_dir()
    if input_dir is None:
        print("ERROR: Cannot find the CelebA images folder!")
        print("Make sure you extracted the Kaggle zip here.")
        print(f"Expected one of: {[str(p) for p in POSSIBLE_INPUTS]}")
        sys.exit(1)

    print(f"Found images in: {input_dir}")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    files = [f.name for f in input_dir.iterdir() if f.suffix.lower() == '.jpg']
    print(f"Found {len(files)} images to process.")

    if len(files) == 0:
        print("Folder is empty!")
        sys.exit(1)

    # Use all available CPU cores
    cpu_count = os.cpu_count() or 4
    print(f"Using {cpu_count} CPU threads...")

    args_list = [(f, input_dir) for f in files]
    errors = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=cpu_count) as executor:
        for result in tqdm(executor.map(process_image, args_list, chunksize=64),
                           total=len(files), unit="img", desc="Cropping"):
            if result is not True:
                errors.append(result)

    if errors:
        print(f"\n{len(errors)} errors occurred:")
        for e in errors[:10]:
            print(f"  {e}")

    processed = len(files) - len(errors)
    print(f"\nCropped {processed} images into '{OUTPUT_DIR}/'")

    # Zip the output
    print(f"Zipping to {ZIP_PATH}...")
    with zipfile.ZipFile(ZIP_PATH, 'w', zipfile.ZIP_STORED) as zf:
        output_files = list(OUTPUT_DIR.iterdir())
        for f in tqdm(output_files, unit="img", desc="Zipping"):
            zf.write(f, arcname=f.name)

    zip_size_mb = ZIP_PATH.stat().st_size / (1024 * 1024)
    print(f"\nDONE! Output: {ZIP_PATH} ({zip_size_mb:.0f} MB)")
    print(f"Upload this file to your Google Drive.")


if __name__ == '__main__':
    main()

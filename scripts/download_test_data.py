import os
from datasets import load_dataset

os.makedirs('test_data/real', exist_ok=True)
os.makedirs('test_data/fake', exist_ok=True)

print("Downloading Real faces from huggan/ffhq_thumbnails...")
try:
    ds_real = load_dataset("huggan/ffhq_thumbnails", split="train", streaming=True)
    real_count = 0
    for item in ds_real:
        img = item['image']
        if img.mode != 'RGB': img = img.convert('RGB')
        img.save(f"test_data/real/ffhq_{real_count}.jpg")
        real_count += 1
        if real_count >= 50: break
    print(f"Downloaded {real_count} real faces.")
except Exception as e:
    print("Failed to download real faces:", e)

print("Downloading Fake faces from huggan/stylegan2-ffhq-1024x1024...")
try:
    ds_fake = load_dataset("huggan/stylegan2-ffhq-1024x1024", split="train", streaming=True)
    fake_count = 0
    for item in ds_fake:
        img = item['image']
        if img.mode != 'RGB': img = img.convert('RGB')
        # Resize to save space and time
        img = img.resize((256, 256))
        img.save(f"test_data/fake/stylegan_{fake_count}.jpg")
        fake_count += 1
        if fake_count >= 50: break
    print(f"Downloaded {fake_count} fake faces.")
except Exception as e:
    print("Failed to download fake faces:", e)

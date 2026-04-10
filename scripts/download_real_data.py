import os
import requests
import time

os.makedirs('test_data/real', exist_ok=True)
print("Downloading real faces from randomuser.me/api/portraits...")

count = 0
for i in range(1, 26):
    urls = [
        f"https://randomuser.me/api/portraits/men/{i}.jpg",
        f"https://randomuser.me/api/portraits/women/{i}.jpg"
    ]
    for url in urls:
        if count >= 50:
            break
        try:
            r = requests.get(url, timeout=5)
            if r.status_code == 200:
                with open(f"test_data/real/real_{count}.jpg", 'wb') as f:
                    f.write(r.content)
                count += 1
        except Exception as e:
            print(f"Failed {url}: {e}")
            
print(f"Successfully downloaded {count} real faces.")

import os
import time
import requests

os.makedirs('test_data/fake', exist_ok=True)

print("Downloading Fake faces from thispersondoesnotexist.com...")
fake_count = 0
headers = {'User-Agent': 'Mozilla/5.0'}

for i in range(50):
    try:
        response = requests.get("https://thispersondoesnotexist.com", headers=headers, timeout=10)
        if response.status_code == 200:
            with open(f"test_data/fake/tpdne_{fake_count}.jpg", 'wb') as f:
                f.write(response.content)
            fake_count += 1
            time.sleep(1) # prevent rate limiting
        else:
            print(f"Failed status code {response.status_code}")
    except Exception as e:
        print("Error downloading fake face:", e)
        
print(f"Downloaded {fake_count} fake faces.")

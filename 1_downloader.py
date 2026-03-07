import requests
import os
import time

threats = {
    "Lion": "Panthera leo",
    "Tiger": "Panthera tigris",
    "Leopard": "Panthera pardus",
    "Elephant": "Elephantidae",
    "Hyena": "Hyaenidae",
    "Bear": "Ursidae",
    "Hippo": "Hippopotamus",
    "Wolf": "Canis lupus"
}

def download_animal_data(animal_name, query, limit=30):
    print(f"\n--- Searching for: {animal_name} ({query}) ---")
    save_dir = f"dataset/raw/{animal_name}"
    os.makedirs(save_dir, exist_ok=True)
    
    # We use 'query' directly as the search term
    url = f"https://www.xeno-canto.org/api/2/recordings?query={query.replace(' ', '%20')}"
    
    try:
        response = requests.get(url, timeout=15)
        data = response.json()
        
        # Check if we got results
        total_found = data.get('numRecordings', 0)
        print(f"Found {total_found} total recordings on Xeno-Canto.")
        
        recordings = data.get('recordings', [])
        download_count = 0
        
        for rec in recordings:
            if download_count >= limit:
                break
            
            # Xeno-Canto sometimes has 'background' species. We want the main one.
            file_url = "httxps:" + rec['file']
            file_id = rec['id']
            file_name = f"{save_dir}/{animal_name}_{file_id}.mp3"
            
            if os.path.exists(file_name):
                download_count += 1
                continue
            
            print(f"Downloading {animal_name} ID: {file_id}...")
            r = requests.get(file_url, stream=True)
            if r.status_code == 200:
                with open(file_name, 'wb') as f:
                    for chunk in r.iter_content(chunk_size=1024):
                        f.write(chunk)
                download_count += 1
                time.sleep(0.2)
        
        return download_count
        
    except Exception as e:
        print(f"Error: {e}")
        return 0

if __name__ == "__main__":
    for animal, query in threats.items():
        count = download_animal_data(animal, query)
        if count == 0:
            print(f"!!! No files found for {animal}. You may need to find manual samples for this one.")
import os
import json
import urllib.request
import time
from urllib.error import HTTPError, URLError
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

BASE_URL = "http://kaldir.vc.in.tum.de/faceforensics/v3/"
FILELIST_URL = BASE_URL + "misc/filelist.json"
METHODS = ["Deepfakes", "Face2Face", "FaceSwap", "NeuralTextures"]
OUTPUT_ROOT = "/workspace/Shashank/FaceForensics++"
MAX_RETRIES = 5
BACKOFF_FACTOR = 2
CONCURRENCY = 16

def download_file_with_retry(url, dest_path):
    if os.path.exists(dest_path) and os.path.getsize(dest_path) > 0:
        return True, "skipped"
    
    os.makedirs(os.path.dirname(dest_path), exist_ok=True)
    
    temp_dest = dest_path + ".tmp"
    retries = 0
    while retries < MAX_RETRIES:
        try:
            urllib.request.urlretrieve(url, temp_dest)
            os.rename(temp_dest, dest_path)
            return True, "downloaded"
        except (HTTPError, URLError, ConnectionResetError) as e:
            retries += 1
            if isinstance(e, HTTPError) and e.code == 404:
                # File genuinely doesn't exist on server (some pairs might not have masks)
                if os.path.exists(temp_dest):
                    os.remove(temp_dest)
                return False, f"404 Not Found: {url}"
            
            sleep_time = BACKOFF_FACTOR ** retries
            time.sleep(sleep_time)
        except Exception as e:
            if os.path.exists(temp_dest):
                os.remove(temp_dest)
            return False, str(e)
            
    if os.path.exists(temp_dest):
        os.remove(temp_dest)
    return False, f"Failed after {MAX_RETRIES} retries"

def main():
    print("Loading filelist from server...")
    try:
        with urllib.request.urlopen(FILELIST_URL) as response:
            file_pairs = json.loads(response.read().decode("utf-8"))
    except Exception as e:
        print(f"Failed to load filelist: {e}")
        return

    # Build full list of files to download
    # 500 pairs -> 1000 filenames per method
    filenames = []
    for pair in file_pairs:
        filenames.append(f"{pair[0]}_{pair[1]}.mp4")
        filenames.append(f"{pair[1]}_{pair[0]}.mp4")

    # We have 4 methods, 1000 files each -> 4000 total downloads
    tasks = []
    for method in METHODS:
        method_url = BASE_URL + f"manipulated_sequences/{method}/masks/videos/"
        method_out_dir = os.path.join(OUTPUT_ROOT, f"manipulated_sequences/{method}/masks/videos")
        for fn in filenames:
            url = method_url + fn
            dest = os.path.join(method_out_dir, fn)
            tasks.append((url, dest))

    print(f"Total masks to verify/download: {len(tasks)}")
    
    downloaded_count = 0
    skipped_count = 0
    failed_tasks = []

    with ThreadPoolExecutor(max_workers=CONCURRENCY) as executor:
        futures = {executor.submit(download_file_with_retry, url, dest): (url, dest) for url, dest in tasks}
        
        for future in tqdm(as_completed(futures), total=len(futures), desc="Downloading masks"):
            url, dest = futures[future]
            try:
                success, status = future.result()
                if success:
                    if status == "downloaded":
                        downloaded_count += 1
                    else:
                        skipped_count += 1
                else:
                    failed_tasks.append((url, dest, status))
            except Exception as e:
                failed_tasks.append((url, dest, str(e)))
                
    print("\n=== Download Summary ===")
    print(f"Successfully verified/skipped: {skipped_count}")
    print(f"Newly downloaded: {downloaded_count}")
    print(f"Failed downloads: {len(failed_tasks)}")
    
    if failed_tasks:
        print("\nSome files failed to download (e.g. 404 or persistent 503). Showing first 10:")
        for url, _, err in failed_tasks[:10]:
            print(f"- {url}: {err}")

if __name__ == "__main__":
    main()

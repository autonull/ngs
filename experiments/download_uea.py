"""Download UEA time series datasets."""
import os, sys, requests, zipfile, numpy as np
from pathlib import Path

data_dir = Path('./data/UEA')
data_dir.mkdir(parents=True, exist_ok=True)

# UEA archive URLs (from timeseriesclassification.com)
datasets = [
    'CharacterTrajectories',
    'EigenWorms', 
    'Heartbeat',
    'JapaneseVowels',
    'Libras',
    'NATOPS',
    'PEMS-SF',
    'RacketSports',
    'SelfRegulationSCP1',
    'SelfRegulationSCP2',
    'SpokenArabicDigits',
    'UWaveGestureLibrary'
]

base_url = "http://www.timeseriesclassification.com/Downloads/"

for ds in datasets:
    zip_path = data_dir / f"{ds}.zip"
    extract_dir = data_dir / ds
    
    if extract_dir.exists():
        print(f"{ds}: already extracted")
        continue
    
    url = f"{base_url}{ds}.zip"
    print(f"Downloading {ds}...")
    try:
        r = requests.get(url, timeout=60)
        r.raise_for_status()
        zip_path.write_bytes(r.content)
        
        with zipfile.ZipFile(zip_path, 'r') as z:
            z.extractall(extract_dir)
        zip_path.unlink()
        print(f"{ds}: extracted to {extract_dir}")
    except Exception as e:
        print(f"{ds}: FAILED - {e}")

print("\nDone. Check data/UEA/")
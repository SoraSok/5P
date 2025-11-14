"""
Script to download the ASL Alphabet dataset using Kaggle API.
Requires KAGGLE_USERNAME and KAGGLE_KEY environment variables.
"""
import os
import subprocess
from pathlib import Path

def load_env_file():
    """Load environment variables from .env file"""
    env_path = Path(__file__).parent.parent / '.env'
    if env_path.exists():
        with open(env_path, 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    key, value = line.split('=', 1)
                    os.environ[key.strip()] = value.strip()

def download_dataset():
    """Download the ASL Alphabet dataset"""
    load_env_file()
    
    username = os.getenv('KAGGLE_USERNAME')
    key = os.getenv('KAGGLE_KEY')
    
    if not username or not key:
        print("Error: KAGGLE_USERNAME and KAGGLE_KEY must be set in .env file")
        return
    
    os.environ['KAGGLE_USERNAME'] = username
    os.environ['KAGGLE_KEY'] = key
    
    print(f"Using Kaggle username: {username}")
    print("Downloading ASL Alphabet dataset...")
    
    data_dir = Path(__file__).parent.parent / 'data'
    data_dir.mkdir(exist_ok=True)
    
    result = subprocess.run(
        ['kaggle', 'datasets', 'download', '-d', 'grassknoted/asl-alphabet'],
        cwd=data_dir,
        env=os.environ.copy()
    )
    
    if result.returncode == 0:
        print("\nDataset downloaded successfully!")
        print("Extracting zip file...")
        
        import zipfile
        zip_path = data_dir / 'asl-alphabet.zip'
        if zip_path.exists():
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(data_dir)
            print("Extraction complete!")
    else:
        print("Failed to download dataset")

if __name__ == '__main__':
    download_dataset()

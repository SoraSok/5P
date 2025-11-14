import os
import subprocess
import shutil
from pathlib import Path

def setup_kaggle_auth():
    """Setup Kaggle authentication"""
    kaggle_dir = Path.home() / '.kaggle'
    kaggle_dir.mkdir(exist_ok=True)
    
    kaggle_json = kaggle_dir / 'kaggle.json'
    
    if not kaggle_json.exists():
        print("ERROR: Kaggle API credentials not found!")
        print("\nPlease follow these steps:")
        print("1. Go to https://www.kaggle.com/account")
        print("2. Scroll down to 'API' section")
        print("3. Click 'Create New API Token'")
        print("4. Download the kaggle.json file")
        print(f"5. Place it in: {kaggle_json}")
        print("\nThen run this script again.")
        return False
    
    print(f"SUCCESS: Kaggle credentials found at: {kaggle_json}")
    return True

def download_dataset(dataset_name, output_dir):
    """Download a Kaggle dataset"""
    try:
        print(f"\nDownloading {dataset_name}...")
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Download dataset
        cmd = f"kaggle datasets download -d {dataset_name} -p {output_dir} --unzip"
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        
        if result.returncode == 0:
            print(f"SUCCESS: Downloaded {dataset_name}")
            return True
        else:
            print(f"ERROR downloading {dataset_name}:")
            print(result.stderr)
            return False
            
    except Exception as e:
        print(f"ERROR downloading {dataset_name}: {e}")
        return False

def organize_datasets():
    """Organize downloaded datasets"""
    print("\nOrganizing datasets...")
    
    # Create main data directory
    data_dir = Path("data/kaggle_datasets")
    data_dir.mkdir(exist_ok=True)
    
    # Move datasets to organized structure
    datasets = [
        {
            "name": "asl-alphabet-test",
            "kaggle_id": "danrasband/asl-alphabet-test",
            "local_dir": data_dir / "asl_alphabet_test"
        },
        {
            "name": "synthetic-asl-alphabet", 
            "kaggle_id": "lexset/synthetic-asl-alphabet",
            "local_dir": data_dir / "synthetic_asl_alphabet"
        }
    ]
    
    success_count = 0
    
    for dataset in datasets:
        print(f"\nProcessing {dataset['name']}...")
        
        # Download dataset
        if download_dataset(dataset['kaggle_id'], str(dataset['local_dir'])):
            success_count += 1
            
            # List contents
            if dataset['local_dir'].exists():
                contents = list(dataset['local_dir'].iterdir())
                print(f"Contents of {dataset['name']}:")
                for item in contents[:10]:  # Show first 10 items
                    print(f"  - {item.name}")
                if len(contents) > 10:
                    print(f"  ... and {len(contents) - 10} more items")
    
    print(f"\nSUCCESS: Downloaded {success_count}/{len(datasets)} datasets")
    return success_count > 0

def main():
    print("Kaggle ASL Dataset Downloader")
    print("=" * 50)
    
    # Check Kaggle authentication
    if not setup_kaggle_auth():
        return
    
    # Download datasets
    if organize_datasets():
        print("\nSUCCESS: All datasets downloaded!")
        print("\nNext steps:")
        print("1. Check the downloaded datasets in data/kaggle_datasets/")
        print("2. Run the dataset converter to extract MediaPipe landmarks")
        print("3. Train the MediaPipe model on the new data")
    else:
        print("\nERROR: Failed to download datasets. Please check your Kaggle credentials.")

if __name__ == "__main__":
    main()

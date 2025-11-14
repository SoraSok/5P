# 5P

_Hackathon repository of a group "5P". The Sign language translator app_

## About

Demo application for reading movements and identifying gestures, automatic translation into the selected language. Currently only has a translation into English, about [n] gestures. Created with the goal of helping and making life and communication easier for deaf and mute people.

## Participants and roles:

**Serbulenco Daniela:** Computer vision + machine learning  
**Rusu Alexandru:** Translations and voice acting  
**Todorova Margarita:** Integration and application logic  
**Boboc Gabriel:** Interface (GUI)  
**Postolachi Dumitru:** The database

## Dataset

This project uses the **ASL Alphabet Dataset** from Kaggle:
- **Source:** [ASL Alphabet Dataset by Grassknoted](https://www.kaggle.com/datasets/grassknoted/asl-alphabet)
- **Description:** A collection of images of alphabets from the American Sign Language, separated into 29 folders representing each letter (A-Z) and additional signs (del, nothing, space).
- **Size:** ~87,000 training images and 29 test images

## Setup

### Prerequisites

- Python 3.x
- Kaggle account with API credentials

### Installation

1. Clone the repository:

```bash
git clone <repository-url>
cd 5P
```

2. Create and activate virtual environment:

```bash
python -m venv venv
.\venv\Scripts\activate  # Windows
source venv/bin/activate  # Linux/Mac
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

4. **Quick Setup (Recommended for team members):**

```bash
python setup.py
```

This will automatically:

- Download all required datasets
- Convert them to MediaPipe landmarks
- Train the model
- Set up everything ready to use

5. Set up Kaggle API credentials:

   - Go to https://www.kaggle.com/account
   - Generate API token (downloads `kaggle.json`)
   - Copy `.env.example` to `.env`
   - Fill in your credentials in `.env`:
     ```
     KAGGLE_USERNAME=your_username
     KAGGLE_KEY=your_api_key
     ```

6. Download the datasets (choose one method):

**Method 1: Using the Kaggle download script (recommended)**

```bash
python scripts/download_kaggle_datasets.py
```

This will automatically download:

- ASL Alphabet Test dataset (danrasband/asl-alphabet-test)
- Synthetic ASL Alphabet dataset (lexset/synthetic-asl-alphabet)

**Method 2: Manual Kaggle download**

```bash
# Download ASL Alphabet Test
kaggle datasets download -d danrasband/asl-alphabet-test -p data/kaggle_datasets/asl_alphabet_test --unzip

# Download Synthetic ASL Alphabet
kaggle datasets download -d lexset/synthetic-asl-alphabet -p data/kaggle_datasets/synthetic_asl_alphabet --unzip
```

**Method 3: Convert existing dataset to MediaPipe landmarks**

If you have the original ASL alphabet dataset:

```bash
python scripts/convert_dataset.py --dataset_path data/asl_alphabet_train/asl_alphabet_train
```

### Dataset Structure

- **Training:** ~87,000 images across 29 classes (A-Z + del, nothing, space)
- **Test:** 29 images (1 per class)
- Images are organized by gesture class in separate folders

## Data Collection

To collect your own ASL hand landmark data for training:

1. **Run the tool:**

   ```bash
   python scripts/mediapipe_hand_detector.py
   ```

2. **Select a class:** Press letter keys (A-Z) to select which ASL sign you want to collect

3. **Capture samples:** Press `c` to capture hand landmark data

4. **Save data:** Press `s` to save collected data

5. **Quit:** Press `q` or `ESC` to exit

### Available Commands:

- **Letter keys (A-Z)**: Select ASL sign class
- **`c`**: Capture sample (extract landmarks)
- **`s`**: Save all collected data
- **`q` or `ESC`**: Quit
- **`r`**: Reset current class

The collected data is saved to `data/mediapipe_data/collected_landmarks.pkl` and can be used to retrain the model.

## Usage

1. **Run the main application:**

   ```bash
   python asl_app.py
   ```

2. **Collect new data (optional):**

   - Click "Collect Data" button in the main app, or
   - Run `python scripts/mediapipe_hand_detector.py` directly

3. **Retrain model with new data:**
   ```bash
   python scripts/landmark_classifier.py --data_file data/mediapipe_data/collected_landmarks.pkl
   ```

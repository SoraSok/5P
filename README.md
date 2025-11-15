# American Sign Language Detection using CNN

## Project Overview

This is the **unihack2025** project that uses **Convolutional Neural Networks (CNN)** and **Deep Learning** to recognize and translate American Sign Language (ASL) gestures into written text and speech.

### Key Features

- **Real-time ASL Gesture Recognition**: Captures ASL gestures using webcam or camera device
- **CNN-based Deep Learning Model**: Trained on a large dataset of ASL images for accurate gesture recognition
- **Text Translation**: Converts recognized ASL gestures into written text
- **Speech Output**: Provides audio translation of recognized gestures
- **User Interface**: Intuitive GUI for easy interaction
- **Image Processing Module**: Extracts features from captured hand gestures
- **Database of ASL Signs**: Comprehensive collection of ASL signs and corresponding text translations
- **Training Module**: Allows the model to be updated and retrained with new data

### Project Goal

The project aims to **improve communication and accessibility** between the deaf and hearing communities by providing an accurate, real-time tool that recognizes and translates ASL gestures into written text, making ASL more accessible to people who are not familiar with it.

---

## Installation & Setup

### In order to Run the program locally

1. Create a virtual environment

   ```bash
   virtualenv venv
   ```

2. Activate it

   ```bash
   Source venv\bin\activate
   venv\Scripts\Activate.ps1
   ```

3. Install all packages and dependencies with

   ```bash
   pip install -r requirements.txt
   ```

4. For collecting Dataset,

   ```bash
   python data_collection_final.py
   ```

5. Finally, Run the application with GUI
   ```bash
   cd "Final Project\Source Code"
   python final_pred.py
   ```

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

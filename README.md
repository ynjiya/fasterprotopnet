# Acceleration of Interpretable Deep Neural Networks

This repository contains the proposed acceleration method – simplified calculation of the loss function – in the research **"Acceleration of Interpretable Deep Neural Networks"** (ISO).

## Repository Overview
This repository includes code implementations for:
- **ProtoPNet** and its experiment results across different batch sizes (stored in the `/saved_models` directory).
- **ProtoViT** and **TesNet**, with experiment results only for batch size **80** (stored in their respective `/saved_models` directories).
- **EvalProtoPNet**, a folder containing the evaluator for **ProtoPNet**.


## Setting Up the Environment
To install the required dependencies for each of these models, run in their subfolders:
```bash
pip install -r requirements.txt
```

---

## Instructions for Preparing the Data
1. Download the dataset **CUB_200_2011.tgz** from [Caltech Vision](http://www.vision.caltech.edu/visipedia/CUB-200-2011.html).
2. Unpack the dataset:
   ```bash
   tar -xvzf CUB_200_2011.tgz
   ```
3. Crop the images using `crop_dataset.py` and bounding box information from `bounding_boxes.txt` (included in the dataset):
   ```bash
   python crop_dataset.py
   ```
4. Split the cropped images into training and test sets using `split_dataset.py` and `train_test_split.txt` (included in the dataset):
   ```bash
   python split_dataset.py
   ```
5. Move the cropped images into appropriate directories:
   ```bash
   mv cropped_train ./datasets/cub200_cropped/train_cropped/
   mv cropped_test ./datasets/cub200_cropped/test_cropped/
   ```
6. Augment the training dataset using `img_aug.py`:
   ```bash
   python img_aug.py
   ```
   This will create an augmented training set in:
   ```
   ./datasets/cub200_cropped/train_cropped_augmented/
   ```

---

## Instructions for Training the Models
### Step 1: Settings
Edit `settings.py` to set appropriate paths for dataset and other hyperpameters

### Step 2: Run Training Scripts
- **For training the original models:**
  ```bash
  bash main.sh
  ```
  or run directly:
  ```bash
  python main.py
  ```
- **For training the accelerated models:**
  ```bash
  bash main_custom.sh
  ```
  or run directly:
  ```bash
  python main_custom.py
  ```

---

## Experiment Setup
All experiments were conducted on an **NVIDIA Tesla T4 GPU with 16GB RAM**.

---

## Acknowledgments
This work builds upon the following repositories:
- **ProtoPNet**: [https://github.com/cfchen-duke/ProtoPNet]()
- **BetterProtoPNet** - fork of ProtoPNet with data preprocessing scripts: [https://github.com/KrystianJachna/BetterProtoPNet]()
- **ProtoViT**: [https://github.com/Henrymachiyu/ProtoViT]()
- **TesNet**: [https://github.com/JackeyWang96/TesNet]()
- **EvalProtoPNet**: [https://github.com/hqhQAQ/EvalProtoPNet]()

---

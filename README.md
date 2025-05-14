# ATNet and TurbNet Dataset Evaluation

This repository is designed to evaluate the performance and generalization capabilities of [AT-Net](https://github.com/rajeevyasarla/AT-Net) and [TurbNet](https://github.com/VITA-Group/TurbNet/tree/main) on different publicly available turbulence mitigation datasets.

## 📌 Objective

The goal of this repository is to **test the impact of training ATNet and TurbNet** on three different datasets and analyze how well the network generalizes across varying types of turbulence distortion.

## 📂 Datasets

We use **three public datasets** for training and evaluation:

1. **OTIS_Dataset**  
2. **HeatChamber Dataset**  
3. **ATSync Dataset**

Each dataset contains synthetic turbulence-degraded imagery, offering diverse degradation patterns. We selected **1800 images** across these datasets for training and **200 unseen images** (not included in training) for evaluation.

## 🛠️ Structure

### 📁 Folders

You will find **three main training folders**, one for each dataset:
- `train_otis/`
- `train_heatchamber/`
- `train_atsync/`

Each folder contains:
- `train.py` – a customized training script based on ATNet’s original implementation.
- `data_loader.py` – defines the `TrainData` class tailored to the folder structure and GT mapping of the corresponding dataset.

Additionally:
- `test_set/` – holds **200 evaluation images** (mixed from all three datasets) used to validate performance on unseen data.

### 🔄 Ground Truth Handling

Each training loader (`TrainData` class) ensures that the correct **ground truth (GT)** image is paired with each turbulence-distorted input image. GT image mapping is handled per-dataset depending on their internal folder structure.

## ▶️ How to Use

1. Clone the original [AT-Net repository](https://github.com/rajeevyasarla/AT-Net) or [TurbNet repository](https://github.com/VITA-Group/TurbNet/tree/main).
2. Replace the original `train.py` and dataset loader with the ones provided in the respective training folders.
3. Run training from any of the three dataset folders:
   ```bash
   python train.py

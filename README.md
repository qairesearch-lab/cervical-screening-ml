# A Variance-Aware Benchmarking Workflow for Reproducible Deep Learning in Cervical Cytology

## Project Description

This repository contains the source code for the manuscript "A Variance-Aware Benchmarking Workflow for Reproducible Deep Learning in Cervical Cytology: A Pilot Study on SIPaKMeD". The code implements deep learning models for cervical cell classification using ResNet-50 with channel attention mechanisms.

This submission package corresponds to the image-only manuscript workflow.
Handcrafted morphological-feature preprocessing explored in separate internal
experiments is intentionally not included in the executable path here.

## Repository Structure

```
├── README.md              # This file
├── requirements.txt       # Python dependencies
├── .gitignore             # Git ignore file
└── src/
    ├── __init__.py
    ├── preprocess.py      # Data preprocessing script
    ├── train.py           # Model training and evaluation script
    ├── run_experiment.py  # One-click experiment runner
    └── generate_figures.py # Figure generation script
```

## Environment Setup

### Requirements

- Python >= 3.8
- PyTorch >= 1.9.0
- torchvision >= 0.10.0
- numpy >= 1.19.0
- pandas >= 1.3.0
- scikit-learn >= 0.24.0
- scipy >= 1.7.0
- matplotlib >= 3.3.0
- seaborn >= 0.11.0

### Installation

```bash
pip install -r requirements.txt
```

## Data Availability

The experiments use the **SIPaKMeD** cervical cell dataset.

- **Source**: https://www.cs.uoi.gr/~marina/sipakmed.html
- **Samples**: 4,049 cell images
- **Classes**: 5 (superficial-intermediate, parabasal, koilocytes, dyskeratotic, metaplastic)
- **Image size**: Variable (resized to 224x224 during preprocessing)

### Data Preparation

1. Download the SIPaKMeD dataset manually from the official source
2. Extract to `./data/raw/` directory with the following structure:

```
data/raw/
├── superficial-intermediate/
├── parabasal/
├── koilocytes/
├── dyskeratotic/
└── metaplastic/
```

## Usage

### One-Click Run (Reference Entry Point)

```bash
python -m src.run_experiment
```

This script is intended to provide a reference entry point for the full experimental workflow, including data preprocessing, model training, and result generation.

⚠️ Note:

- The script assumes that the dataset has been prepared as described in the Data section.
- Execution may require adjustment depending on the local environment (e.g., GPU availability, file paths).
- For full control and reproducibility, users are encouraged to follow the step-by-step instructions below.

### Step-by-Step Run

#### Step 1: Data Preprocessing

```bash
python -m src.preprocess
```

Expected outputs include:

- Validated and loaded raw images
- Preprocessed images (resized to 224x224)
- Train/val/test splits (70/15/15) saved to `./data/splits/`

#### Step 2: Model Training

```bash
python -m src.train
```

Expected outputs include:

- Three trained models: Baseline ResNet-50, +dual-pooling (layer4), +dual-pooling (avgpool)
- Cross-validation results (3 seeds × 5 folds)
- Final model evaluation on held-out test set
- Statistical analysis results

#### Step 3: Generate Figures

```bash
python -m src.generate_figures
```

This script is intended to generate all paper figures:

- Figure 1: Model Comparison Bar Chart
- Figure 2: Confusion Matrix
- Figure 3: Training Curves
- Figure 4: Class-wise Performance
- Figure 5: Statistical Analysis
- Figure 6: Model Architecture Diagram

### Output

Results are expected to be saved to:

```
results/experiment_results/
├── cv_summary_baseline.json
├── cv_summary_se.json
├── cv_summary_se_avgpool.json
├── test_summary.json
├── statistical_analysis.json
└── REPORT.md
```

Figures are expected to be saved to:

```
figures/
├── figure1_model_comparison.png
├── figure2_confusion_matrix.png
├── figure3_training_curves.png
├── figure4_class_performance.png
├── figure5_statistical_analysis.png
└── figure6_architecture.png
```

## Model Architecture

Three ResNet-50 based models are evaluated:

1. **Baseline**: Standard ResNet-50 (ImageNet pretrained)
2. **+dual-pooling (layer4)**: ResNet-50 with channel attention after layer4
3. **+dual-pooling (avgpool)**: ResNet-50 with channel attention after avgpool

Channel attention module implements channel-wise feature recalibration via:

- Squeeze: Global average pooling + Global max pooling (dual-pooling)
- Excitation: FC(2048 -> 128) -> ReLU -> FC(128 -> 2048) -> Sigmoid
- Scale: Channel-wise multiplication

## Training Configuration

| Parameter | Value |
|-----------|-------|
| Image size | 224 x 224 |
| Batch size | 32 |
| Epochs | 15 |
| Learning rate | 1e-4 |
| Optimizer | AdamW |
| Weight decay | 1e-4 |
| Scheduler | CosineAnnealingLR |
| Dropout | 0.5 |
| SE reduction | 16 |
| CV folds | 5 |
| Seeds | [42, 52, 62] |

## Reproducibility

### Main Results

- Primary results are reported from 3 seeds × 5-fold cross-validation (15 runs total)
- Mean ± standard deviation is calculated across all 15 runs
- Cross-validation is performed only on train/val set (85% of data)

### Test Set Usage

- Test set (15% of data) is held out during model development and hyperparameter tuning
- Test set is evaluated only once per seed after final model selection
- Test set results serve as final validation of model performance

### Randomness Control

- All random seeds are set for reproducibility
- Data splits are fixed across runs using the specified seeds
- Model initialization is controlled via PyTorch's random seed

## Notes

- This repository is intended to provide the code and configuration required to reproduce the reported experiments.
- The `results/` and `figures/` directories are automatically created in the project root.
- Pre-trained weights are not provided in this repository.
- All experiments were conducted on standard hardware (GPU recommended).

## Citation

Citation information will be updated after publication.

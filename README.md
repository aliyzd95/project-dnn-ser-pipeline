# Speech Emotion Recognition (SER) using Deep Neural Networks

This repository contains a complete machine learning pipeline for Speech Emotion Recognition (SER) using Deep Neural Networks (DNNs). The pipeline is built with reproducibility and experiment tracking in mind, utilizing DVC for data and pipeline management, and MLflow (integrated with DagsHub) for experiment logging. Please see the [DagsHub repository](https://dagshub.com/aliyzd95/project-dnn-ser-pipeline).

## 🔍 Overview

The goal of this project is to classify emotions from speech audio using a Convolutional Neural Network (CNN)-based architecture. The system supports five emotion classes and is trained and evaluated using stratified 5-fold cross-validation.

## 🧩 Project Structure

```
├── data/
│   ├── modified_shemo.json     # Preprocessed metadata
│   └── npy/                    # Numpy feature arrays
├── models/                     # Trained Keras models per fold
├── results/                    # Evaluation reports and confusion matrices
├── runs/                       # Run ID files for MLflow linkage
├── src/
│   ├── preprocess.py           # Feature extraction and preprocessing
│   ├── train.py                # Training with Optuna hyperparameter tuning
│   └── test.py                 # Evaluation and logging of test results
├── dvc.yaml                    # DVC pipeline configuration
├── params.yaml                # Hyperparameters and path configuration
└── README.md                   # Project documentation
```

## ⚙️ Pipeline Stages

### 1. Preprocessing

- Extracts features from the Shemo dataset and saves them in `data/npy/`.
- Controlled by parameters in `params.yaml > preprocess`.

### 2. Training

- Performs 5-fold cross-validation.
- Uses Optuna for hyperparameter tuning within each fold.
- Logs the best model for each fold to MLflow and DagsHub.
- Controlled by parameters in `params.yaml > train`.

### 3. Testing

- Loads each fold’s best model and evaluates it on the test set.
- Logs metrics and confusion matrices to MLflow.
- Controlled by parameters in `params.yaml > test`.

## 📦 Reproducibility

This project uses [DVC](https://dvc.org/) to track data and pipeline stages. To reproduce the entire pipeline:

```bash
git clone https://github.com/your-username/project-dnn-ser-pipeline.git
cd project-dnn-ser-pipeline
dvc pull           # fetch data (if remote is configured)
dvc repro          # run the pipeline from scratch
```

## 📊 Experiment Tracking

We use [MLflow](https://mlflow.org/) integrated with [DagsHub](https://dagshub.com/) to track experiments, parameters, models, and metrics. Each fold in the cross-validation is logged as a nested run.

## 🔧 Requirements

- Python 3.10+
- TensorFlow, NumPy, Scikit-learn, Optuna, DVC, MLflow, etc.

Install dependencies using:

```bash
pip install -r requirements.txt
```

## 📁 Configuration

Hyperparameters and paths are managed via `params.yaml`. Example:

```yaml
train:
  inputs_path: data/npy/
  models_path: models/
  runs_path: runs/
  n_trials: 20
```

## 📬 Contact

For questions or feedback, please open an issue or contact [aliyzd95](https://github.com/aliyzd95).

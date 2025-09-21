# Beijing Air Quality Forecasting Project

## Overview
This project focuses on **time series forecasting** of PM2.5 concentrations in Beijing using air quality data, as part of a Kaggle competition. The goal is to minimize Root Mean Squared Error (RMSE) by leveraging Recurrent Neural Networks (RNNs), specifically LSTMs and GRUs, with advanced feature engineering and hybrid architectures. The implementation is contained in a Jupyter Notebook (`air_quality_finalized_notebook_humanized (1).ipynb`), designed to be reproducible, well-documented, and aligned with machine learning best practices.

### Objectives
- Predict hourly PM2.5 levels using meteorological and temporal features.
- Explore data, preprocess, engineer features, and experiment with models.
- Achieve competitive RMSE through systematic hyperparameter tuning and hybrid models.

## Dataset
The dataset consists of hourly air quality and meteorological data from Beijing (2010–2014), sourced from a Kaggle competition (or similar, e.g., UCI Beijing PM2.5 dataset).

- **Files**:
  - `train.csv`: Training data (~43,800 rows) with columns: `No`, `DEWP`, `TEMP`, `PRES`, `Iws`, `Is`, `Ir`, `datetime`, `cbwd` (wind direction), `pm2.5` (target).
  - `test.csv`: Test data for predictions, same features excluding `pm2.5`.
  - `sample_submission.csv`: Template for Kaggle submission with `row ID` and predicted `pm2.5`.
- **Features**:
  - Numerical: Dew point (`DEWP`), temperature (`TEMP`), pressure (`PRES`), wind speed (`Iws`), snow (`Is`), rain (`Ir`).
  - Categorical: Wind direction (`cbwd`: NW, SE, cv).
  - Temporal: `datetime` (hourly).
  - Target: `pm2.5` (PM2.5 concentration in µg/m³).
- **Challenges**: Missing PM2.5 values (~2%), skewness in target, strong seasonality.

## Project Structure
- `air_quality_finalized_notebook_humanized (1).ipynb`: Main Jupyter Notebook with data exploration, preprocessing, feature engineering, modeling, experimentation, and submission generation.
- `train.csv`, `test.csv`, `sample_submission.csv`: Input datasets (assumed stored in Google Drive: `/content/drive/MyDrive/air-quality-forecasting/`).
- `enhanced_submission.csv`: Output file with Kaggle predictions.
- `README.md`: This file.

## Setup Instructions
### Prerequisites
- **Environment**: Google Colab (recommended) or local Jupyter with Python 3.7+.
- **Dependencies**:
  ```bash
  pandas
  numpy
  matplotlib
  seaborn
  tensorflow>=2.4
  scikit-learn
 **Install in collab**
```bash  
!pip install pandas numpy matplotlib seaborn tensorflow scikit-learn


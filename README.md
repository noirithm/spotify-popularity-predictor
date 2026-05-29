# Spotify Popularity Predictor

Predicting Spotify track popularity from audio features using machine learning.

## Overview

This project builds a machine learning pipeline to predict the popularity score of a Spotify track based on its audio characteristics such as energy, instrumentalness, speechiness, and genre. The dataset contains 114,000 tracks sourced from Kaggle.

## Dataset

Download from Kaggle and place in a `data/` folder as `dataset.csv`.

https://www.kaggle.com/datasets/maharshipandya/-spotify-tracks-dataset

## Results

| Model | RMSE | STD |
|---|---|---|
| Linear Regression | 18.50 | 0.11 |
| Decision Tree | 18.15 | 0.09 |
| KNN | 17.87 | 0.07 |
| XGBoost | 15.20 | 0.10 |
| Random Forest | **13.95** | **0.04** |

Random Forest achieved the best performance with an RMSE of 13.95, meaning predictions are off by approximately 14 popularity points on average across a 0 to 100 scale.

## Key Findings

Genre was the strongest predictor of popularity with an importance score of 0.25, indicating that certain genres are consistently more popular than others. Duration was the second most important feature at 0.08, suggesting that track length has a meaningful influence on popularity.

Instrumentalness, acousticness, danceability, speechiness, loudness, valence, energy, tempo, and liveness all showed similar importance scores ranging from 0.05 to 0.08. This indicates that popularity is influenced by a combination of audio characteristics rather than a single dominant musical feature.

Features such as key, explicitness, mode, and time signature had near-zero predictive power and contributed minimally to model performance.

Linear models failed to capture the relationship between audio features and popularity, confirming that the underlying patterns are non-linear. Tree-based models dominated the comparison as a result.

## Project Structure

```
spotify-popularity-predictor/
├── notebook.ipynb        
├── requirements.txt      
├── .gitignore            
└── README.md             
```

## How to Run

Install dependencies:

```bash
pip install -r requirements.txt
```

Launch the notebook:

```bash
jupyter notebook notebook.ipynb
```

## Libraries

pandas, numpy, scikit-learn, xgboost, matplotlib, jupyter

## Limitations

The model achieves moderate accuracy with an RMSE of 14 on a 0 to 100 scale. Popularity is influenced by factors outside this dataset such as artist fame, marketing, playlist placement, and release timing. These external factors limit the ceiling of any audio features based model.
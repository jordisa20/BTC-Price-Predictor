# Bitcoin Price / Trend Predictor

A comparative study of **Elastic Net**, **XGBoost**, and **LSTM deep learning** for forecasting next-day Bitcoin closing prices and price trends.

---

## Project Overview

Cryptocurrency markets are highly volatile and influenced by external events, making price prediction a challenging problem.  
This project evaluates three different AI model families to determine which approach generalizes best:

| Model Type | Model | Category |
|------------|--------|----------|
| Linear ML | Elastic Net | Baseline |
| Ensemble ML | XGBoost | Tree-based |
| Deep Learning | LSTM | Neural Network |

Our goal is to predict:
- **Next-day Bitcoin closing price**
- **Price trend direction (up or down)**

---

## Dataset
- **4008 daily Bitcoin records**
- **Period:** 2014 → 2025
- **109 engineered technical indicators**
  (RSI, MACD, moving averages, volatility measures, bull/bear regime flags, etc.)

**Target variable:** `target_close_1d`  
→ Tomorrow’s closing price (close at t+1)

---

## Methodology
1. Sort by date and convert time-series to numerical dataset
2. Create prediction target (`close` shifted by -1 day)
3. Forward-fill and back-fill missing values
4. Time-based split (no randomization):
   - 70% Training
   - 15% Validation
   - 15% Testing
5. Compare three AI models using identical metrics

---

## Evaluation Metrics
All models were evaluated using:

| Metric | Meaning |
|--------|---------|
| MAE | Mean Absolute Error |
| RMSE | Root Mean Squared Error |
| R² | Coefficient of Determination |
| Directional Accuracy | % of times model predicted up/down correctly |

Directional Accuracy is critical because many financial traders care more about trend prediction than price precision.

---

## Results Summary

| Model | MAE (Test) | RMSE (Test) | R² (Test) | Directional Accuracy |
|--------|-----------|-------------|-----------|----------------------|
| **Elastic Net** | **2024** | **2639** | **0.985** | **51.5%** |
| XGBoost | 21903 | 29146 | −0.842 | 48.7% |
| XGBoost (Tuned) | 21052 | 28466 | −0.757 | 50.7% |
| LSTM | 81040 | 83839 | −14.239 | 47.0% |

### Key Findings
- **Elastic Net generalized best** despite being the simplest model
- XGBoost overfit volatility spikes even after tuning
- LSTM **failed to generalize** from historical windows due to crypto noise
- **Complexity ≠ better performance in financial time-series forecasting**

---

## Visualizations

| Plot | Purpose |
|------|---------|
| Elastic Net - Actual vs Predicted | Shows best model closely follows real market behavior |
| XGBoost - Actual vs Predicted | Demonstrates overfitting and instability |
| LSTM - Loss Curve | Shows training worked, but validation stayed high |

### Elastic Net - Actual vs Predicted
![Elastic Net Prediction](notebooks/reports/figures/elastic_net_prediction.png)

<p align="center"><i>Elastic Net closely tracks price volatility and general trends</i></p>

### XGBoost - Actual vs Predicted
![XGBoost Prediction](notebooks/reports/figures/xgboost_prediction.png)

<p align="center"><i>XGBoost prediction curve demonstrates overfitting and chasing spikes instead of general patterns.</i></p>

### Actual vs Predicted - Elastic Net vs XGBoost
![Elastic vs XGBoost](notebooks/reports/figures/elastic_vs_xgboost.png)

<p align="center"><i>Elastic Net generalizes significantly better than XGBoost across the same test interval.</i></p>

### LSTM - Training Loss Curve
![LSTM Loss](notebooks/reports/figures/lstm_loss.png)

<p align="center"><i>LSTM training loss decreases while validation loss remains high indicating underfitting / poor generalization.</i></p>


---

## Reproducibility

### Requirements
Install dependencies:

```bash
pip install -r requirements.txt
```
Open in VS Code / Jupyter:
```
notebooks/bitcoin_main.ipynb
```
Run top to bottom 

### Repository Syructure
```
bitcoin-price-prediction-ai/
│
├── data/                    
├── notebooks/                 
│   └── bitcoin_main.ipynb
├── reports/
│   ├── figures/               
├── requirements.txt
└── README.md
```
---
### Conclusion

This project demonstrates that:

```More complex models do not always outperform simpler ones in cryptocurrency forecasting.```

Elastic Net outperformed XGBoost and LSTM due to its robustness against noise and extreme volatility.

## Authors
### **Team Members:**
- 
-
-
-
-

***Course: CAP 4630 — Introduction to Artificial Intelligence***  
***Semester: Fall 2025***
---

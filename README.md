# ✈️ Flight Delay Prediction using Machine Learning

## Overview

This project develops a Machine Learning model to predict flight delays for **United Airlines (UA)** flights operating at **John F. Kennedy International Airport (JFK), New York**.

The goal is to identify flights that are likely to arrive or depart **15 minutes or more behind schedule**, enabling proactive operational decisions and reducing disruption costs.

---

## Business Problem

Undetected flight delays can lead to:

* Increased staffing and operational costs
* Passenger rebooking expenses
* Schedule disruptions and cascading delays
* Reduced customer satisfaction

### Solution

A Machine Learning-based early warning system that predicts whether a flight will be delayed before departure.

---

## Project Objectives

* Predict flight delays ≥ 15 minutes
* Improve operational planning
* Support airline decision-making
* Identify key drivers of delays

---

## Machine Learning Approach

### Model

* **Algorithm:** LightGBM (Gradient Boosting)
* **Problem Type:** Binary Classification

### Target Variable

| Value | Description              |
| ----- | ------------------------ |
| 0     | On Time (< 15 min delay) |
| 1     | Delayed (≥ 15 min delay) |

---

## Dataset

**Source:** BTS On-Time Performance Dataset

### Scope

* Airline: United Airlines (UA)
* Airport: JFK (New York)
* Historical Data: 1987–2022

### Input Files

```text
inputs/
├── Detailed_Statistics_Arrivals.csv
└── Detailed_Statistics_Departures.csv
```

---

## Project Structure

```text
project/
│
├── flight_delay_prediction.py
├── config.yaml
├── requirements.txt
├── README.md
│
├── inputs/
│   ├── Detailed_Statistics_Arrivals.csv
│   └── Detailed_Statistics_Departures.csv
│
└── outputs/
    ├── flight_delay_analysis.png
    ├── model.txt
    └── metrics.json
```

---

## Installation

Clone the repository and install dependencies:

```bash
pip install -r requirements.txt
```

---

## Usage

Run the training and evaluation pipeline:

```bash
python flight_delay_prediction.py
```

---

## Generated Outputs

The script automatically creates the following artifacts:

### Visualization

```text
outputs/flight_delay_analysis.png
```

Exploratory analysis and delay distribution charts.

### Trained Model

```text
outputs/model.txt
```

Serialized LightGBM model.

### Evaluation Metrics

```text
outputs/metrics.json
```

Performance metrics including AUC and F1-score.

### Console Report

Includes:

* Model performance summary
* Classification report
* Business insights
* Recommended operating threshold

---

## Features

### Included Features

* Month
* Day of Week
* Hour of Day
* Season
* Peak Hour Indicator
* Historical Airline Performance
* Route Information (Categorical Feature)

### Excluded Features (Data Leakage Prevention)

The following features are intentionally excluded because they are not known at prediction time:

* Weather Delay
* Carrier Delay
* NAS Delay
* Security Delay
* Late Aircraft Delay

---

## Model Evaluation

### Metrics

* AUC-ROC
* Precision
* Recall
* F1-Score
* Confusion Matrix

### Threshold Optimization

Prediction thresholds are optimized using the **F1-Score** to balance precision and recall.

---

## Key Findings

### Delay Patterns

* Flight delays are more frequent during afternoon and evening hours.
* Weekend operations show higher delay rates.
* Historical airline performance is one of the strongest predictors.

### Business Impact

Accurate delay prediction enables:

* Better crew scheduling
* Improved gate allocation
* Earlier passenger communication
* Reduced operational disruption

---

## Future Improvements

### Planned Enhancements

* ✅ Streamlit Dashboard for live monitoring
* ✅ FastAPI REST API deployment
* ✅ Time-Series Cross Validation
* ✅ Automated Feature Store Integration
* ✅ Real-Time Flight Data Integration
* ✅ SHAP Explainability Dashboard
* ✅ Model Monitoring and Drift Detection

---

## Technologies Used

* Python
* Pandas
* NumPy
* Scikit-Learn
* LightGBM
* Matplotlib
* Seaborn
* YAML

---

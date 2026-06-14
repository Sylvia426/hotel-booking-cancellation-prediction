# Hotel Booking Cancellation Prediction

Predicting hotel booking cancellations on **119,390 real reservation records**
using machine learning, and uncovering the booking patterns that drive
cancellation behavior.

---

## Overview

Booking cancellations are one of the biggest sources of lost revenue in the
hospitality industry. This project analyzes a real-world dataset of hotel
reservations to (1) understand what drives cancellations and (2) build a
classifier that predicts whether a booking will be canceled — giving hotels a
tool to manage overbooking, deposit policy, and targeted retention.

**Dataset:** [Hotel Booking Demand](https://www.kaggle.com/datasets/jessemostipak/hotel-booking-demand)
— 119,390 bookings from a City Hotel (Lisbon) and a Resort Hotel (Algarve),
spanning July 2015 – August 2017, with 32 features per booking.

---

## Key Results

- Trained a **Random Forest classifier** that predicts cancellations with
  **~83% accuracy**, with balanced precision and recall.
- Identified the **top drivers of cancellation** via feature-importance analysis:
  **deposit type**, **lead time**, and **average daily rate (ADR)**.
- Surfaced a counter-intuitive insight: **non-refundable deposits had a ~99%
  cancellation rate** — far higher than no-deposit (28%) or refundable (22%)
  bookings — suggesting deposit policy interacts with high-risk segments.
- Mapped **seasonal demand**: both hotels peak in **August**, with City Hotel
  bookings far more volatile (cancellation rate 41.7% vs. 27.8% for the Resort Hotel).
- Found that **Portugal (domestic) guests** drove the largest share of both
  bookings and cancellations, informing geo-targeted policy.

---

## Methods

**Data preparation**
- Handled missing values (mode imputation for `country`, rounded-mean for
  `children`, `0`-fill for `agent`/`company`, dropped phantom bookings with no guests).
- Removed ADR outliers to stabilize the distribution.

**Modeling (scikit-learn)**
- Built a reproducible **`Pipeline`** combining a **`ColumnTransformer`**
  (one-hot encoding for categoricals, imputation for numerics) with the classifier.
- Compared **Random Forest** and **Decision Tree** classifiers using an 80/20
  train/test split and cross-validation.
- **Deliberately excluded leakage-prone fields** (`reservation_status`,
  `assigned_room_type`, etc.) so the model generalizes to *future* bookings.
- Extracted feature importances to interpret the drivers of cancellation.

---

## Tech Stack

`Python` · `pandas` · `scikit-learn` · `matplotlib` · `seaborn`

---

## Repository Contents

```
.
├── README.md
└── hotel_cancellation_prediction.ipynb   # EDA + modeling
```

---

## How to Run

```bash
# 1. Download the dataset from the Kaggle link above and place
#    hotel_bookings.csv in the same folder as the notebook.

# 2. Install dependencies
pip install pandas numpy scikit-learn matplotlib seaborn

# 3. Open the notebook
jupyter notebook hotel_cancellation_prediction.ipynb
```

---

## Possible Extensions

- Hyperparameter tuning (GridSearch) and gradient-boosted models (XGBoost / LightGBM).
- K-Means clustering to define data-driven customer segments.
- Sentiment analysis on guest reviews to enrich the feature set.
- Deploying the model behind a simple API for real-time cancellation scoring.

---

## Author

**Ruimin (Sylvia) Pei** — M.S. Business Analytics & AI, Johns Hopkins University
[LinkedIn](https://www.linkedin.com/in/ruiminpei/)

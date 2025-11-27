<!-- Time Series Analysis Template -->


## 1. Data & Transformations

- **Variable:** Torque
- **Transformation:** Box-Cox Transformation
- **Lambda (Guerrero):** 1.988653

---

## 2. Stationarity Analysis

- **Trend Strength:** 0.219506

| Test | Statistic | Critical Value | Conclusion|
|-----|-----------|---------|-----------|
|ADF| -5.952 | -2.876714 | Rejected|
|KPSS| 0.7673 | 0.463 | Rejected|

**Stationarity Conclusion:** difference / non-stationary

## 3. Cutoff Thresholds (Tran & Reed)

| Plot_Type   |   Lag |   Cut_T_Value |
|:------------|------:|--------------:|
| ACF         |     1 |      2.62411  |
| PACF        |     1 |      2.64238  |
| ACF         |     2 |      0.528673 |
| PACF        |     2 |      1.81056  |


- **Minimum Threshold Value:** Minimum Lag: 2 / Value: 0.528673
- **Suggested Model:** ARIMA(0, 1, 1)
---

## 4. Model Selection

| Model | AIC | Ljung-Box p |
|-------|------|------------|
|(0, 1, 1)|-583.3409|0.051835|
|(0, 1, 2)|-584.1929|0.249887|
|(0, 1, 3)|-583.0461|0.383952|


**Selected Model:** **ARIMA(0, 1, 2)**

---

## 5. Final Model Diagnostics

- **Number of Observation:** 195
- **AIC:** -584.19287

|        |           0 |
|:-------|------------:|
| ma.L1  | -0.841999   |
| ma.L2  | -0.113632   |
| sigma2 |  0.00276033 |
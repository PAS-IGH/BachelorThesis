<!-- Time Series Analysis Template -->


## 1. Data & Transformations

- **Variable:** Torque
- **Transformation:** Box-Cox Transformation
- **Lambda (Guerrero):** 1.988788

---

## 2. Stationarity Analysis

- **Trend Strength:** 0.650079

| Test | Statistic | Critical Value | Conclusion|
|-----|-----------|---------|-----------|
|ADF| -3.1143 | -3.433979 | Failed to reject|
|KPSS| 0.3849 | 0.146 | Rejected|

**Stationarity Conclusion:** difference / non-stationary

## 3. Cutoff Thresholds (Tran & Reed)

| Plot_Type   |   Lag |   Cut_T_Value |
|:------------|------:|--------------:|
| ACF         |     1 |       1.73431 |
| PACF        |     1 |       1.74227 |
| ACF         |     2 |       1.07465 |
| PACF        |     2 |       1.70479 |


- **Minimum Threshold Value:** Minimum Lag: 2 / Value: 1.074646
- **Suggested Model:** ARIMA(0, 1, 1)
---

## 4. Model Selection

| Model | AIC | Ljung-Box p |
|-------|------|------------|
|(0, 1, 1)|-967.2735|0.04258|
|(0, 1, 2)|-975.4492|0.449504|
|(0, 1, 3)|-973.5984|0.510378|


**Selected Model:** **ARIMA(0, 1, 2)**

---

**5. Final Model Diagnostics**

- **Number of Observation:** 195
- **AIC:** -975.449202

|        |            0 |
|:-------|-------------:|
| ma.L1  | -0.632134    |
| ma.L2  | -0.235811    |
| sigma2 |  0.000369311 |
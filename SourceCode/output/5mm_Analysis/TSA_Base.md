<!-- Time Series Analysis Template -->


## 1. Data & Transformations

- **Variable:** Torque
- **Transformation:** Box-Cox Transformation
- **Lambda (Guerrero):** 1.989755

---

## 2. Stationarity Analysis

- **Trend Strength:** 0.72809

| Test | Statistic | Critical Value | Conclusion|
|-----|-----------|---------|-----------|
|ADF| -2.9983 | -3.434106 | Failed to reject|
|KPSS| 0.3133 | 0.146 | Rejected|

**Stationarity Conclusion:** difference / non-stationary

## 3. Cutoff Thresholds (Tran & Reed)

| Plot_Type   |   Lag |   Cut_T_Value |
|:------------|------:|--------------:|
| ACF         |     1 |       1.67962 |
| PACF        |     1 |       1.68708 |
| ACF         |     2 |       1.26159 |
| PACF        |     2 |       1.93137 |


- **Minimum Threshold Value:** Minimum Lag: 2 / Value: 1.26159
- **Suggested Model:** ARIMA(0, 1, 1)
---

## 4. Model Selection

| Model | AIC | Ljung-Box p |
|-------|------|------------|
|(0, 1, 1)|-769.0858|0.002748|
|(0, 1, 2)|-775.4659|0.034029|
|(0, 1, 3)|-774.8019|0.144904|


**Selected Model:** **ARIMA(0, 1, 3)**

---

## 5. Final Model Diagnostics**

- **Number of Observation:** 195
- **AIC:** -774.801877

|        |          0 |
|:-------|-----------:|
| ma.L1  | -0.659332  |
| ma.L2  | -0.265842  |
| ma.L3  |  0.0866165 |
| sigma2 |  0.0010288 |
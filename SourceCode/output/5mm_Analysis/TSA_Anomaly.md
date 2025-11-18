<!-- Time Series Analysis Template -->


## 1. Data & Transformations

- **Variable:** Torque
- **Transformation:** Box-Cox Transformation
- **Lambda (Guerrero):** -0.891621

---

## 2. Stationarity Analysis

- **Trend Strength:** 0.659477

| Test | Statistic | Critical Value | Conclusion|
|-----|-----------|---------|-----------|
|ADF| -5.0859 | -3.439584 | Rejected|
|KPSS| 0.0724 | 0.146 | Failed to reject|

**Stationarity Conclusion:** trend / non-stationary

## 3. Cutoff Thresholds (Tran & Reed)

| Plot_Type   |   Lag |   Cut_T_Value |
|:------------|------:|--------------:|
| ACF         |     1 |      1.24834  |
| PACF        |     1 |      1.25378  |
| ACF         |     2 |      0.977568 |
| PACF        |     2 |      0.852838 |


- **Minimum Threshold Value:** Minimum Lag: 2 / Value: 0.852838
- **Suggested Model:** ARIMA(1, 0, 0)
---

## 4. Model Selection

| Model | AIC | Ljung-Box p |
|-------|------|------------|
|(1, 0, 0)|-322.917|0.411536|
|(2, 0, 0)|-323.8222|0.507024|
|(3, 0, 0)|-323.6669|0.663085|


**Selected Model:** **ARIMA(2, 0, 0)**

---

## 5. Final Model Diagnostics**

- **Number of Observation:** 156
- **AIC:** -323.822195

|        |           0 |
|:-------|------------:|
| const  | -0.191565   |
| x1     |  0.00137218 |
| ar.L1  |  0.227065   |
| ar.L2  |  0.136789   |
| sigma2 |  0.00695378 |
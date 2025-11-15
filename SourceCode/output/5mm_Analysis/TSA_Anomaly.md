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

**Stationarity Conclusion:** stationary / stationary

## 3. Cutoff Thresholds (Tran & Reed)

| Plot_Type   |   Lag |   Cut_T_Value |
|:------------|------:|--------------:|
| ACF         |     1 |       2.74396 |
| PACF        |     1 |       2.77036 |
| ACF         |     2 |       2.09963 |
| PACF        |     2 |       1.42231 |


- **Minimum Threshold Value:** Minimum Lag: 2 / Value: 1.422311
- **Suggested Model:** ARIMA(2, 0, 0)
---

## 4. Model Selection

| Model | AIC | Ljung-Box p |
|-------|------|------------|
|(1, 0, 0)|-296.6|0.001772|
|(2, 0, 0)|-306.6653|0.09097|
|(3, 0, 0)|-310.9912|0.297964|
|(4, 0, 0)|-309.6205|0.286237|


**Selected Model:** **ARIMA(3, 0, 0)**

---

**5. Final Model Diagnostics**

- **Number of Observation:** 156
- **AIC:** -310.991238

|        |           0 |
|:-------|------------:|
| const  | -0.0831992  |
| ar.L1  |  0.31514    |
| ar.L2  |  0.201945   |
| ar.L3  |  0.200844   |
| sigma2 |  0.00745167 |
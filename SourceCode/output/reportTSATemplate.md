<!-- Time Series Analysis Template -->


## 1. Data & Transformations

- **Variable:** {var_name}
- **Transformation:** Box-Cox Transformation
- **Lambda (Guerrero):** {lambda}

---

## 2. Stationarity Analysis

- **Trend Strength:** {trend_strength}

| Test | Statistic | Critical Value | Conclusion|
|-----|-----------|---------|-----------|
{test_table_adf}
{test_table_kpss}

**Stationarity Conclusion:** {stationary_type} / {stationary_status}

## 3. Cutoff Thresholds (Tran & Reed)

{threshold_table}


- **Minimum Threshold Value:** Minimum Lag: {Min_Lag} / Value: {Minimum_value}
- **Suggested Model:** {sugg_Model}
---

## 4. Model Selection

| Model | AIC | Ljung-Box p |
|-------|------|------------|
{candidate_table}

**Selected Model:** **{chosen_model}**

---

## 5. Final Model Diagnostics

- **Number of Observation:** {n_obs}
- **AIC:** {optimal_model_aic}

{coefficient_table}
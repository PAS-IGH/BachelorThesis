## 1. MAE Evaluation of the Generated Models
- **Out of Sample MAE Base Model:** {base_to_base}
- **Out of Sample MAE Anomaly Model:**{ano_to_ano}
- **Out of Sample MAE Base Model to Anomalous Observation:** {base_to_ano}
- **Out of Sample MAE Anomaly Model to Base Observation:** {ano_to_base}

## 2. Confusion Matrix
|                   | Predicted Normal | Predicted Anomaly  |
|-------------------| -----------------|--------------------|
| **True Normal**   |   {TN}           |    {FP}            |
| **True Anomaly**  |   {FN}           |    {TP}            |

- **Precision:**{precision}
- **Recall:**{recall}
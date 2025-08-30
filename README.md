SmartGridGuard: AI-Powered Short-Term Energy Load Forecasting and Anomaly Detection
Authors:
Sahana Samanta, Sristi Saha

---

Inspiration
Buildings account for nearly one-third of global energy use and emissions. Accurate short-term energy load forecasting is crucial for balancing demand, integrating renewable energy, and preventing grid instability.
We were inspired by the idea of applying AI-based forecasting and anomaly detection to electricity consumption. The thought that our system could contribute — even in a small way — toward energy efficiency, reliability, and sustainability motivated us to build SmartGridGuard.

---

What It Does

1. SmartGridGuard is an AI-driven forecasting and anomaly detection system designed for smart grids and building energy management.
2. Uses Transformer-based time series models (TSFM) to forecast short-term electricity demand.
3. Applies contextual anomaly detection using Isolation Forest on forecast residuals combined with calendar and temporal features (day of week, weekends, holidays, hour of day).
4. Identifies abnormal load patterns such as spikes, drops, or unexpected behavior indicating equipment faults, inefficiencies, or unusual consumption.
5. Provides a Streamlit dashboard for interactive visualization:

   * Forecast vs. Actual Demand
   * Highlighted Anomalies in real-time

In essence: Predict → Detect → Visualize → Act

---

How We Built It

1. Dataset and Preprocessing

   * Used the Electricity Load Forecasting dataset from Kaggle as the primary source of historical demand data.
   * Retained contextual features: dayOfWeek, weekend, holiday, lag features (week\_X-2, week\_X-3, week\_X-4), moving average (MA\_X-4), hourOfDay, weather (T2M\_toc).
   * Dropped unnecessary identifiers like Holiday\_ID to avoid data leakage.
   * Data was normalized and converted into time-series sequences suitable for Transformer models.

2. Forecasting with Transformer (TSFM)

   * Trained a Transformer-based model to predict short-term electricity demand.
   * Configured with multiple encoder layers, multi-head attention, positional encoding, and dropout to capture temporal dependencies.
   * Evaluation metrics: RMSE, MAE, MAPE, R².

3. Anomaly Detection

   * Residuals (difference between actual and predicted values) were computed.
   * Residuals combined with contextual features (dayOfWeek, weekend, holiday, etc.) were passed into Isolation Forest.
   * Applied score percentile thresholding instead of a fixed contamination rate, flagging anomalies above a chosen percentile (e.g., top 1.5%).

4. Visualization and Deployment

   * Full pipeline visualized via Streamlit dashboard.
   * Default Kaggle dataset integrated for evaluation.
   * Upload new datasets, run TSFM model for forecasting, apply anomaly detection.
   * View results in real time with intuitive plots (Actual vs Forecast and flagged anomalies).

---

Challenges We Ran Into

1. Data quality and variability: Noisy and seasonal energy data makes anomaly detection tricky.
2. False positives: Peaks during holidays/weekends initially looked anomalous; mitigated with contextual features.
3. Model generalization: Avoided overfitting of Transformer using dropout, validation splits, and attention-based architecture.
4. Thresholding anomalies: Score percentile thresholding worked better than fixed contamination for Isolation Forest.
5. Integration with Streamlit: Ensured seamless training → forecasting → anomaly detection → plotting.

---

Accomplishments We’re Proud Of

1. Built a fully functional AI pipeline from raw energy data to anomaly visualization.
2. Achieved accurate short-term load forecasting with Transformer TSFM.
3. Designed a robust anomaly detection module combining residuals and context features.
4. Developed an interactive Streamlit app supporting retraining, visualization, and user-friendly anomaly insights.
5. Created a system deployable in real-world smart grid monitoring environments.

---

What We Learned

1. How to integrate deep learning Transformers with classical ML anomaly detection for time series.
2. Contextual features are critical: not every spike is an anomaly.
3. Streamlit is powerful for rapid prototyping and visualization.
4. Real-world ML systems need both predictive performance and interpretability.
5. Correct handling of scaling, windowing, and inverse transforms is key in time-series forecasting.

---

What’s Next for SmartGridGuard

1. Expand datasets: commercial, residential, industrial building types.
2. Incorporate pre-trained TSFM when datasets are provided; for now, demonstrated workflow trained from scratch.
3. Real-time deployment: streaming data integration for continuous monitoring.
4. Explainable AI (XAI): use SHAP/feature attribution to explain flagged anomalies.
5. Integration with IoT and smart meters: deploy directly in energy management systems.
6. Actionable recommendations: beyond flagging anomalies, suggest corrective actions (for example, check HVAC, reschedule equipment usage).

---

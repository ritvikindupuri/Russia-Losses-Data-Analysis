# Operational Intelligence Pipeline: "Army of Drones" Statistical Forensics

## Executive Summary
This project establishes a scalable data intelligence pipeline designed to transform raw, unstructured operational reporting into high-fidelity strategic assets. Focusing on the "Army of Drones" initiative (Summer 2023 – Summer 2024), the analysis addresses the critical challenge of interpreting noisy, episodic operational data.

Rather than relying on manual review to distinguish between baseline attrition and significant strategic events, I implemented a rigorous statistical framework to automate this distinction. By deploying **Unsupervised Machine Learning (Isolation Forest)**, the system successfully identified critical operational surges—most notably in June 2024—that deviated more than **3 standard deviations** from the statistical mean.

This project demonstrates the ability to translate static CSV logs into a dynamic intelligence dashboard, proving that drone assets are primarily deployed during synchronized combined-arms offensives rather than isolated skirmishes.

## Data Source & Attribution
* **Dataset:** "Army of Drones" (Russian Losses attributed to project assets).
* **Stakeholders:** General Staff of the Armed Forces, State Special Communications Service, Ministry of Digital Transformation.
* **Attribution:** Data sourced via United24 and officially released under **CC BY-NC-SA 4.0**.
* **Scope:** Strictly isolates losses attributed to the "Army of Drones" initiative to ensure attribution integrity.
* **File Used:** `russia_losses.csv` (Structured extraction from weekly reports).

## Technology Architecture
* **Data Pipeline:** **Pandas** (ETL, Vectorized Feature Engineering, Time-Series conversion).
* **Machine Learning:** **Scikit-Learn** (Isolation Forest for Anomaly Detection, StandardScaler for normalization).
* **Visualization:** **Matplotlib & Seaborn** (Dual-axis time series and correlation heatmaps).
* **Environment:** Python 3.10+

## Methodology: The Data Science Lifecycle

### 1. Ingestion & Feature Engineering
Raw operational logs often contain irregularities. I implemented a robust imputation strategy to treat `NaN` values as "zero recorded activity," effectively preserving the integrity of aggregate sums. I then engineered the `Total_Equipment` composite metric, aggregating heterogeneous hardware (Tanks, APVs, Artillery) into a scalar vector to allow for macro-trend analysis.

### 2. Exploratory Data Analysis (EDA)
I deployed Pearson correlation matrices to quantify linear relationships between asset classes, statistically confirming unit composition during engagements.

### 3. Algorithmic Deep Dive (Unsupervised Learning)
To eliminate human bias in defining "high intensity" conflict phases, I utilized an **Isolation Forest** algorithm.
* **Logic:** Unlike distance-based methods (e.g., K-Means), Isolation Forest isolates anomalies by randomly splitting feature values. Rare events require fewer splits to isolate.
* **Tuning:** Applied `StandardScaler` to normalize distributions (preventing high-magnitude Personnel counts from overpowering low-magnitude Radar systems) and set a contamination rate of 0.1 to reflect the episodic nature of strategic pushes.

## Visual Analysis & Strategic Insights

### Equipment Attrition Composition
Understanding the tactical focus of drone operators is critical for assessing doctrine.

<p align="center">
  <img src=".assets/Distribution of Equipment Losses.png" alt="Distribution of Equipment Losses" width="800"/>
  <br>
  <b>Figure 1: Distribution of Equipment Losses</b>
</p>

The data reveals a clear strategic doctrine. **Strongpoints** (static fortifications) account for the largest share of kinetic effects at **44.1%**, followed by **Vehicles** (15.9%) and **APVs** (11.6%).
* **Strategic Insight:** The primary mission profile is the dismantling of fixed defensive infrastructure (bunkers/trenches) followed by the degradation of rear-echelon mobility. This high volume of Strongpoint destruction correlates with the widespread use of FPV drones as precision artillery.

### Temporal Operational Tempo
Analyzing the rhythm of the conflict reveals how drone operations are synchronized with broader maneuvers.

<p align="center">
  <img src=".assets/Dual-Axis Time Series (Personnel Vs Equipment).png" alt="Temporal Trends" width="800"/>
  <br>
  <b>Figure 2: Dual-Axis Time Series (Personnel vs. Equipment)</b>
</p>

The time-series data shows a high-amplitude "pulse" pattern rather than a linear climb.
* **Strategic Insight:** The tight synchronization between **Personnel** (Dark Blue) and **Equipment** (Light Blue) lines confirms that drone strikes are integrated into **combined-arms maneuvers**. The simultaneous spikes indicate that mechanized pushes (Equipment) occur in parallel with the neutralization of infantry support (Personnel), contradicting the hypothesis that drone activity is limited to asymmetric harassment.

### Asset Correlation Analysis
To understand unit composition, I analyzed the statistical relationship between different target types.

<p align="center">
  <img src=".assets/Correlation Heatmap.png" alt="Correlation Heatmap" width="800"/>
  <br>
  <b>Figure 3: Pearson Correlation Heatmap</b>
</p>

The heatmap reveals fundamental rules of engagement.
* **Strong Correlation (r > 0.90):** `Total_Equipment` vs. `Vehicles`/`Cannons`. This confirms these assets are the primary drivers of high-intensity loss volumes.
* **Weak Correlation (r < 0.5):** High-Value Targets (e.g., `Radio Equipment`) show low correlation with mass-casualty events. This suggests electronic warfare assets are targeted in separate, precision-based kill chains (deep penetration sorties) that operate independently of the main battle line.

## Advanced Analytics: Automated Anomaly Detection
Using the Isolation Forest model, I mathematically defined "Operational Anomalies" to detect strategic inflection points without human bias.

<p align="center">
  <img src=".assets/Automated Event Detection via Isolation Forest.png" alt="Anomaly Detection" width="800"/>
  <br>
  <b>Figure 4: Automated Event Detection via Isolation Forest</b>
</p>

The model successfully flagged specific weeks where loss volumes defied statistical probability (marked in Red):
* **The Autumn Surge (Oct 16, 2023):** Marked the end of a defensive lull.
* **The Statistical Extreme (June 24, 2024):** The model identified a massive outlier with **1,581 Equipment Losses** (>3σ deviation). This represents the single largest anomaly in the dataset, suggesting a catastrophic front collapse or a coordinated swarm offensive that dictates the upper bound of the model's threshold.

## Conclusion
This project validates that data science methodologies can transform operational reporting from a retrospective accounting exercise into a predictive strategic asset. By transitioning from simple counting to algorithmic analysis, we mathematically verified the doctrinal integration of drone warfare within combined-arms maneuvers.

The architecture developed here serves as a foundational prototype for a real-time intelligence microservice, capable of ingesting daily field reports and triggering automated alerts the moment operational anomalies—such as the massive June 2024 surge—are detected.

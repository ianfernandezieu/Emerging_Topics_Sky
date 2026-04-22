# Report Outline

**Assigned to:** H3 - Report Writing and Interpretation Lead
**Target length:** ~20 pages (excluding references and appendices)
**Status:** TODO H3

---

## 1. Executive Summary (~1 page)

TODO H3: Write after all other sections are complete. Summarize:

- [ ] The research question and why it matters
- [ ] Data sources used and the congestion pressure target
- [ ] Key methodology (baselines, time-series, ML, geospatial)
- [ ] Main findings (which model performed best, which features matter most)
- [ ] Primary recommendation and limitations

---

## 2. Introduction and Problem Statement (~2 pages)

TODO H3: Draft with input from H1's domain research.

- [ ] Why airport congestion forecasting matters (stakeholder perspective: airport ops, airlines, passengers)
- [ ] Context on Madrid-Barajas: traffic volume, hub status, known congestion challenges
- [ ] Research question: Can we forecast congestion pressure 1h and 3h ahead using public data?
- [ ] Brief overview of approach and project scope
- [ ] What this project does NOT claim to do (not a replacement for ATC, not airline-level delay prediction)

---

## 3. Data Sources and Quality Assessment (~2-3 pages)

TODO H3: Coordinate with H2 (data quality) and C1 (data engineering).

- [ ] Overview table of all data sources with coverage period, granularity, and access method
- [ ] OpenSky: what was collected, time range, any gaps or limitations
- [ ] Open-Meteo: weather variables used, coverage, quality checks
- [ ] FlightRadarAPI: role as live enrichment layer, limitations of unofficial wrapper
- [ ] Calendar/metadata: holidays, airport reference data
- [ ] Data quality summary: missingness, outliers, timezone handling, validation against Aena
- [ ] TODO H2: Insert data quality findings here

---

## 4. Methodology (~5-6 pages)

### 4.1 Target Definition (~1 page)

TODO H3: Explain with input from C2.

- [ ] How the Airport Congestion Pressure Score (ACPS) is constructed
- [ ] Formula: movements, baseline comparison, standardization
- [ ] Classification thresholds (Low/Medium/High) and justification
- [ ] Why this target is appropriate vs. alternatives (e.g., official delay minutes)

### 4.2 Time-Series Features (~1 page)

TODO H3: Describe with input from C2.

- [ ] Lag features: which lags and why (1h, 2h, 3h, 6h, 12h, 24h, 48h, 168h)
- [ ] Rolling statistics: means, standard deviations, window sizes
- [ ] Ratio features: current vs. historical same-hour baseline
- [ ] Calendar features: hour, day-of-week, weekend, month, holiday flags

### 4.3 Weather Features (~0.5 page)

TODO H3: Describe with input from C1/C2.

- [ ] Variables: temperature, precipitation, wind speed/direction, pressure, humidity, weather code
- [ ] Encoding decisions: wind direction as sin/cos, weather code as categorical
- [ ] Interaction features: weather x peak-hour if used

### 4.4 Geospatial Analysis (~1 page)

TODO H3: Describe with input from C3.

- [ ] Spatial scope: concentric rings around MAD (0-15km, 15-40km, 40-80km)
- [ ] Density analysis: how nearby flight counts are computed
- [ ] Visualization approach: folium maps, hexbin, heatmaps
- [ ] How spatial features (if any) feed into the model

### 4.5 Modeling Choices (~1.5 pages)

TODO H3: Describe with input from C2/C3.

- [ ] Train/validation/test split: chronological, 70/15/15
- [ ] Baselines: previous hour, same hour previous day, same hour previous week
- [ ] Classical time-series: SARIMAX configuration and rationale
- [ ] ML models: HistGradientBoosting (regressor + classifier), any others
- [ ] Hyperparameter approach: defaults vs. tuned, and why
- [ ] Evaluation metrics: MAE, RMSE for regression; F1, precision, recall for classification

---

## 5. Results (~5-6 pages)

### 5.1 EDA Findings (~1.5 pages)

TODO H3: Write after notebooks 05 and 06 produce outputs.

- [ ] Temporal patterns: hourly, daily, weekly, seasonal traffic profiles
- [ ] Weather correlations with congestion
- [ ] Holiday and special period effects
- [ ] TODO: Insert actual EDA figures from notebooks

### 5.2 Baseline vs. Advanced Models (~2 pages)

TODO H3: Write after notebooks 08, 09, and 10 produce outputs.

- [ ] Model comparison table (all models, both horizons, key metrics)
- [ ] Which model won and by how much over baselines
- [ ] t+1 vs. t+3 performance comparison (does accuracy degrade with horizon?)
- [ ] Regression results narrative
- [ ] Classification results narrative (confusion matrices, class-level metrics)
- [ ] TODO: Insert actual model comparison table and figures

### 5.3 Feature Importance (~1 page)

TODO H3: Write after C3 produces importance/SHAP outputs.

- [ ] Top 10 most important features for regression and classification
- [ ] Role of weather features vs. temporal features
- [ ] Any surprising findings or non-obvious predictors
- [ ] TODO: Insert actual feature importance plots

### 5.4 Spatial Findings (~1 page)

TODO H3: Write after notebook 06 produces geospatial outputs.

- [ ] Traffic density patterns around MAD during low vs. high congestion
- [ ] Approach/departure concentration areas
- [ ] How spatial patterns differ by time of day or weather conditions
- [ ] TODO: Insert actual geospatial figures (maps)

---

## 6. Limitations and Future Work (~1.5 pages)

TODO H3: Write honestly. Coordinate with H2 for data limitations.

- [ ] Data limitations: coverage period, API reliability, unofficial data sources
- [ ] Methodological limitations: no causal claims, potential confounders, aggregation level
- [ ] Target limitations: ACPS is a proxy, not an official congestion metric
- [ ] Model limitations: no deep learning explored, single airport only
- [ ] Future work: second airport validation, real-time dashboard, longer history, official data access
- [ ] TODO H2: Insert specific data quality limitations here

---

## 7. Conclusions and Recommendations (~1 page)

TODO H3: Write last, after results are finalized.

- [ ] Restate the research question and whether it was answered
- [ ] Key takeaway: what works for congestion forecasting at MAD
- [ ] Practical recommendation for stakeholders
- [ ] Final honest assessment of the project's contribution

---

## 8. References (~1 page)

TODO H3: Format all citations from bibliography.md in APA 7th edition.

- [ ] TODO H1: Provide finalized bibliography
- [ ] TODO H3: Format and cross-reference all in-text citations

---

## 9. Contribution Statement (~0.5 page)

TODO H3: Pull from contribution_statement.md.

- [ ] TODO ALL: Finalize individual contributions before submission

---

## Page Budget Summary

| Section | Estimated Pages |
|---------|----------------|
| Executive Summary | 1 |
| Introduction | 2 |
| Data Sources & Quality | 2.5 |
| Methodology | 5.5 |
| Results | 5.5 |
| Limitations & Future Work | 1.5 |
| Conclusions | 1 |
| References | 1 |
| Contribution Statement | 0.5 |
| **Total** | **~20.5** |

---

## Figures and Tables Checklist

TODO H3: Track which figures/tables are needed and their status:

- [ ] Table: Data source summary (Section 3)
- [ ] Figure: Hourly traffic profile (Section 5.1)
- [ ] Figure: Weekly/seasonal patterns (Section 5.1)
- [ ] Figure: Weather vs. congestion scatter/box plots (Section 5.1)
- [ ] Table: Model comparison - regression metrics (Section 5.2)
- [ ] Table: Model comparison - classification metrics (Section 5.2)
- [ ] Figure: Confusion matrix for best classifier (Section 5.2)
- [ ] Figure: Actual vs. predicted time-series plot (Section 5.2)
- [ ] Figure: Feature importance bar chart (Section 5.3)
- [ ] Figure: SHAP summary plot if available (Section 5.3)
- [ ] Figure: Geospatial density map - low congestion (Section 5.4)
- [ ] Figure: Geospatial density map - high congestion (Section 5.4)

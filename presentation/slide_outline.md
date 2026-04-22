# Presentation Slide Outline

**Assigned to:** H4 - Presentation, Design, and Q&A Lead
**Target duration:** 10-15 minutes (approx. 1-1.5 minutes per slide)
**Status:** TODO H4

---

## Slide 1: Why Airport Congestion Matters

**Title:** The Cost of Airport Congestion

**Key Points:**
- Airport congestion causes cascading delays affecting airlines, passengers, and operations
- European air traffic is growing and hub airports face increasing capacity pressure
- Early warning of congestion periods enables better resource allocation and planning
- Our question: Can we forecast when congestion will peak at a major European hub?

**Suggested Visual:** A striking statistic or infographic on European flight delays / congestion costs (source: Eurocontrol CODA report or similar)

**Speaker:** TODO H4

---

## Slide 2: Why Madrid-Barajas

**Title:** Madrid-Barajas: A High-Traffic European Hub

**Key Points:**
- One of Europe's busiest airports by passenger volume
- Hub for Iberia and Air Europa with structured bank/wave schedules
- 4 runways, 5 terminals, complex operational environment
- Clear seasonal and daily traffic patterns make it ideal for forecasting study

**Suggested Visual:** Airport map or aerial photo of MAD with key stats overlay (terminals, runways, annual passengers)

**Speaker:** TODO H4

---

## Slide 3: Data Sources and Target Definition

**Title:** Building a Congestion Signal from Public Data

**Key Points:**
- Core data: OpenSky (flight movements), Open-Meteo (weather), Nager.Date (holidays)
- Enrichment: FlightRadarAPI (live snapshots), OurAirports (metadata)
- Target: Airport Congestion Pressure Score (ACPS) combining movements and baseline ratios
- Classification: Low / Medium / High congestion (60th / 85th percentile thresholds)

**Suggested Visual:** Data pipeline diagram showing sources flowing into the ACPS target, or a simple table of data sources with icons

**Speaker:** TODO H4

---

## Slide 4: EDA - When Congestion Happens

**Title:** Temporal Patterns in Airport Traffic

**Key Points:**
- Strong hourly pattern: distinct morning, midday, and evening peaks aligned with hub banks
- Day-of-week effects: weekday vs. weekend traffic profiles differ significantly
- Seasonal variation: summer peak, holiday spikes, January/February lows
- These patterns form the baseline that models must beat

**Suggested Visual:** Hourly traffic heatmap (hour-of-day x day-of-week) or overlaid daily traffic profiles by season

**Speaker:** TODO H4

---

## Slide 5: Geospatial View - Where Traffic Concentrates

**Title:** Spatial Density Around Madrid-Barajas

**Key Points:**
- Flight density mapped in concentric rings: 0-15km, 15-40km, 40-80km
- Approach and departure corridors show clear concentration patterns
- High-congestion periods show measurably denser traffic in the inner ring
- Spatial analysis connects time-series forecasting to physical airport geography

**Suggested Visual:** Side-by-side folium maps comparing low-congestion vs. high-congestion flight density (or a single map with a density heatmap)

**Speaker:** TODO H4

---

## Slide 6: Forecasting Approach

**Title:** From Baselines to Machine Learning

**Key Points:**
- Three naive baselines: previous hour, same hour yesterday, same hour last week
- Classical time-series: SARIMAX with weather exogenous variables
- Feature-based ML: HistGradientBoosting with 40+ engineered features (lags, rolling stats, weather, calendar)
- Strict chronological split: train (70%) / validation (15%) / test (15%) -- no data leakage

**Suggested Visual:** Methodology flowchart showing the progression from baselines to ML, or a feature category breakdown diagram

**Speaker:** TODO H4

---

## Slide 7: Model Comparison

**Title:** How Well Can We Forecast Congestion?

**Key Points:**
- TODO: Insert actual metrics after final model runs
- Compare all models on MAE, RMSE (regression) and F1, balanced accuracy (classification)
- Show results for both t+1 and t+3 forecast horizons
- Highlight the performance gain of ML over naive baselines

**Suggested Visual:** Bar chart comparing model MAE/RMSE across all approaches, or a summary comparison table with the best model highlighted

**Speaker:** TODO H4

---

## Slide 8: What Drives High Congestion

**Title:** Feature Importance and Key Drivers

**Key Points:**
- TODO: Insert actual top features after final training
- Expected: recent lag features dominate, but weather and calendar add value during disruptions
- Weather matters most during off-peak and transition periods
- Holiday effects and bridge days create predictable but unusual congestion spikes

**Suggested Visual:** Feature importance bar chart (top 10 features) or SHAP summary plot showing direction of effects

**Speaker:** TODO H4

---

## Slide 9: Limitations

**Title:** What We Cannot Claim

**Key Points:**
- ACPS is a proxy, not an official congestion metric -- it reflects movement volume, not delay
- Data coverage limited to the collection period; longer history would improve seasonal modeling
- FlightRadarAPI is unofficial -- live enrichment is demonstrative, not production-grade
- Single airport study: results may not generalize to airports with different operational profiles

**Suggested Visual:** Clean text slide with honest limitations listed. Optionally a "what we know vs. what we don't" two-column layout.

**Speaker:** TODO H4

---

## Slide 10: Recommendations and Next Steps

**Title:** Looking Ahead

**Key Points:**
- Short-horizon forecasting of congestion pressure is feasible with public data
- Practical use case: early-warning dashboard for operations planning teams
- Next steps: validate on a second airport (e.g., Barcelona), extend history, integrate official data
- The framework is reusable for any airport with OpenSky coverage

**Suggested Visual:** A forward-looking diagram showing potential extensions, or a mock dashboard screenshot if available

**Speaker:** TODO H4

---

## Presentation Logistics

### Speaker Allocation

TODO H4: Assign speakers so that every team member presents at least one section:

| Slide | Speaker | Backup Speaker |
|-------|---------|---------------|
| 1 | TODO H4 | TODO H4 |
| 2 | TODO H4 | TODO H4 |
| 3 | TODO H4 | TODO H4 |
| 4 | TODO H4 | TODO H4 |
| 5 | TODO H4 | TODO H4 |
| 6 | TODO H4 | TODO H4 |
| 7 | TODO H4 | TODO H4 |
| 8 | TODO H4 | TODO H4 |
| 9 | TODO H4 | TODO H4 |
| 10 | TODO H4 | TODO H4 |

### Design Guidelines

TODO H4:
- [ ] Use consistent slide template with project branding
- [ ] Maximum 4-5 bullet points per slide
- [ ] Every slide has one clear visual (chart, map, diagram, or infographic)
- [ ] Font size minimum 24pt for body text
- [ ] Include slide numbers and a subtle project title footer

### Rehearsal Plan

TODO H4:
- [ ] First run-through: individual slide drafts reviewed by H3
- [ ] Second run-through: full team dry run with timing
- [ ] Final rehearsal: practice Q&A with likely professor questions
- [ ] Prepare 2-minute backup explanation for each slide in case of questions

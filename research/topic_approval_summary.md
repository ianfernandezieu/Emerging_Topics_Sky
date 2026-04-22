# Topic Approval Summary

**Assigned to:** H1 - Domain Research and Topic Approval Lead
**Purpose:** One-page summary for instructor approval. Keep it concise and focused.
**Status:** TODO H1

---

## Project Title

TODO H1: Confirm final title with instructor. Current working title:

> Forecasting Airport Congestion at Madrid-Barajas Using Flight Tracking, Weather, and Calendar Signals

## Team Members

| # | Name | Role |
|---|------|------|
| 1 | TODO H1 | C1 - Data Engineering and API Ingestion Lead |
| 2 | TODO H1 | C2 - Feature Engineering and Forecasting Lead |
| 3 | TODO H1 | C3 - ML, Geospatial, and Evaluation Lead |
| 4 | TODO H1 | H1 - Domain Research and Topic Approval Lead |
| 5 | TODO H1 | H2 - Data Quality, Validation, and Ethics Lead |
| 6 | TODO H1 | H3 - Report Writing and Interpretation Lead |
| 7 | TODO H1 | H4 - Presentation, Design, and Q&A Lead |

## Research Question

TODO H1: Write a clear 2-3 sentence research question. Draft:

> Can we forecast airport congestion pressure at Madrid-Barajas 1 hour and 3 hours ahead using flight movement, weather, and calendar signals? Which factors are most predictive of high-congestion periods, and how much advance warning is practically achievable?

## Data Sources

TODO H1: Briefly describe each source and its role:

- **OpenSky Network** - Historical airport arrivals and departures (core training data)
- **Open-Meteo** - Historical hourly weather variables (temperature, wind, precipitation, pressure)
- **FlightRadarAPI** - Live/near-live flight snapshots for enrichment (educational wrapper)
- **OurAirports** - Airport metadata and coordinates
- **Nager.Date** - Spanish public holiday calendar

## Methodology Overview

TODO H1: Summarize the analytical approach in 3-5 sentences covering:

- Target definition: hourly Airport Congestion Pressure Score (ACPS) derived from flight movements
- Feature engineering: time-series lags, rolling statistics, weather variables, calendar indicators
- Modeling: naive baselines, SARIMAX, tree-based ML models (HistGradientBoosting)
- Evaluation: chronological train/validation/test split, MAE/RMSE for regression, F1/precision/recall for classification
- Geospatial: density mapping of flight traffic around the airport

## Expected Deliverables

TODO H1: Confirm with instructor which of these are required vs. optional:

- [ ] Reproducible code repository with execution instructions
- [ ] 11 Jupyter notebooks covering data collection through final visuals
- [ ] ~20-page written report with methodology, results, and limitations
- [ ] 10-15 minute group presentation with slides
- [ ] Contribution statement documenting each member's role

---

**Submission date for approval:** TODO H1
**Instructor feedback received:** TODO H1

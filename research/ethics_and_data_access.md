# Ethics and Data Access

**Assigned to:** H2 - Data Quality, Validation, and Ethics Lead
**Purpose:** Document the ethical and legal basis for all data used in this project.
**Status:** TODO H2

---

## Data Sources and Legality

TODO H2: For each data source, confirm its legal status and terms of use:

### OpenSky Network
- [ ] TODO H2: Review OpenSky Network data licensing terms (https://opensky-network.org/)
- [ ] TODO H2: Confirm that academic/research use is permitted
- [ ] TODO H2: Note any attribution or citation requirements
- [ ] TODO H2: Document the data access method used (REST API, Trino, data dumps)

### Open-Meteo
- [ ] TODO H2: Review Open-Meteo terms of service and licensing
- [ ] TODO H2: Confirm that the free tier permits academic project use
- [ ] TODO H2: Note attribution requirements (e.g., credit to data providers like DWD, NOAA, etc.)

### OurAirports
- [ ] TODO H2: Confirm open data license for OurAirports dataset
- [ ] TODO H2: Note any restrictions on redistribution

### Nager.Date
- [ ] TODO H2: Confirm public API terms for holiday data
- [ ] TODO H2: Note whether data is derived from official government sources

### Aena (if used)
- [ ] TODO H2: Clarify whether any Aena data was scraped vs. manually referenced
- [ ] TODO H2: If screenshots are included in the report, note they are for academic illustration only

## Rate Limiting Compliance

TODO H2: Document how the project respects rate limits for each API:

- [ ] TODO H2: OpenSky - Document the rate limits and how the code respects them (e.g., sleep between requests, caching)
- [ ] TODO H2: Open-Meteo - Document the rate limits for free-tier users and compliance measures
- [ ] TODO H2: FlightRadarAPI wrapper - Document how requests are throttled and cached
- [ ] TODO H2: Confirm that all fetcher modules implement retry logic with backoff, not aggressive polling

## Unofficial API Considerations

TODO H2: Address the use of the FlightRadarAPI unofficial wrapper specifically:

- [ ] TODO H2: Note that `FlightRadarAPI` (PyPI) is an unofficial educational wrapper, NOT the official Flightradar24 API
- [ ] TODO H2: Explain why it is used (educational demonstration of live data integration) and why it is NOT the core training data source
- [ ] TODO H2: Document the fallback strategy if this source becomes unavailable (project continues with OpenSky core pipeline)
- [ ] TODO H2: Confirm that the project does not violate Flightradar24's terms of service for end users
- [ ] TODO H2: Note any disclaimers that should appear in the report regarding this data source

## Privacy

TODO H2: Address privacy considerations:

- [ ] TODO H2: Confirm that no personally identifiable information (PII) is collected or stored
- [ ] TODO H2: Note that flight data is aggregated at the airport/hourly level, not tracking individual passengers
- [ ] TODO H2: Confirm that no airline-specific commercial data is exposed in ways that could harm competitive interests
- [ ] TODO H2: If any aircraft registration numbers or callsigns appear in raw data, confirm they are not published in outputs
- [ ] TODO H2: Note GDPR considerations if any (likely minimal given aggregated public data)

## Academic Use Justification

TODO H2: Write a brief statement (3-5 sentences) justifying the data use for inclusion in the report:

- [ ] TODO H2: State that all data sources are publicly available or openly licensed
- [ ] TODO H2: State that data is used solely for academic coursework and not for commercial purposes
- [ ] TODO H2: State that the project complies with each source's terms of use
- [ ] TODO H2: State that rate limiting and caching are implemented to avoid unnecessary load on public services
- [ ] TODO H2: Note that AI tools (Claude Code) were used in development, reviewed by humans, and disclosed in the contribution statement

---

## Sign-off

| Item | Confirmed by | Date |
|------|-------------|------|
| All data sources reviewed for legal compliance | TODO H2 | TODO |
| Rate limiting measures verified in code | TODO H2 | TODO |
| Privacy assessment completed | TODO H2 | TODO |
| Academic use justification drafted | TODO H2 | TODO |

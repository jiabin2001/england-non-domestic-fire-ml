# Day 1 feasibility audit

> Historical feasibility audit. The counts, source details and original proceed decision below are retained without re-importing data. Subsequent documentation corrections concerning field timing and annual validation are recorded in [no-rerun revision note](no_rerun_revision.md); the original availability checks do not establish that fields were known at prediction time.

## Decision

**Feasible — proceed to cohort construction and modelling.**

The current official **Other Building Fires Dataset** is an incident-level dataset of primary fires attended by Fire and Rescue Services in other buildings in England. The project's term *non-domestic building fires* refers to this official category. It includes commercial, industrial, public and institutional buildings, and can include hotels, hostels, care homes and student halls; it must not be interpreted as containing only buildings with no accommodation function.

## Source and import

- Source page: https://www.gov.uk/government/statistics/fire-statistics-incident-level-datasets
- Guidance: https://www.gov.uk/government/statistics/fire-statistics-incident-level-datasets/other-building-fires-dataset-guidance
- ODS URL: https://assets.publishing.service.gov.uk/media/6a5e7a52c7c34404041b4663/Other_building_fires_dataset.ods
- Official page update: 2026-07-22
- Raw file: `Other_building_fires_dataset.ods` (45,066,461 bytes)
- SHA-256: `560e5c1a18731cf8189b28471f6813675d6f95f447d680ec5603d9bb97718595`
- ODS sheets: Cover_sheet, Datasheet, config
- Incident sheet: `Datasheet`
- Logical shape after resolving documented export artifacts: 245,207 rows × 40 fields
- Exact duplicate logical records: 23

The ODS parser initially exposed two unnamed trailing columns and 245,205 non-record rows without `FINANCIAL_YEAR`. Inspection showed these cells repeat the two final outcome columns below the incident table. The reproducible import rule retains rows with a financial year and drops only unnamed trailing columns. The original ODS remains unchanged.

## Target audit

All seven observed `SPREAD_OF_FIRE` strings are listed verbatim in `outputs/tables/spread_of_fire_categories.csv`. Six map unambiguously to the study's ordered room→floor→whole-building spread estimand. `Roofs/ Roof spaces` does not locate an incident uniquely on that ordering, so it is excluded from the main outcome. Official sources differ: the current Fire statistics definitions omit roofs from the larger-fire list, while FIRE0304-linked detailed releases include them. The roof-positive sensitivity tests that alternative published convention. There are no missing target strings in logical incident rows.

For 2024/25, the main study estimand (excluding roofs) gives 23.9% among mappable records. Mapping `Roofs/ Roof spaces` as larger and using all 2024/25 records gives 26.0%, reproducing the approximately 26% proportion in the detailed FIRE0304-linked release. Reporting both definitions makes the official-source inconsistency explicit.

## Time and data quality

- Available years: 2010/11, 2011/12, 2012/13, 2013/14, 2014/15, 2015/16, 2016/17, 2017/18, 2018/19, 2019/20, 2020/21, 2021/22, 2022/23, 2023/24, 2024/25, 2025/26.
- All prespecified main years 2010/11–2023/24 are present.
- The two latest complete main-window years become temporal test (2022/23–2023/24); the preceding two become validation (2020/21–2021/22); earlier years form training.
- The dataset has no incident date or month. Calendar timing is limited to financial year, day of week and four day-part bands. It also contains banded process durations (`IGNITION_TO_DISCOVERY`, `DISCOVERY_TO_CALL`, `RESPONSE_TIME`, `TIME_AT_SCENE`); the latter are not calendar timestamps.
- Suffolk FRS is incomplete from September 2024 to March 2025 according to current official guidance, so 2024/25 is excluded from the main analysis.
- 2025/26 spans the IRS-to-FaRDaP collection transition beginning November 2025 and is excluded from the main analysis.
- `RAPID_FIRE_GROWTH` is absent in 92.4% of records and is excluded from all main models as prespecified.

## Feasibility checks

- `SPREAD_OF_FIRE` found: yes.
- Incident-level structure supported: yes (one row per disclosed incident record; no public incident ID).
- Target categories reliably mappable: yes, with roofs explicitly excluded in the main definition.
- Financial-year design usable: yes.
- Key Block A/B/C fields usable in 2010/11–2023/24: yes; maximum field missingness is 0.0%.
- Guidance consistency: the original audit recorded no blocking inconsistency. Later review identified an interpretation limitation: `OCCUPIED_TIME` may count people in buildings reached by spread, and Block C inherits retrospective investigation fields. These issues require the revised interpretation; their effects on performance have not been tested.

## Main analysis window

Use 2010/11–2023/24. Exclude exact duplicate rows, late calls and target-excluded rows. Do not use `FINANCIAL_YEAR` or FRS territory as predictors. Use validation data for compact model selection and evaluate the resulting train-fitted model on the temporal test.

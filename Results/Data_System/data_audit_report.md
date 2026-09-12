# Data System Audit Report

- Checked at: `2026-09-12T14:22:05+02:00`
- Overall status: **WARNING**
- Bundle fingerprint: `4f2db574266725ab28d1d913ca216b421793b5613f1e9c8d2f098b935aa12e46`
- Checks: PASS 54 / WARNING 4 / FAIL 0
- Audit mode: read-only; no data were downloaded, deleted or corrected.

## Checks

| # | Section | Status | Check | Result |
| ---: | --- | --- | --- | --- |
| 1 | Files | **PASS** | prices | File opens successfully |
| 2 | Files | **PASS** | returns | File opens successfully |
| 3 | Files | **PASS** | volume | File opens successfully |
| 4 | Files | **PASS** | volume_quality | File opens successfully |
| 5 | Files | **PASS** | liquidity | File opens successfully |
| 6 | Files | **PASS** | prices_long | File opens successfully |
| 7 | Files | **PASS** | availability | File opens successfully |
| 8 | Files | **PASS** | forward_returns | File opens successfully |
| 9 | Files | **PASS** | membership | File opens successfully |
| 10 | Files | **PASS** | quality | File opens successfully |
| 11 | Files | **PASS** | universe | File opens successfully |
| 12 | Files | **PASS** | historical_components | File opens successfully |
| 13 | Files | **PASS** | risk_free_rate | File opens successfully |
| 14 | Files | **PASS** | Equity file generation times | Modification-time spread: 0.10 hours |
| 15 | Matrices | **PASS** | Price date index | Rows: 4,686; dates: 2008-01-02 00:00:00 -> 2026-08-18 00:00:00 |
| 16 | Matrices | **PASS** | Unique price tickers | Tickers: 900 |
| 17 | Matrices | **PASS** | Valid observed prices | Non-positive: 0; infinite: 0 |
| 18 | Matrices | **PASS** | returns alignment | Shape: (4686, 900); expected: (4686, 900) |
| 19 | Matrices | **PASS** | volume alignment | Shape: (4686, 900); expected: (4686, 900) |
| 20 | Matrices | **PASS** | volume_quality alignment | Shape: (4686, 900); expected: (4686, 900) |
| 21 | Matrices | **PASS** | liquidity alignment | Shape: (4686, 900); expected: (4686, 900) |
| 22 | Matrices | **PASS** | availability alignment | Shape: (4686, 900); expected: (4686, 900) |
| 23 | Matrices | **PASS** | forward_returns alignment | Shape: (4686, 900); expected: (4686, 900) |
| 24 | Matrices | **PASS** | membership alignment | Shape: (4686, 900); expected: (4686, 900) |
| 25 | Matrices | **PASS** | quality alignment | Shape: (4686, 900); expected: (4686, 900) |
| 26 | Matrices | **PASS** | availability boolean dtype | All columns are boolean |
| 27 | Matrices | **PASS** | membership boolean dtype | All columns are boolean |
| 28 | Matrices | **PASS** | quality boolean dtype | All columns are boolean |
| 29 | Matrices | **PASS** | volume_quality boolean dtype | All columns are boolean |
| 30 | Quality | **PASS** | Quality requires a positive observed price | Invalid True cells: 0 |
| 31 | Calculations | **PASS** | Daily returns reproduce from prices and quality | Mismatched cells: 0 |
| 32 | Quality | **PASS** | Unclipped large returns | Absolute returns >=50%: 92; unconfirmed returns >=100%: 0 |
| 33 | Calculations | **PASS** | 21-day forward returns reproduce from prices and quality | Mismatched cells: 0 |
| 34 | Calculations | **PASS** | Availability formula | Mismatched cells: 0 |
| 35 | Calculations | **PASS** | Liquidity formula | Mismatched cells: 0 |
| 36 | Calculations | **PASS** | Long prices rows and uniqueness | Rows: 2,827,310; expected: 2,827,310; duplicate date-ticker pairs: 0 |
| 37 | Calculations | **PASS** | Long prices reproduce wide prices | Same keys: True; mismatched prices: 0 |
| 38 | Universe | **PASS** | Historical component snapshots | Snapshots: 2,720; component range: 496 -> 507 |
| 39 | Universe | **PASS** | Membership reproduces from historical snapshots | Mismatched cells: 0 |
| 40 | Universe | **PASS** | Price data does not exceed membership source | Last price: 2026-08-18 00:00:00; last snapshot: 2026-08-18 00:00:00 |
| 41 | Universe | **PASS** | Universe report schema | Missing columns: none |
| 42 | Universe | **PASS** | Universe report matches price columns | Only in universe: 0; only in prices: 0 |
| 43 | Universe | **PASS** | Universe contains the historical ticker union | Source union: 900; report: 900; difference: 0 |
| 44 | Universe | **PASS** | Valid membership coverage values | Invalid rows: 0 |
| 45 | Universe | **WARNING** | Historical data availability | No prices during membership: 247; coverage below 80%: 262; missing downloads: 191; rejected reused symbols: 20 |
| 46 | Universe | **PASS** | Universe membership counts | Mismatched tickers: 0 |
| 47 | Prices | **WARNING** | Missing-price runs longer than 5 trading dates during membership | Found 277 runs (429,023 observations) across 263 tickers. Longest runs: AVB: 4663 trading dates (2008-01-02 -> 2026-07-16); EA: 4663 trading dates (2008-01-02 -> 2026-07-16); EQR: 4663 trading dates (2008-01-02 -> 2026-07-16); BK: 4625 trading dates (2008-01-02 -> 2026-05-20); MMC: 4537 trading dates (2008-01-02 -> 2026-01-13); K: 4515 trading dates (2008-01-02 -> 2025-12-10); IPG: 4506 trading dates (2008-01-02 -> 2025-11-26); WBA: 4442 trading dates (2008-01-02 -> 2025-08-27); HES: 4416 trading dates (2008-01-02 -> 2025-07-22); JNPR: 4406 trading dates (2008-01-02 -> 2025-07-08). The audit reports these gaps but does not fill or remove them. |
| 48 | Volume | **PASS** | Valid volume values | Negative: 0; infinite: 0 |
| 49 | Volume | **PASS** | Volume quality formula | Mismatched cells: 0 |
| 50 | Volume | **WARNING** | Missing and zero volume during membership | Relevant observations: 1,921,650; missing: 0 (0.0000%); zero: 33 (0.0017%) |
| 51 | Volume | **PASS** | Raw invalid volume excluded by volume_quality | Invalid raw observations: 33; excluded: 33; still accepted: 0 |
| 52 | Volume | **WARNING** | Positive volume jumps remaining after volume_quality >= 100x | Remaining positive-to-positive events: 36 across 27 tickers; most affected: GENZ: 5, BIIB: 4, BF-B: 2, HUBB: 2, AMGN: 1, AMTM: 1, BRK-B: 1, CINF: 1, COST: 1, BEN: 1. Not removed automatically because both observations are valid positive volumes. |
| 53 | Volume | **PASS** | Runs of at least 5 missing/zero volume observations | Found 0 runs (0 observations) across 0 tickers: None. |
| 54 | Volume | **PASS** | Long missing/zero runs excluded by volume_quality | Observations inside long runs: 0; excluded: 0; still accepted: 0 |
| 55 | Volume | **PASS** | Invalid volume excluded from liquidity inputs | Excluded from volume/liquidity analysis while retained for price analysis: 33 observations |
| 56 | Risk-free rate | **PASS** | DGS3MO schema | Columns: ['annual_rate_pct'] |
| 57 | Risk-free rate | **PASS** | Valid DGS3MO observations | Observed: 4,660; missing: 200; negative: 0; infinite: 0 |
| 58 | Risk-free rate | **PASS** | DGS3MO covers the equity period | Rates: 2008-01-02 00:00:00 -> 2026-08-18 00:00:00; equities: 2008-01-02 00:00:00 -> 2026-08-18 00:00:00 |

## File inventory

| Dataset | File | State | Size | Modified |
| --- | --- | --- | --- | --- |
| prices | `C:\Users\God\PycharmProjects\Cross-Sectional Alpha Modeling and Portfolio Construction\Data\Data_System\Raw\prices.parquet` | FOUND | 18.89 MB | 2026-09-11T16:23:51+02:00 |
| returns | `C:\Users\God\PycharmProjects\Cross-Sectional Alpha Modeling and Portfolio Construction\Data\Data_System\Processed\returns.parquet` | FOUND | 26.06 MB | 2026-09-11T16:23:52+02:00 |
| volume | `C:\Users\God\PycharmProjects\Cross-Sectional Alpha Modeling and Portfolio Construction\Data\Data_System\Raw\volume.parquet` | FOUND | 24.61 MB | 2026-09-11T16:23:54+02:00 |
| volume_quality | `C:\Users\God\PycharmProjects\Cross-Sectional Alpha Modeling and Portfolio Construction\Data\Data_System\Processed\volume_quality.parquet` | FOUND | 0.43 MB | 2026-09-11T16:23:54+02:00 |
| liquidity | `C:\Users\God\PycharmProjects\Cross-Sectional Alpha Modeling and Portfolio Construction\Data\Data_System\Raw\liquidity.parquet` | FOUND | 26.14 MB | 2026-09-11T16:23:55+02:00 |
| prices_long | `C:\Users\God\PycharmProjects\Cross-Sectional Alpha Modeling and Portfolio Construction\Data\Data_System\Processed\prices_long.parquet` | FOUND | 17.20 MB | 2026-09-11T16:23:56+02:00 |
| availability | `C:\Users\God\PycharmProjects\Cross-Sectional Alpha Modeling and Portfolio Construction\Data\Data_System\Processed\availability.parquet` | FOUND | 0.42 MB | 2026-09-11T16:23:56+02:00 |
| forward_returns | `C:\Users\God\PycharmProjects\Cross-Sectional Alpha Modeling and Portfolio Construction\Data\Data_System\Processed\forward_returns.parquet` | FOUND | 26.12 MB | 2026-09-11T16:23:53+02:00 |
| membership | `C:\Users\God\PycharmProjects\Cross-Sectional Alpha Modeling and Portfolio Construction\Data\Data_System\Processed\membership.parquet` | FOUND | 0.43 MB | 2026-09-11T16:23:57+02:00 |
| quality | `C:\Users\God\PycharmProjects\Cross-Sectional Alpha Modeling and Portfolio Construction\Data\Data_System\Processed\data_quality.parquet` | FOUND | 0.42 MB | 2026-09-11T16:23:57+02:00 |
| universe | `C:\Users\God\PycharmProjects\Cross-Sectional Alpha Modeling and Portfolio Construction\Data\Data_System\Raw\universe.csv` | FOUND | 0.06 MB | 2026-09-11T16:23:57+02:00 |
| historical_components | `C:\Users\God\PycharmProjects\Cross-Sectional Alpha Modeling and Portfolio Construction\Data\Data_System\Raw\sp500_historical_components.csv` | FOUND | 5.27 MB | 2026-09-11T16:18:01+02:00 |
| risk_free_rate | `C:\Users\God\PycharmProjects\Cross-Sectional Alpha Modeling and Portfolio Construction\Data\Data_System\Raw\dgs3mo.parquet` | FOUND | 0.05 MB | 2026-09-11T16:24:01+02:00 |

## Interpretation

- **FAIL** means that a file or a mathematical relationship is incorrect.
- **WARNING** means that the dataset remains usable only with a documented limitation.
- **PASS** means that the specific internal check succeeded.

## Limits of this audit

- Internal consistency does not prove that Yahoo or the historical membership source is correct.
- Missing delisted securities and survivorship/data-availability bias cannot be repaired by this audit.
- Modification times help detect mixed builds but do not make multi-file saving atomic.
- Large price and volume moves are diagnostics; economic reality may require an external source.

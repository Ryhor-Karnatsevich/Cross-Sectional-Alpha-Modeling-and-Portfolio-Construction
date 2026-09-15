# Cross-Sectional Equity Factor Research

## Work in Progress

This project is currently under active development. The Data System, Factor Layer and Factor Selection Layer are completed. The Research Layer is the current stage.


## Research Objective

The purpose of this project is to study whether information contained in historical stock prices can identify differences in future returns across S&P500 companies.

The project does not attempt to forecast the direction of the complete stock market. It performs cross-sectional research: on every observation date, stocks are compared with other stocks available in the index universe at that time.

The research begins with interpretable price-based factors. Different factor definitions, parameter settings and return horizons are tested across multiple historical periods. Their statistical and economic results are then compared to determine whether any signal is sufficiently stable for further research.

The current objective is factor discovery and evaluation. A production trading strategy or final portfolio construction model is not assumed before the factors demonstrate useful out-of-sample behaviour.


## Project Roadmap

| stage                  | status            |
|------------------------|-------------------|
| Data System            | **completed**     |
| Factor Layer           | **completed**     |
| Factor Selection Layer | **completed**     |
| Research Layer         | <- here right now |


## Project Structure
src/


  - Data_System/
    - __init__.py
    - config.py
    - risk_free_rate.py
    - get_tickers.py
    - data_quality.py
    - equity_data.py
    - data_audit.py
    - **pipeline.py**
    - delete.py


  - Factors_Layer/
    - factor_config.py
    - factors.py
    - transforms.py
    - factor_builder.py
    - forward_returns.py
    - factor_storage.py
    - **pipeline.py**
    - delete.py


  - Factor_Selection_Layer/
    - selection_config.py
    - selection_storage.py
    - quantile_analysis.py
    - hypothesis_analysis.py
    - pattern_classification.py
    - classifier_validation.py
    - selection_report.py
    - **pipeline.py**


  - Research_Layer/
    - __init__.py
    - research_config.py
    - research_storage.py
    - research_data.py
    - multiple_testing_research.py
    - factor_overlap.py
    - portfolio_construction.py
    - risk_exposure.py
    - portfolio_engine.py
    - portfolio_evaluation.py
    - walk_forward_research.py
    - regime_research_current.py
    - research_report.py
    - **pipeline.py**
    - delete.py


  - Pipeline
    - run.py


Data/


  - Data_System/
    - Raw/
    - Processed/
    - Cache/


  - Factors_Layer/
    - Cache/
      - Factor_Matrices/
      - Forward_Return_Matrices/


  - Factor_Selection_Layer/
    - Cache/
      - daily_quantile_results.parquet
      - time_stability.parquet


  - Research_Layer/
    - Cache/


Results/


  - Data_System/
    - Figures/
    - data_audit_report.md


  - Factors_Layer/
    - factor_run_metadata.json


  - Factor_Selection_Layer/
    - Figures/
    - quantile_run_metadata.json
    - selection_run_metadata.json
    - hypothesis_cards.csv
    - effect_tests.csv
    - quantile_curves.csv
    - classifier_validation.csv
    - factor_selection_report.md


  - Research_Layer/
    - Figures/
    - multiple_testing_comparison.csv
    - multiple_testing_summary.csv
    - factor_overlap.csv
    - portfolio_metadata.csv
    - portfolio_statistics.csv
    - calendar_phase_stability.csv
    - walk_forward_periods.csv
    - walk_forward_summary.csv
    - risk_exposure_summary.csv
    - sector_exposure_status.csv
    - regime_performance.csv
    - research_run_metadata.json
    - research_report.md


`Data` contains large datasets, calculated matrices and disposable caches. It is excluded from Git because every current file can be downloaded or calculated again.

`Results` contains compact tables, run metadata, audit reports and figures intended for direct reading and Git history.


## Data System [1]

The purpose of this layer is to create the complete data foundation used by the Factor Layer. It downloads the historical S&P500 ticker universe, obtains available Yahoo Finance market data, reconstructs point-in-time index membership and prepares aligned research matrices.

Equity market data are downloaded from 2008. This provides historical observations that can later be used as factor warm-up data. The Data System itself does not select factor-research periods or divide observations into research and validation windows.

The completed layer creates 12 core equity datasets with different metrics and formats:
- Seven **"Processed"** files.
- Five **"Raw"** files.
- One additional macro dataset containing the US three-month Treasury rate.

The final output contains adjusted prices, returns, compatible volume, liquidity, historical index membership, price and volume quality, price availability, long-format prices and 21-day forward returns. All wide matrices use aligned trading dates and historical ticker columns.

Every completed Data System bundle is checked by a separate read-only audit. The audit verifies file readability, matrix alignment, calculated relationships, historical-universe construction, price gaps, volume quality and risk-free-rate coverage. Its results are written to a separate Markdown report and do not automatically change the datasets.


**Limitations**:
- Historical membership is taken from a community-maintained GitHub repository, not from official S&P data.
- yfinance may have missing or incomplete history for delisted stocks and old ticker symbols. Therefore survivorship bias is reduced but not fully removed.
- All Factor Layer results must be interpreted with this remaining survivorship/data-availability bias in mind.
- The dataset ends on the latest repository snapshot instead of assuming an unknown index composition after that date.


IMPORTANT:
- All future features must be computed using data up to t-1
- daily returns represent t-1 -> t
- saved forward returns represent t -> t+21 trading days



### **config.py**:
- Defines separate paths for heavy Data System files in `Data/Data_System` and readable outputs in `Results/Data_System`.
- Stores raw datasets in `Data/Data_System/Raw`, processed matrices in `Data/Data_System/Processed` and the yfinance cache in `Data/Data_System/Cache`.
- Stores the audit report in `Results/Data_System` and audit figures in `Results/Data_System/Figures`.
- Defines the historical S&P500 components URL and the FRED DGS3MO URL.
- `DATA_START_DATE = 2008-01-01`.
- Price-quality parameters:
  - `SUSPICIOUS_ABS_DAILY_RETURN = 0.5`.
  - `MAX_ABS_DAILY_RETURN = 1.0`.
  - `ROUND_TRIP_RETURN_TOLERANCE = 0.25`.
- Audit parameters:
  - `AUDIT_NUMERIC_TOLERANCE = 1e-10`.
  - `AUDIT_FILE_TIME_SPREAD_HOURS = 6`.
  - `AUDIT_VOLUME_JUMP_RATIO = 100`.
- `CONFIRMED_REAL_RETURN_EVENTS` contains manually verified extreme market moves. Currently: `HIG` on `2008-12-05`, `GME` on `2021-01-27` and `NKTR` on `2025-06-24`.
- `YAHOO_REUSED_TICKERS` contains obsolete symbols whose Yahoo history belongs to another security. These symbols are rejected when no verified continuous alias exists.
- `YAHOO_TICKER_ALIASES` contains only verified direct company/ticker changes. Acquisitions, unrelated mergers and post-bankruptcy securities are not treated as continuous aliases.



### **risk_free_rate.py**
- Prepares the daily three-month US Treasury yield used as the project risk-free-rate dataset.
- Uses the official FRED `DGS3MO` series and stores annual yield in percent.

#### download_dgs3mo
- Downloads DGS3MO observations for the requested start and end dates without an API key.
- Requires exactly two columns from FRED and renames them to `date` and `annual_rate_pct`.
- Parses dates and converts rate observations to numeric values.
- Keeps unavailable rate observations as missing.
- Sorts observations and limits them to the requested period.
- Requires at least 100 valid rate observations.
- Rejects duplicated dates and negative rates.
- Returns the prepared rate DataFrame without saving it.

#### prepare_risk_free_rate
- Used by the main Data System pipeline.
- Keeps the existing `Data/Data_System/Raw/dgs3mo.parquet` without downloading it again.
- Calls `download_dgs3mo` only when the local file does not exist.
- Creates the destination directory and saves the downloaded rate dataset.
- Does not inspect or extend an existing file; its quality and coverage are checked later by `data_audit.py`.

#### ensure_risk_free_rate
- Used when another research module directly requires a rate dataset for a specific period.
- Loads the existing file when forced downloading is disabled.
- Checks whether valid local observations cover the requested start and end dates.
- Returns the requested slice when local coverage is sufficient.
- Downloads and overwrites the rate dataset when the file is missing, coverage is insufficient or `force_download=True`.



### **get_tickers.py**:
- Prepares the historical S&P500 universe and point-in-time membership.
- Uses the community-maintained `fja05680/sp500` GitHub repository.

#### normalize_ticker
- Removes surrounding spaces and converts the ticker to uppercase.
- Replaces dots with dashes for yfinance compatibility.

#### get_sp500_history
- Downloads `S&P 500 Historical Components & Changes (Updated).csv` when `refresh=True`.
- Saves the downloaded source file in `Data/Data_System/Raw`.
- Uses the local copy when the online request fails.
- Raises the original download error when both the online source and local copy are unavailable.
- Requires `date` and `tickers` columns.
- Keeps only these two required columns.
- Parses and sorts snapshot dates.
- Keeps the last snapshot when the source contains the same date more than once.
- Normalizes every ticker inside every snapshot.
- Requires historical snapshots covering `DATA_START_DATE` or later.
- Requires between 450 and 550 components in every relevant snapshot.
- Rejects duplicated tickers inside one snapshot.
- Returns the prepared historical snapshots DataFrame.

#### get_sp500_tickers
- Loads historical snapshots when they were not provided by the caller.
- Starts with the latest snapshot available on or before the requested start date.
- Raises an error when the requested start date is earlier than the available history.
- Combines tickers from every relevant snapshot into one historical ticker union.
- Removes duplicated ticker symbols and returns the sorted list.

#### get_sp500_tickers_by_date
- Loads historical snapshots when they were not provided by the caller.
- Finds the latest index snapshot available on or before the requested date.
- Returns the S&P500 component list from that snapshot.
- Raises an error when the requested date is earlier than the available history.

#### build_membership_matrix
- Receives historical snapshots, equity trading dates and the complete ticker list.
- Creates one boolean snapshot row for every historical component date.
- Reindexes snapshots to the equity trading calendar.
- Carries each snapshot forward until the next index change.
- Uses `True` when a ticker belongs to the index and `False` when it does not.
- Raises an error when membership history does not cover the first equity trading date.
- Warns when trading dates extend beyond the latest available historical snapshot.
- Returns the daily boolean membership matrix.



### **data_quality.py**
- Contains price- and volume-quality functions used by `equity_data.py`.
- Does not download, save or change datasets by itself.

#### build_volume_quality_mask
- Receives the compatible-volume matrix.
- Creates a boolean matrix with the same dates and tickers.
- Returns `True` only when volume exists, is finite and is greater than zero.
- Returns `False` for missing, zero, negative and infinite volume.
- This mask is applied before liquidity is calculated.

#### build_confirmed_event_mask
- Receives prices and the manually verified extreme market events from `config.py`.
- Creates a boolean matrix filled with `False`.
- Marks `True` only on the ticker-date pairs listed in `CONFIRMED_REAL_RETURN_EVENTS`.
- Allows known real extreme moves to remain usable instead of being treated as data errors.

#### build_data_quality_mask
- Receives the adjusted-price matrix and price-quality thresholds from `config.py`.
- Requires observed prices to be positive.
- Calculates raw daily returns without filling missing prices.
- Flags returns with an absolute move of at least 50% as suspicious diagnostics.
- Flags returns with an absolute move of at least 100% as extreme.
- Detects possible one-day spike reversals: a large move followed by a large opposite move that approximately returns to the original price.
- Removes manually confirmed real extreme events from the anomaly list.
- Creates an anomaly trigger from unconfirmed extreme returns or spike reversals.
- Quarantines a ticker from its first anomaly trigger onward.
- Returns four boolean matrices: price quality, suspicious returns, anomaly triggers and quarantine state.

#### first_true_date
- Receives a boolean matrix such as the quarantine matrix.
- Finds the first date with `True` for every ticker.
- Returns `NaT` when a ticker never has a `True` observation.
- Used by `equity_data.py` to store `quarantined_from` in `universe.csv`.



### **equity_data.py**:
- Downloads, calculates, aligns and saves the complete equity dataset for further factor analysis.

#### download_yahoo_request
- Sends one yfinance request for the provided tickers and start date.
- Uses `auto_adjust=False` to receive both raw close and adjusted close.
- Returns an empty DataFrame when Yahoo returns nothing or the request fails.

#### Download helpers
- `ticker_has_prices` checks whether a ticker has at least one adjusted close observation.
- `select_ticker_data` selects one Yahoo ticker and renames it to the required historical ticker.
- `merge_downloads` combines additional observations without overwriting existing non-missing data.
- `replace_ticker_data` removes the existing ticker series before inserting its verified replacement.

#### download_data
- Downloads the historical ticker union from `DATA_START_DATE` in batches of 50.
- Uses the local yfinance timezone and cookie cache.
- Merges batches and removes duplicated columns.
- Retries individually only tickers with no adjusted price observations after batch downloading.
- Uses explicit aliases only for verified direct ticker changes.
- Replaces the old ticker series with the verified alias series.
- Rejects every ticker from `YAHOO_REUSED_TICKERS` when no verified continuous alias exists.
- Keeps unavailable and rejected historical tickers in the universe but without false price data.
- Records `batch`, `individual_retry`, `alias`, `missing` or `reused_symbol_rejected` for every ticker.
- Raises an error only when Yahoo returns no usable price data for the entire ticker universe.

#### get_price_matrix
- Extracts adjusted close prices.
- Sorts observations by date.
- Removes dates where every ticker is missing.
- Keeps individual missing prices as missing.

#### get_volume_matrix
- Extracts raw Yahoo volume.
- Extracts raw close and adjusted close.
- Rescales volume by `raw close / adjusted close`.
- Keeps `adjusted price x stored volume` equal to `raw close x raw Yahoo volume`.
- Replaces negative volume with missing values.
- Does not replace zero volume at that step.

#### compute_returns
- Computes daily return from `t-1` to `t` using adjusted prices.
- Does not clip large real market moves.
- Requires both the current and previous price to pass data quality.
- Replaces a return with missing when either endpoint failed data quality.

#### compute_liquidity
- Receives volume after the `volume_quality` mask has been applied.
- Computes daily dollar volume as `adjusted price x compatible volume`.
- Calculates the 20-day rolling mean of dollar volume.
- Requires 20 valid observations in the rolling window.
- Applies the `log(1 + x)` transform.

#### to_long
- Converts the wide adjusted-price matrix into `date`, `ticker`, `price` rows.
- Stores only observed prices.

#### compute_forward_returns
- Uses 21 trading days as the default horizon.
- Calculates `price[t+21] / price[t] - 1` without a forecasting model.
- Requires both the starting and ending price to pass data quality.
- Keeps unavailable forward returns as missing.

#### compute_availability
- Creates the final price availability matrix.
- An observation is available only when price exists, the ticker is an index member and price quality is valid.
- Does not include volume quality in price availability.

#### sanity_checks
- Checks that the price index is sorted.
- Checks that the dataset contains more than 100 ticker columns.
- Checks price and volume index and column alignment.
- Raises an error for negative volume.
- Raises an error for duplicated dates.
- Prints the average missing-value ratio for volume.

#### filter_universe
- Counts available S&P500 members for every date.
- Removes dates with fewer than 150 available members.
- Initially filters prices, liquidity, membership and price quality.
- Returns, forward returns, volume and volume quality are aligned to the retained dates afterwards.
- Does not remove individual tickers using full-period coverage.

#### save_all
- Saves adjusted prices, compatible volume and liquidity in `Data/Data_System/Raw`.
- Saves returns, forward returns, long prices, membership, data quality, volume quality and availability in `Data/Data_System/Processed`.
- Saves the universe report as `Data/Data_System/Raw/universe.csv`.
- Writes every dataset directly to its configured parquet or csv path.

#### load_saved_equity_data
- Loads prices, returns, volume, liquidity, long prices, availability and forward returns.
- Returns these seven datasets to `pipeline.py` in a fixed order.

#### build_and_save_dataset
- Runs the complete equity data-building sequence.
- Downloads data and creates the download report.
- Creates price and compatible-volume matrices with every historical ticker column.
- Limits prices and volume to the latest membership snapshot date.
- Builds point-in-time membership using `get_tickers.py`.
- Aligns volume to prices and keeps volume only where price exists.
- Calls `data_quality.py` to identify suspicious returns, anomaly triggers and ticker quarantine dates.
- Uses full-period coverage and anomaly statistics only for the universe diagnostic report.
- Creates returns and 21-day forward returns using price quality.
- Creates `volume_quality`, excludes invalid volume from liquidity input and computes liquidity.
- Removes dates with fewer than 150 available index members.
- Aligns all output matrices to the retained dates.
- Creates availability and long-format prices.
- Runs sanity checks.
- Saves all equity datasets and the universe report.
- Returns prices, returns, volume, liquidity, long prices, availability and forward returns.



### **data_audit.py**
- Runs a read-only audit of all Data System datasets.
- Does not download, delete or correct market data.

#### add_check
- Adds one `PASS`, `WARNING` or `FAIL` result to the common checks list.
- Stores the section, check name, status and result details.

#### normalize_ticker
- Converts a ticker to text, removes surrounding spaces and converts it to uppercase.
- Replaces dots with dashes to match the Data System ticker format.

#### normalize_history
- Keeps only `date` and `tickers` from the historical components file.
- Parses and sorts dates and keeps the last snapshot when a date is duplicated.
- Applies ticker normalization to every historical snapshot.

#### frame_is_aligned
- Checks whether another matrix has exactly the same dates and ticker columns as the reference matrix.

#### numeric_mismatch_count
- Compares two numeric matrices using `AUDIT_NUMERIC_TOLERANCE`.
- Treats missing values in the same positions as equal.
- Returns the number of cells that do not match.

#### build_bundle_fingerprint
- Reads the file name, size and modification time of every expected dataset.
- Includes missing-file states.
- Creates one SHA-256 fingerprint for the complete Data System bundle.

#### load_files
- Checks whether every required parquet and csv file exists.
- Records file path, state, size and modification time for the report inventory.
- Opens every available file and reports unreadable or missing files as `FAIL`.
- Returns all successfully loaded datasets to the remaining audit checks.

#### check_file_generation_times
- Compares modification times of all equity files except the risk-free-rate file.
- Returns `WARNING` when some equity files are missing.
- Returns `WARNING` when the modification-time spread exceeds `AUDIT_FILE_TIME_SPREAD_HOURS`.

#### check_wide_matrices
- Checks that the price index contains unique sorted dates.
- Checks that price ticker columns are unique.
- Rejects non-positive and infinite observed prices.
- Checks that returns, volume, volume quality, liquidity, availability, forward returns, membership and price quality are aligned with prices.

#### check_boolean_matrices
- Checks that availability, membership, price quality and volume quality contain boolean columns.
- Checks that price quality cannot be `True` where a positive observed price does not exist.

#### check_calculated_matrices
- Recalculates daily returns from prices and price quality and compares every cell with the saved matrix.
- Confirms that large returns were not clipped and no unconfirmed return of at least 100% remains usable.
- Recalculates 21-day forward returns and compares them with the saved matrix.
- Recalculates availability as `price exists AND membership AND price quality`.
- Recalculates liquidity from prices and volume after applying volume quality.

#### longest_true_run
- Calculates the longest consecutive `True` sequence in one boolean series.
- Currently remains as a helper but is not called by the audit workflow.

#### find_true_runs
- Finds every consecutive `True` sequence with at least the requested length.
- Returns the start date, end date and length of every sequence.
- Used to find long missing-price and invalid-volume runs.

#### check_price_gaps
- Finds consecutive missing-price observations during actual index membership.
- Reports `WARNING` when a price gap is longer than 5 trading dates.
- Reports the number of runs, affected observations and affected tickers.
- Shows the ticker, length, start date and end date of the 10 longest gaps.
- Does not fill, remove or correct these gaps.

#### check_volume
- Rejects negative and infinite volume values.
- Confirms that volume quality is `True` only for observed, finite and positive volume.
- Counts missing and zero volume during actual index membership.
- Confirms that every invalid raw volume observation is excluded by volume quality.
- Reports positive-to-positive volume changes of at least `AUDIT_VOLUME_JUMP_RATIO`.
- Reports runs of at least 5 missing or zero-volume observations.
- Confirms that invalid volume is excluded from liquidity while the price observation remains available.

#### check_prices_long
- Requires `date`, `ticker` and `price` columns in the long prices dataset.
- Checks row count and duplicated date-ticker pairs.
- Recreates long prices from the wide price matrix and compares all keys and values.

#### historical_ticker_union
- Recreates the complete historical ticker union beginning with the latest snapshot available on or before `DATA_START_DATE`.

#### build_expected_membership
- Recreates the daily membership matrix directly from historical component snapshots.
- Carries every snapshot forward until the next snapshot.
- Uses missing membership as `False`.

#### check_history_and_universe
- Validates historical component columns, dates, component counts and unique snapshots.
- Recreates membership and compares it with the saved membership matrix.
- Confirms that price data does not extend beyond the latest membership snapshot.
- Validates the columns and coverage values in `universe.csv`.
- Confirms that universe, prices and historical ticker union contain the same ticker set.
- Reports tickers without membership-period prices, coverage below 80%, missing downloads and rejected reused symbols.
- Confirms that membership counts in `universe.csv` match the membership matrix.

#### check_risk_free_rate
- Requires the `annual_rate_pct` column.
- Checks sorted unique dates, missing values, negative values and infinite values.
- Confirms that valid DGS3MO observations cover the equity data period with a seven-day boundary tolerance.

#### overall_status
- Returns `FAIL` when at least one check failed.
- Otherwise returns `WARNING` when at least one warning exists.
- Returns `PASS` only when every check passed.

#### run_check_group
- Runs one group of audit checks.
- Converts an unexpected exception inside a check group into a visible `FAIL` instead of stopping the complete audit.

#### markdown_escape
- Escapes table separators and line breaks before values are inserted into the Markdown report.

#### render_report
- Creates the main numbered checks table and file inventory.
- Adds the bundle fingerprint, total status and status counts.
- Adds interpretation rules and limitations of the audit.

#### render_summary_image
- Creates `Results/Data_System/Figures/data_audit_summary.png` from the datasets and completed audit checks.
- Shows ticker coverage during membership, Yahoo download results and audit status counts.
- Uses the same loaded data and checks as the Markdown audit report.
- Atomically replaces the previous image on every audit run.

#### render_availability_timeline
- Creates `Results/Data_System/Figures/membership_availability_timeline.png` from membership and availability.
- Calculates the percentage of actual index members with an available price on every trading date.
- Shows the daily percentage, its 63-trading-day rolling mean and the full-period mean.
- Uses a 60% to 100% vertical scale to keep the observed historical changes readable.
- Atomically replaces the previous image on every audit run.

#### write_report
- Writes the report to a temporary file in the Reports directory.
- Atomically replaces the previous `Results/Data_System/data_audit_report.md` after writing succeeds.

#### run_data_audit
- Builds the expected file list, timestamp and bundle fingerprint.
- Loads all datasets.
- Runs file, matrix, calculation, universe, price-gap, volume and risk-free-rate check groups.
- Creates and writes the final Markdown report, PNG summary and availability timeline.
- Prints the overall status and report path.
- Returns the overall status and all individual check results.
- Runs independently or automatically at the end of `pipeline.py`.


### **pipeline.py**
- Orchestrates the complete Data System workflow.
- Does not contain downloading, cleaning or audit logic itself.

#### EQUITY_REQUIRED_PATHS
- Lists every file required for a complete equity dataset.
- Includes raw prices, all processed matrices, the universe report and historical membership source.
- Does not include the risk-free-rate file because it is prepared separately.

#### prepare_equity_data
- Checks whether every file in `EQUITY_REQUIRED_PATHS` exists.
- Loads the saved equity datasets when the complete bundle exists.
- Rebuilds the complete equity bundle when at least one required file is missing.
- During a rebuild, loads historical S&P500 snapshots and creates the historical ticker union.
- Passes historical snapshots and tickers to `equity_data.py` for downloading, processing and saving.
- Does not rebuild only one missing equity file; it rebuilds the complete equity bundle to keep all matrices aligned.

#### run_pipeline
- Starts the complete Data System process.
- Calls `prepare_equity_data` to load or rebuild the equity bundle.
- Uses the first returned dataset, adjusted prices, to determine the equity end date.
- Calls `prepare_risk_free_rate` for the period from `DATA_START_DATE` to the final equity date.
- Runs `data_audit.py` after all datasets are available.
- Prints that the Data System is ready and returns the loaded equity datasets.

#### print_summary
- Receives the seven returned equity datasets.
- Prints DataFrame information for prices, returns, volume, liquidity, long prices, availability and forward returns.
- Does not change or save data.


### **delete.py**:
- Deletes all generated Data System datasets, including the risk-free rate.
- Deletes the generated audit report and both audit figures from `Results/Data_System`.
- It could be useful to clean space for further data updating.





## Data System Result

![Data System audit summary](Results/Data_System/Figures/data_audit_summary.png)

![Membership price availability through time](Results/Data_System/Figures/membership_availability_timeline.png)

**Build snapshot**:
- Equity period: `2008-01-02` to `2026-08-18`.
- Trading dates retained: `4,686`.
- Historical ticker union: `900` tickers.
- Historical component snapshots: `2,720`.
- Final wide-matrix shape: `4,686 dates x 900 tickers`.
- Long-format adjusted prices: `2,827,310` observations.

**Price availability**:
- Point-in-time membership observations: `2,350,698`.
- Available price observations during membership: `1,921,650` (`81.75%`).
- Missing or unavailable price observations during membership: `429,048` (`18.25%`).
- Tickers without any price during their membership period: `247`.
- Tickers with membership-period price coverage below 80%: `262`.
- Yahoo tickers still reported as missing: `191`.
- Reused Yahoo symbols rejected to prevent another security from entering the dataset: `20`.
- No ticker is removed from the historical universe because of missing prices; unavailable ticker-date observations remain unavailable.

**Quality effects**:
- Price quality does not clip valid large returns.
- The saved return matrix contains `92` absolute daily returns of at least 50% and no usable unconfirmed return of at least 100%.
- `33` missing or zero-volume observations during membership are rejected by `volume_quality` and excluded only from liquidity calculations.
- `36` positive-to-positive volume changes of at least 100x remain as warnings because both observations are valid positive values.
- The audit reports `277` missing-price runs longer than 5 trading dates, containing `429,023` observations across `263` tickers.

**Audit result**:
- Overall status: **WARNING**.
- Checks: `54 PASS / 4 WARNING / 0 FAIL`.
- Every required dataset opens successfully.
- All saved wide matrices are aligned and all audited calculated relationships reproduce from their source matrices.
- The risk-free-rate dataset covers the complete equity period.

**Conclusion**:

The Data System produces a structurally consistent point-in-time dataset that is ready to be consumed by the Factor Layer. No failed internal checks remain. The main unresolved limitation is historical price availability from Yahoo Finance: missing and rejected old ticker histories reduce usable membership observations and leave residual survivorship/data-availability bias in all later factor results.




## Factor Layer [2]

The purpose of this layer is to convert the completed market dataset into factor data prepared for cross-sectional research.

The layer calculates the same factor logic for every historical trading date and every available S&P500 member. It does not decide whether a factor works. It only creates the matrices that will be compared with future returns in the Factor Selection Layer.

The completed layer creates:
- `56` factor-score matrices.
- `8` forward-return matrices.
- One factor metadata table.
- One cache manifest.
- One run metadata file.

Every matrix contains `4,686` trading dates and `900` historical ticker columns. Observations from 2008 provide warm-up history for long factor windows. The intended research period begins in 2010.


**Limitations**:
- Every factor is based only on prices, returns or volume. Fundamental company information is not used.
- Factor quality is limited by the remaining Yahoo and historical-membership limitations described in the Data System.
- Winsorization is disabled, so unusual but valid factor values remain in the matrices.
- A calculated factor score is not evidence of alpha. Economic and statistical evaluation belongs to the next layer.


IMPORTANT:
- A factor value is calculated on date `t` from information available through date `t`.
- The Factor Layer does not shift the signal by one day.
- The Factor Selection Layer later uses `factor[t-1]` against a forward return beginning on date `t`.
- A factor matrix contains normalized factor scores, not ranks, IC values or predicted returns.
- A forward-return matrix contains realised future returns, not model forecasts.
- Missing source data remain missing in the final matrices.


### **factor_config.py**:
- Defines every Factor Layer data, cache and result path.
- Stores factor matrices in `Data/Factors_Layer/Cache/Factor_Matrices`.
- Stores forward-return matrices in `Data/Factors_Layer/Cache/Forward_Return_Matrices`.
- Stores factor metadata and the cache manifest in `Data/Factors_Layer/Cache`.
- Stores run metadata in `Results/Factors_Layer`.
- Requires at least 80% of observations inside every factor window.
- Keeps winsorization limits at the 1st and 99th cross-sectional percentiles.
- Sets `APPLY_WINSORIZATION = False`, so these limits are not currently applied.
- Uses `252` trading days as the annualization factor.
- Defines eight forward-return horizons: `1`, `5`, `10`, `21`, `42`, `63`, `126` and `252` trading days.
- Defines `56` parameter configurations across `11` factor families.
- Does not contain signal lag, research periods, IC settings or robustness windows. These belong to the Factor Selection Layer.


### **factors.py**:
- Contains the mathematical logic of every factor family.
- Does not contain file paths, parameter grids, saving, ranking, IC calculation or factor selection.
- Receives every window and parameter directly from `factor_builder.py`.


**Momentum**
- Measures cumulative historical return.
- A higher raw value means that the stock rose more strongly during the formation period.
- Can skip the most recent observations before calculating the formation return.
- *Different windows and skipped recent periods are defined in **`factor_config.py`**.*


**Low Volatility**
- Measures the rolling standard deviation of daily returns.
- Lower volatility is preferred, so the result is multiplied by `-1`.
- A higher factor value therefore represents a less volatile stock.


**Trend**
- Calculates `price / moving average - 1`.
- Measures how far the current price is above or below its moving average.
- A positive value means that price is above the moving average.


**Short-Term Reversal**
- Measures recent cumulative return with the opposite sign.
- Recent losers receive higher values and recent winners receive lower values.
- Tests whether short-term price movements tend to reverse.


**Residual Momentum**
- Calculates the equal-weighted return of all available index members on every date.
- Subtracts that market return from every stock return.
- Calculates Momentum from the remaining stock-specific return.
- A higher value means that the stock outperformed the cross-sectional market component.


**Volatility-Scaled Momentum**
- Calculates historical Momentum.
- Divides Momentum by the stock's rolling volatility.
- Rewards strong historical return relative to the amount of price variation used to produce it.
- Replaces zero volatility with missing data before division.


**High Proximity**
- Finds the highest price inside the rolling window.
- Calculates `current price / rolling high - 1`.
- A value closer to zero means that the stock trades closer to its historical high.


**Trend Slope**
- Fits a straight trend line to logarithmic prices inside every rolling window.
- Uses the slope of that line as the factor value.
- Converts the daily slope into an annual rate using `252` trading days.
- Measures the speed and direction of the price trend instead of distance from a moving average.


**Risk-Adjusted Trend**
- Calculates annualized Trend Slope.
- Calculates annualized return volatility over the same window.
- Divides Trend Slope by volatility.
- Rewards a strong trend relative to the instability of daily returns.


**Liquidity Change**
- Calculates daily dollar volume as `price x volume`.
- Compares its short rolling average with its long rolling average.
- Calculates the logarithm of `short average / long average`.
- A positive value means that recent trading activity is higher than its longer-term level.


**Price-Volume Confirmation**
- Starts with normalized Momentum.
- Changes its strength using normalized Liquidity Change.
- Increasing liquidity strengthens Momentum and decreasing liquidity weakens it.
- Limits the liquidity adjustment before combining both signals.



#### _cumulative_log_return
- Converts daily simple returns into logarithmic returns using `log(1 + return)`.
- Shifts returns when the most recent observations must be skipped.
- Adds the remaining log returns inside the formation window.
- Rejects impossible window and minimum-observation settings.


#### _rolling_log_price_slope
- Converts prices into logarithmic prices.
- Fits a rolling linear trend separately for every ticker.
- Handles missing prices by using only observed values inside each window.
- Converts the estimated daily slope into an annual rate.
- Rejects windows with too few observations or no valid time variation.


### **transforms.py**:
- Contains the common cross-sectional preparation applied after a raw factor is calculated.


#### winsorize
- Finds the configured lower and upper factor percentiles separately on every date.
- Replaces values outside these boundaries with the boundary values.
- Does nothing in the current pipeline because winsorization is disabled.


#### zscore
- Calculates the cross-sectional mean and standard deviation on every date.
- Subtracts the same-date mean from every available stock value.
- Divides the result by the same-date standard deviation.
- A score of `0` represents the cross-sectional average on that date.
- A score of `1` represents one cross-sectional standard deviation above the average.
- Replaces zero standard deviation with missing data.


#### prepare_factor
- Removes factor observations that are not available according to the Data System availability matrix.
- Applies winsorization only when it is enabled in `factor_config.py`.
- Applies cross-sectional z-score normalization.
- Returns the final factor-score matrix.


### **factor_builder.py**:
- Connects configurations from `factor_config.py` with formulas from `factors.py`.
- Builds the complete set of factor-score matrices.


#### required_observations
- Multiplies a factor window by the common 80% observation requirement.
- Rounds the result upward.
- Requires at least two observations.


#### configuration_parameters
- Removes the readable `variant` name from one factor configuration.
- Keeps the numerical parameters that produced the matrix.
- These parameters are later stored in factor metadata.


#### factor_key
- Combines factor family and variant into one unique `family|variant` identifier.


#### prepare_raw_factor
- Sends one raw factor matrix to `transforms.py`.
- Applies availability, optional winsorization and cross-sectional z-score normalization.


#### build_factor_scores
- Receives one factor family, one parameter configuration and the Data System inputs.
- Selects the correct function from `factors.py`.
- Passes prices, returns, volume, windows and minimum observations required by that factor.
- Uses volume only after the volume-quality mask has been applied.
- Returns one prepared date-by-ticker factor-score matrix.


#### build_factor_matrices
- Goes through all 56 configurations in `factor_config.py`.
- Calls `build_factor_scores` for every configuration.
- Saves every factor matrix through `factor_storage.py`.
- Creates a metadata row containing key, family, variant, parameters and file path.
- Returns one metadata table with exactly 56 unique rows.
- Raises an error when the number of matrices is not 56 or a key is duplicated.


### **forward_returns.py**:
- Creates the realised-return matrices required for later factor evaluation.
- Does not use a forecasting model.


#### compute_forward_returns
- Receives adjusted prices, price quality and one horizon `h`.
- Calculates `price[t+h] / price[t] - 1`.
- Requires an observed and valid price at both `t` and `t+h`.
- Keeps the result missing when either endpoint is unavailable or failed price quality.


#### build_forward_return_matrices
- Calls `compute_forward_returns` for all eight configured horizons.
- Creates one date-by-ticker matrix for every horizon.
- Saves every matrix through `factor_storage.py`.
- Does not rank returns or combine them with factor scores.


### **factor_storage.py**:
- Loads Factor Layer inputs and controls all Factor Layer file locations, saving and cache validation.


#### prepare_factor_directories
- Creates the Factor Layer Data, Cache, Factor Matrices, Forward Return Matrices and Results directories.


#### load_factor_inputs
- Loads adjusted prices, returns, compatible volume, availability, membership, price quality and volume quality from the Data System.
- Uses the price matrix as the reference for all dates and ticker columns.
- Aligns every remaining matrix to that reference.
- Converts availability and quality matrices to boolean values.
- Removes invalid volume before volume reaches liquidity-based factors.
- Requires unique sorted dates and unique ticker columns.
- Confirms that availability equals `price exists AND membership AND price quality`.
- Returns all prepared inputs to `pipeline.py`.


#### safe_factor_name
- Combines factor family and variant into a safe file name.
- Replaces spaces and unsupported path separators.


#### factor_matrix_cache_path
- Returns the Parquet path for one factor-score matrix.


#### forward_return_matrix_cache_path
- Returns the Parquet path for one forward-return horizon.


#### expected_factor_matrix_paths
- Creates the complete list of 56 expected factor-matrix paths from `factor_config.py`.


#### expected_forward_return_matrix_paths
- Creates the complete list of eight expected forward-return matrix paths.


#### temporary_path
- Creates the temporary file name used during atomic saving.


#### save_parquet
- Saves a DataFrame to a temporary Parquet file.
- Replaces the previous final file only after saving succeeds.


#### save_csv
- Saves a DataFrame to a temporary CSV file.
- Replaces the previous final file only after saving succeeds.


#### save_json
- Saves a dictionary to a temporary JSON file.
- Replaces the previous final file only after saving succeeds.


#### file_state
- Records the relative path, file size and modification time of one Data System input.


#### source_hash
- Reads one source-code file and calculates its SHA-256 hash.
- Allows the cache to detect changes in factor calculation code.


#### cache_signature_payload
- Collects the states of all Data System inputs.
- Collects hashes of `factors.py`, `transforms.py`, `factor_builder.py` and `forward_returns.py`.
- Collects every Factor Layer setting that affects calculated values.
- Returns the complete cache description.


#### cache_signature
- Converts the cache description into one stable SHA-256 signature.


#### current_cache_manifest
- Combines the current cache signature with the complete cache description.


#### save_cache_manifest
- Saves the current cache manifest after all Factor Layer matrices are successfully created.


#### factor_metadata_is_valid
- Checks that factor metadata can be opened.
- Requires the expected columns, 56 rows and unique factor keys.


#### factor_cache_is_valid
- Requires all 56 factor matrices and all eight forward-return matrices.
- Requires factor metadata and the cache manifest.
- Requires valid 56-row factor metadata.
- Recalculates the current cache signature and compares it with the saved signature.
- Returns `False` when data, code, settings or required files changed.


#### save_factor_matrix
- Converts one factor-score matrix to `float32`.
- Saves it atomically to its configured Parquet path.
- Returns the saved path for factor metadata.


#### save_forward_return_matrix
- Converts one forward-return matrix to `float32`.
- Saves it atomically to its configured Parquet path.


#### save_factor_metadata
- Saves the 56-row factor metadata table as CSV.


#### load_factor_metadata
- Loads the saved factor metadata.
- Requires exactly 56 unique factor keys.


#### save_run_metadata
- Saves the current Factor Layer run information as JSON.


### **pipeline.py**:
- Orchestrates the complete Factor Layer workflow.
- Does not contain factor formulas, transformations, file-writing implementation or factor evaluation logic.


#### prepare_factor_cache
- Checks whether the complete Factor Layer cache is valid.
- Loads existing factor metadata without recalculating matrices when the cache is valid.
- Rebuilds all 56 factor matrices and eight forward-return matrices when the cache is missing or outdated.
- Saves new factor metadata and the cache manifest after a successful rebuild.
- Returns factor metadata and whether the cache was reused.


#### run_pipeline
- Creates all required Factor Layer directories.
- Loads completed Data System matrices through `factor_storage.py`.
- Calls `prepare_factor_cache` to reuse or rebuild the Factor Layer cache.
- Saves the run time, data period, matrix dimensions, factor counts, horizons and cache status.
- Prints the final number of factor and forward-return matrices.
- Returns the 56-row factor metadata table.
- Does not calculate IC, robustness, quantiles or factor selection.


### **delete.py**:
- Deletes all generated Factor Layer matrices, metadata, cache files and results.
- Does not delete Data System or Factor Selection Layer files.
- It can be used before a complete Factor Layer rebuild.


## Factor Layer Result

**Build snapshot**:
- Equity period: `2008-01-02` to `2026-08-18`.
- Trading dates: `4,686`.
- Historical ticker columns: `900`.
- Factor families: `11`.
- Factor configurations: `56`.
- Forward-return horizons: `8`.
- Every matrix shape: `4,686 dates x 900 tickers`.

**Stored matrices**:
- Factor-score matrices: `56 / 56`.
- Total factor-score cells: `236,174,400`.
- Observed factor scores: `104,582,220`.
- Factor matrix storage: approximately `580.55 MB`.
- Forward-return matrices: `8 / 8`.
- Total forward-return cells: `33,739,200`.
- Observed forward returns: `22,218,069`.
- Forward-return matrix storage: approximately `122.09 MB`.

**Cache result**:
- Factor metadata contains `56` unique rows.
- All 64 required matrices exist.
- Factor and forward-return matrices use the same dates and ticker columns.
- The saved cache manifest matches the current Data System files, factor code and configuration.
- Current cache status: **valid**.

**Conclusion**:

The Factor Layer is complete. It produces a reproducible set of normalized factor scores and realised forward returns without evaluating or selecting factors. These matrices are ready to be combined in the Factor Selection Layer.


## Factor Selection Layer [3]

The purpose of this layer is to identify the shape and strength of the relationship between factor scores and future cross-sectional returns. It combines the 56 Factor Layer configurations with eight return horizons, creating 448 separate hypotheses. It does not recalculate factor formulas and it does not yet build a trading portfolio.

The data foundation is `daily_quantile_results.parquet`. One row represents one trading date and one factor-horizon hypothesis. The complete file contains 2,099,328 rows: 4,686 dates multiplied by 56 factor configurations and eight horizons.

The selection system then converts these daily observations into one readable card per hypothesis. It recognizes positive and negative monotonic relationships, upper-tail effects, lower-tail effects, two-tail effects and cases without stable structure. Positive IC is not required: a stable negative relationship can also be useful when the factor direction is reversed.

The current setting uses `ACTIVE_FACTOR_KEYS = None`. Every run therefore analyzes all 56 factor configurations, all eight horizons, 448 hypotheses and 2,240 separate effect tests.


### selection_config.py
- Defines every Layer 3 input and output path.
- Uses the same eight return horizons produced by the Factor Layer.
- Uses a one-day signal lag, ten quantiles and at least 30 valid stocks per daily comparison.
- Marks dates from `2010-01-01` as research observations while keeping 2008-2009 warm-up rows in the source dataset.
- Defines `ACTIVE_FACTOR_KEYS = None` for the complete 448-hypothesis scope.
- Defines the initial descriptive boundaries for monotonicity, tail dominance, data coverage and time consistency.
- Defines the 5% multiple-testing level and the maximum number of factor-specific figure sets.


### selection_storage.py

#### prepare_selection_directories
- Creates Layer 3 Data, Cache, Results and Figures directories.

#### safe_factor_name
- Converts a factor family and variant into the same safe file name used by the Factor Layer.

#### factor_matrix_path
- Returns the expected path of one factor-score matrix.

#### forward_return_matrix_path
- Returns the expected path of one forward-return matrix.

#### load_factor_metadata
- Loads the Factor Layer metadata table.
- Requires exactly 56 unique factor configurations.

#### load_membership
- Loads point-in-time index membership.
- Requires unique sorted dates and ticker columns.

#### validate_matrix_axes
- Rejects a matrix when its dates or tickers differ from membership.

#### load_factor_matrix
- Loads and validates one factor-score matrix.

#### load_forward_return_matrices
- Loads and validates all requested forward-return matrices.

#### load_factor_quantile_results
- Reads only one requested factor configuration from the 2.1-million-row Parquet file.
- Uses Parquet filtering so the complete dataset does not need to be loaded into memory.

#### temporary_path
- Creates the temporary path used for atomic output replacement.

#### save_json
- Saves run metadata atomically.

#### save_csv
- Saves readable result tables atomically.

#### save_parquet
- Saves large internal tables atomically with Zstandard compression.

#### save_text
- Saves the Markdown report atomically.

#### save_parquet_chunks
- Streams daily result chunks into one compressed temporary Parquet file.
- Replaces the previous complete file only after every chunk succeeds.


### quantile_analysis.py

#### prepare_factor_quantiles
- Shifts the factor matrix by one trading day.
- Keeps only stocks that are index members on the new trade date.
- Ranks the available factor scores cross-sectionally on every date.
- Assigns each stock to Q1-Q10 without using its future return.

#### grouped_median
- Calculates one median for every date-quantile group.
- Keeps empty groups as missing.

#### rowwise_linear_relationship
- Calculates cross-sectional correlation and factor beta for every date.
- Requires at least 30 valid factor-return pairs.

#### compute_daily_relationship_metrics
- Calculates daily Spearman Rank IC, Pearson correlation and factor beta.

#### aggregate_quantile_relationships
- Combines factor scores, quantile assignments and realised future returns.
- Calculates stock count, mean factor score, median factor score, mean return and median return inside Q1-Q10.

#### research_date_mask
- Marks observations from 2010 onward as eligible for research conclusions.

#### build_quantile_result_chunk
- Creates one row per date for one factor-horizon hypothesis.
- Stores coverage, daily relationship metrics and Q1-Q10 factor and return statistics.
- Stores both mean-return and median-return `Q10 - Q1` spreads.

#### quantile_result_chunks
- Loads one of the 56 factor matrices at a time.
- Reuses its quantile assignments across all eight horizons.
- Produces 448 Parquet row groups without holding the complete result in memory.

#### run_quantile_analysis
- Creates `daily_quantile_results.parquet` and its run metadata.
- Verifies the expected 2,099,328-row result.
- Can be run independently when the daily foundation must be rebuilt.


### hypothesis_analysis.py

#### select_factor_metadata
- Applies the temporary active-factor restriction.
- Uses all 56 configurations when the restriction is `None`.

#### required_quantile_columns
- Lists only the Parquet columns required for hypothesis analysis.

#### hac_tstat
- Calculates a t-statistic adjusted for autocorrelation caused by overlapping forward-return horizons.

#### direction_rate
- Measures how often an effect has the same sign as its complete-period mean.

#### period_means
- Converts daily observations into monthly or annual mean effects.

#### summarize_effect
- Calculates coverage, mean, median, standard deviation, HAC t-statistic and approximate p-value.
- Measures daily, monthly and annual direction consistency.

#### daily_effects
- Creates five separate daily relationships: Spearman IC, `Q10 - Q1`, `Q10 - middle`, `middle - Q1` and both edges against the middle.
- Uses Q4-Q7 as the middle of the factor distribution.

#### quantile_curve
- Creates the average ten-point factor-score and future-return curve for one hypothesis.

#### curve_shape_statistics
- Measures Q1-Q10 monotonicity and the share of adjacent quantile steps moving in the same direction.

#### rank_autocorrelation_by_horizon
- Compares cross-sectional factor ranks with their previous ranks after 1, 5, 10, 21, 42, 63, 126 and 252 trading days.
- Shows how quickly the factor ranking changes.

#### analyze_factor
- Reads one factor configuration from the daily Parquet dataset.
- Calculates all eight hypothesis cards, 40 effect tests, quantile curves and time-stability observations.

#### build_hypothesis_metrics
- Repeats the same analysis for every active factor configuration.
- Returns the complete tables required for classification and reporting.


### pattern_classification.py

#### apply_multiple_testing
- Applies Benjamini-Hochberg false-discovery-rate correction across every effect test in the active run.
- Marks the correction as preliminary while fewer than all 448 hypotheses are active.
- Also reports a conservative Bonferroni value for the planned full 2,240 effect tests.

#### statistically_visible
- Checks whether the absolute HAC t-statistic crosses the initial descriptive boundary.

#### classify_pattern
- Classifies each hypothesis as positive monotonic, negative monotonic, upper tail, lower tail, both tails against the middle or no stable structure.
- Keeps pattern direction separate from pattern strength.

#### strongest_economic_effect
- Finds the largest absolute HAC t-statistic among the four return-based effects.
- Reports it separately without treating a significant IC as an economic return pattern.

#### primary_effect_values
- Selects the effect that directly represents the detected pattern.
- Prevents every pattern from being judged only by Mean IC.

#### evidence_status
- Separates insufficient data, weak patterns, unstable patterns and provisional candidates.
- Uses final FDR candidate status only when the full 448-hypothesis run is complete.

#### classification_reason
- Writes a short human-readable explanation of the assigned pattern.

#### classify_hypotheses
- Applies pattern recognition, multiple-testing information and evidence status to every hypothesis card.
- Records the strongest economic effect separately from IC.
- Uses `primary_effect = none` when no stable economic structure is detected.
- Does not delete hypotheses from the result table.


### classifier_validation.py

#### validation_cases
- Defines six transparent examples with known positive monotonic, negative monotonic, upper-tail, lower-tail, two-tail and no-structure results.
- Does not use real market observations or optimize classification boundaries.

#### run_classifier_validation
- Sends every known example through the same pattern classifier used for real hypotheses.
- Saves `classifier_validation.csv`.
- Stops the Layer 3 pipeline when any expected pattern is classified incorrectly.


### selection_report.py

#### save_figure
- Saves every figure atomically.

#### factor_figure_path
- Creates a readable figure name from the factor key.

#### heatmap_panel
- Draws one factor-by-horizon metric panel.

#### plot_hypothesis_overview
- Visualizes Mean IC, the strongest economic-effect t-statistic, monthly consistency and rank autocorrelation across all 448 hypotheses.

#### plot_quantile_curves
- Shows Q1-Q10 returns relative to the Q4-Q7 middle for all eight horizons.

#### plot_factor_dashboard
- Shows IC, strongest economic-effect strength, time consistency and rank stability for one factor configuration.

#### plot_monthly_ic
- Creates eight year-by-month IC heatmaps for one factor configuration.

#### factor_keys_for_figures
- Limits detailed figures to the most informative active factor configurations.
- Prevents a full run from creating 448 separate graph sets.

#### create_selection_figures
- Creates the complete visual output for the active analysis.

#### format_number
- Formats report values consistently and keeps missing values visible as `NA`.

#### markdown_cell
- Escapes factor keys so vertical bars do not break Markdown tables.

#### markdown_table
- Converts IC and stability diagnostics into a compact readable table.

#### economic_markdown_table
- Shows the four directly interpretable return effects and their HAC t-statistics.
- Keeps economic patterns in front of IC diagnostics.

#### build_selection_report
- Creates `factor_selection_report.md` with scope, pattern counts, evidence status, hypothesis cards, figures and interpretation rules.
- Shows at most 50 detected patterns in Markdown while keeping all 448 rows in CSV.
- Clearly separates economic pattern evidence from supporting IC information.


### **pipeline.py**

#### prepare_quantile_data
- Reuses `daily_quantile_results.parquet` when it already exists.
- Rebuilds it only when the file is missing.

#### run_factor_classification
- Validates all six known classifier patterns before reading real results.
- Calculates hypothesis metrics and pattern classifications.
- Saves cards, effect tests, quantile curves, time stability, figures and the Markdown report.

#### run_pipeline
- Runs the complete current Factor Selection Layer.
- Saves separate metadata describing the active scope and created results.
- Prints the total execution time.
- Does not run the Factor Layer or any Research Layer validation.


## Factor Selection Layer Result

The complete run analyzes 56 factor configurations across eight horizons. It creates:

- `hypothesis_cards.csv`: one readable classification row per hypothesis.
- `effect_tests.csv`: five separate relationship tests per hypothesis.
- `quantile_curves.csv`: Q1-Q10 factor and return curves.
- `classifier_validation.csv`: the six-case classifier self-check.
- `time_stability.parquet`: monthly and annual effect histories used by the figures.
- `factor_selection_report.md`: the main readable report.
- `Figures/`: the complete-scope overview and detailed figures for the active factor.

The completed full run found 426 hypotheses without stable structure and 22 descriptive economic patterns. Ten patterns were unstable through time and twelve were rejected after global multiple-testing correction. No economic hypothesis became a final candidate. One volatility-scaled momentum IC result survived global FDR, but its Q10-Q1 return spread was not statistically supported. These results define the frozen candidate leads passed to the Research Layer.


## Research Layer [4]

The purpose of this layer is to test whether the frozen Factor Selection leads can be converted into an economically usable portfolio. It does not redefine factor formulas and it does not search the complete 448-hypothesis grid again.

The current shortlist contains four deliberately labelled leads: the globally significant one-day volatility-scaled momentum rank signal and three exploratory return-pattern leads from short-term reversal, liquidity change and low volatility. Passing a lead into this layer does not mean that alpha has already been found.

Layer 4 compares continuous and tail-based portfolio construction, four rebalance frequencies, calendar starting phases, beta-neutral and unconstrained implementations, four transaction-cost assumptions, full-history risk statistics, two walk-forward structures and point-in-time market regimes.

The walk-forward output is explicitly post-selection. The candidates were already discovered using the complete historical sample, so historical OOS windows cannot become a genuinely untouched final test after the fact. They show temporal behaviour and implementation stability without erasing this limitation.


### research_config.py
- Defines every current Research Layer input, cache, result and figure path.
- Freezes the four candidate leads received from the completed Selection Layer.
- Defines six portfolio-construction methods.
- Uses 1, 5, 21 and 63-day rebalance frequencies.
- Defines 0, 5, 10 and 25 bps transaction-cost scenarios, with 10 bps as the primary result.
- Defines trailing beta, risk, walk-forward and market-regime settings.


### research_storage.py

#### prepare_research_directories
- Creates current Layer 4 Cache, Results and Figures directories.

#### input_signature
- Records the size and modification time of every required input.
- Includes the frozen candidates and portfolio configuration.
- Creates one signature used to decide whether the expensive portfolio cache is still compatible.

#### portfolio_cache_is_valid
- Reuses portfolio paths only when the signature, every required matrix, metadata and calendar-phase result are present.

#### save_portfolio_cache
- Saves reusable portfolio return, turnover and exposure matrices in `Data/Research_Layer/Cache`.
- Saves readable path metadata in `Results/Research_Layer`.

#### load_portfolio_cache
- Loads all compatible portfolio matrices and path metadata without recalculating positions.

#### clear_current_research_outputs
- Deletes only current Research Layer data and results.


### research_data.py

#### load_research_data
- Loads daily returns, point-in-time membership and final price availability including the 2008-2009 warm-up history.
- Aligns all dates and ticker columns.
- Keeps portfolio and walk-forward results limited to 2010 onward.

#### load_risk_free_daily
- Loads the annual three-month Treasury rate.
- Forward-fills it only across the already available historical dates.
- Converts the annual percentage rate into a daily rate used for long-only Sharpe calculations.

#### candidate_table
- Locates every frozen candidate in Factor Layer metadata.
- Adds its Selection Layer pattern, status, IC and primary economic-effect information.
- Rejects factor paths outside the Factor Layer directory.

#### load_candidate_factors
- Loads and aligns only the four factor matrices required by Layer 4.

#### load_sector_history
- Loads optional point-in-time sector history when `sector_history.csv` exists.
- Requires ticker, sector, start date and end date.
- Reports `NOT_AVAILABLE` instead of applying current sectors to old companies when the file is absent.

#### build_sector_matrix
- Converts available point-in-time sector intervals into a date-by-ticker matrix.


### multiple_testing_research.py

#### bh_values
- Calculates Benjamini-Hochberg q-values without changing the original p-values.

#### simes_p_value
- Combines the tests inside one predefined family into one family-level p-value.

#### add_group_bh
- Recalculates exploratory FDR separately by effect, factor family and factor-family/effect group.

#### add_hierarchical_fdr
- First tests all predefined factor-family/effect groups.
- Applies a second within-group correction only inside groups that passed the first stage.
- Keeps the hierarchical result separate from the original global FDR result.

#### compare_multiple_testing
- Compares raw p-values, global FDR, exploratory within-group FDR and hierarchical FDR across all 2,240 Selection Layer tests.
- Does not replace the completed Selection Layer decision after seeing the data.


### factor_overlap.py

#### rowwise_correlation
- Calculates one cross-sectional correlation for every date with at least 30 common stocks.

#### run_factor_overlap
- Compares the daily ranks of every pair of frozen candidates.
- Shows whether different formulas mostly reproduce the same stock ordering.


### portfolio_construction.py

#### percentile_ranks
- Ranks only eligible point-in-time index members on the current trade date.

#### continuous_weights
- Uses the complete factor ranking instead of discarding stocks outside Q1 and Q10.
- Gives larger absolute weights to stocks farther from the cross-sectional middle.
- Creates equal long and short gross exposure.

#### equal_leg_weights
- Creates equal-weight long and short baskets with at least five names in each leg.

#### long_only_weights
- Creates a one-sided Q1 or Q10 basket with total exposure equal to one.

#### beta_neutralize
- Uses trailing known betas to scale long and short books toward zero estimated market beta.
- Excludes names without sufficient beta history from the beta-neutral implementation.
- Does not silently use an unconstrained portfolio when neutralization is impossible.

#### build_target_weights_from_ranks
- Creates continuous high-minus-low, Q10-Q1, Q10-middle, middle-Q1, long-Q10 or long-Q1 target weights.

#### build_target_weights
- Combines daily ranking and the requested portfolio rule.


### risk_exposure.py

#### build_market_return
- Creates an equal-weight point-in-time S&P500 return proxy from available members.

#### build_trailing_betas
- Estimates every stock beta from trailing 252-day returns.
- Requires at least 126 observations and shifts beta by one day.

#### expanding_binary_state
- Compares one current market indicator only with its expanding historical median.
- Prevents future regime information from defining past thresholds.

#### build_market_regimes
- Creates point-in-time high/low volatility, dispersion and average-correlation states.
- Separately marks whether the previously known US three-month rate is above 2%.


### portfolio_engine.py

#### calendar_offsets
- Creates up to three evenly spaced starting phases for each rebalance frequency.

#### path_specifications
- Combines four candidates, six portfolio methods, beta constraints, rebalance frequencies and calendar phases.

#### drift_weights
- Carries existing position weights through realised daily stock returns between rebalances.

#### simulate_factor_paths
- Calculates one cross-sectional rank per factor and date.
- Updates all methods, frequencies and calendar phases for that factor together.
- Uses the previous trading date's factor to set positions before applying the current `t-1 -> t` return.
- Applies point-in-time membership, records forced exits and never uses a missing return as verified price movement.
- Records gross return, turnover, beta, gross and net exposure, maximum position, holding count, unpriced weight and available sector diagnostics.

#### build_phase_paths
- Runs the optimized portfolio engine for every frozen candidate.

#### aggregate_calendar_phases
- Treats different start dates as equal portfolio sleeves and averages their daily paths.
- Creates one ensemble path for every factor, method, constraint and rebalance frequency.
- Saves the dispersion between individual calendar phases separately.


### portfolio_evaluation.py

#### apply_transaction_costs
- Deducts one-way trading costs from realised turnover.

#### annualized_return
- Compounds the daily portfolio path and annualizes it using 252 trading days.

#### annualized_sharpe
- Calculates annualized return per unit of daily volatility.

#### maximum_drawdown
- Measures the largest peak-to-trough loss of the compounded path.

#### alpha_beta_test
- Regresses portfolio returns on the point-in-time market proxy.
- Uses HAC standard errors for annualized alpha and its t-statistic.

#### evaluate_one_path
- Calculates net return, volatility, Sharpe, drawdown, time consistency, alpha, beta, turnover, cost drag and data-quality diagnostics.
- Uses return above the risk-free rate for long-only portfolios.
- Uses the self-financing spread return for market-neutral portfolios.

#### evaluate_portfolios
- Applies all four transaction-cost assumptions to every calendar-phase ensemble path.


### walk_forward_research.py

#### walk_forward_periods
- Creates a short 18-month training / 6-month OOS sequence.
- Creates a long four-year training / one-year OOS sequence.
- Moves each sequence forward by exactly its OOS length.

#### oriented_net_returns
- Applies the direction selected from the preceding training period.
- Reverses gross market-neutral return without reversing transaction costs.

#### run_walk_forward
- Keeps every portfolio method and parameter setting in the result.
- Chooses only the market-neutral direction from its preceding training observations.
- Freezes that direction over the next OOS period.
- Saves every individual period, complete stitched OOS paths and one summary per implementation.
- Marks all results as post-selection rather than untouched OOS evidence.


### regime_research_current.py

#### run_regime_analysis
- Separates completed walk-forward returns by risk-free-rate, volatility, dispersion and correlation state.
- Reports return, Sharpe and positive-day frequency inside every state.


### research_report.py
- Creates figures for multiple-testing sensitivity, factor overlap, costs, turnover, walk-forward paths and regimes.
- Creates `research_report.md` with frozen candidates, statistical sensitivity, implementation results, risk diagnostics, walk-forward behaviour and interpretation limits.


### **pipeline.py**

#### run_multiple_testing_comparison
- Rebuilds the readable FDR sensitivity tables from the completed Selection Layer effects.

#### prepare_portfolio_paths
- Reuses a compatible portfolio cache.
- Otherwise calculates trailing betas, all calendar-phase paths and phase ensembles once.

#### risk_summary
- Extracts beta, leverage, turnover, concentration, sector and missing-return diagnostics at the primary 10 bps cost assumption.

#### run_pipeline
- Runs the complete current Research Layer in chronological order.
- Always recreates readable statistics, walk-forward results, figures and the Markdown report.
- Rebuilds expensive portfolio paths only when their inputs or configuration changed.
- Prints the complete execution time.
- Does not run or change Data System, Factor Layer or Factor Selection Layer.


### delete.py
- Deletes current Research Layer cache and readable results.
- Does not delete any previous project layer.


## Research Layer Interpretation

- Full-history portfolio results are descriptive because the same history influenced the candidate shortlist.
- Global FDR remains the original confirmatory multiple-testing result.
- Within-family and hierarchical corrections show whether the original correction was too broad, but they are not used to rewrite the discovery result.
- Walk-forward analysis measures historical temporal behaviour but cannot recreate a genuinely untouched test after the candidates have already been viewed on the complete history.
- Sector exposure remains explicitly unavailable until a reliable point-in-time sector dataset is added to the Data System.
- A factor is not called alpha unless its return survives costs, turnover, risk exposure, calendar phases, both walk-forward structures and market-regime analysis.

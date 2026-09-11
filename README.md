# WORK IN PROGRESS
Status: This project is currently under active development. Some features might not work as expected.


**Project Roadmap**:

| stage        | status              |
|--------------|---------------------|
| Data System  | **completed**       |
| Factor Layer | <- here right now   |


### Project Structure
src/


  - Data_System/
    - _init_.py
    - config.py
    - risk_free_rate.py
    - get_tickers.py
    - data_quality.py
    - equity_data.py
    - data_audit.py
    - **pipeline.py**
    - delete.py


  - Factors_Layer/
    - **pipeline.py**
    - factors.py
    - transforms.py
    - research.py
    - candidate_research.py
    - walk_forward.py


## Data [1]

During that stage the data has been downloaded and initially cleaned and prepared.

At the end there are 11 core equity datasets with different metrics and formats:
- Six **"Processed"** files.
- Five **"Raw"** files.
- One additional macro file with the risk-free rate.


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
- Defines paths for Raw, Processed, Cache and Reports files.
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
- `CONFIRMED_REAL_RETURN_EVENTS` contains manually verified extreme market moves. Currently: `HIG` on `2008-12-05`.
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
- Keeps the existing `Data/Raw/dgs3mo.parquet` without downloading it again.
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
- Downloads point-in-time S&P500 component snapshots from the community-maintained `fja05680/sp500` GitHub repository.
- Uses `S&P 500 Historical Components & Changes (Updated).csv` and keeps a local copy in `Data/Raw`.
- Falls back to the local copy if the online source is temporarily unavailable.
- Requires `date` and `tickers` columns, parses and sorts dates, and keeps the last snapshot when a date is duplicated.
- Normalizes ticker symbols to uppercase and replaces dots with dashes for yfinance compatibility.
- Validates that every relevant snapshot contains between 450 and 550 components and has no duplicated ticker.
- Extracts the union of all historical tickers beginning with the latest snapshot available on or before `DATA_START_DATE`.
- Returns the component list for any requested date using the most recent snapshot available on or before that date.
- Builds the daily membership matrix by carrying every snapshot forward until the next index change.
- Raises an error when membership history does not cover the first requested trading date.
- Warns if requested trading dates extend beyond the latest available membership snapshot.


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

#### check_extreme_gaps
- Finds consecutive missing-price observations during actual index membership.
- Prints a warning for ticker gaps longer than 5 trading dates.
- Does not fill, remove or correct these gaps.

#### save_all
- Saves adjusted prices, compatible volume and liquidity in `Data/Raw`.
- Saves returns, forward returns, long prices, membership, data quality, volume quality and availability in `Data/Processed`.
- Saves the universe report as `Data/Raw/universe.csv`.
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
- Runs sanity and gap diagnostics.
- Saves all equity datasets and the universe report.
- Returns prices, returns, volume, liquidity, long prices, availability and forward returns.


### **delete.py**:
- Deletes all generated Data System files, including the risk-free rate.
- It could be useful to clean space for further data updating.


### **pipeline.py**
- Executes whole code - data importing / rebuilding.


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
- Used to find long missing or zero-volume runs.

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

#### write_report
- Writes the report to a temporary file in the Reports directory.
- Atomically replaces the previous `Data/Reports/data_audit_report.md` after writing succeeds.

#### run_data_audit
- Builds the expected file list, timestamp and bundle fingerprint.
- Loads all datasets.
- Runs file, matrix, calculation, universe, volume and risk-free-rate check groups.
- Creates and writes the final report.
- Prints the overall status and report path.
- Returns the overall status and all individual check results.
- Runs independently or automatically at the end of `pipeline.py`.


## Factor Layer [2]

The goal of that stage is to build factor architecture and search for robust cross-sectional factors.


### Baseline factors

**Momentum**
- Measures previous cumulative return.
- Different windows and skipped recent periods are tested.
- Winsorized and Normalized.


**Low Volatility**
- Measures historical return volatility.
- Lower volatility is better, so factor sign is negative.
- Winsorized and Normalized.


**Trend**
- Price / SMA - 1.
- Measures how far price is from moving average.
- Winsorized and Normalized.


### Candidate factors
- Short-Term Reversal
- Residual Momentum
- Volatility-Scaled Momentum
- High Proximity
- Trend Slope
- Risk-Adjusted Trend
- Liquidity Change
- Price-Volume Confirmation


### transforms.py
- Winsorize values cross-sectionally for every date.
- Default limits are 1% and 99%.
- Normalize factor values with cross-sectional z-score.
- After normalization factor mean is near 0 and standard deviation is near 1 for every date.


### Factor pipeline.py
- Loads returns, prices, availability and forward returns datasets.
- Builds factor matrices.
- Uses availability mask before factor transformation.
- Calculates Spearman Rank IC between factor score known at t-1 and forward return from t to t+h.
- Requires the stock to be an index member on the evaluation date.
- Dates with less than 30 valid assets are excluded.


### Factor sensitivity
- Tests 56 factor specifications.
- Tests 5, 21, 63 and 126 trading-day forward return horizons.
- Uses one minimum data coverage rule: 80% of every effective factor window.
- Uses the same factor settings and horizons in every research window.
- Factor calculations may use warm-up data from 2008. Evaluation starts in 2010.


### Walk-forward robustness
- Short layer: 18 months selection -> next 6 months OOS -> 6 months shift.
- Long layer: 4 years selection -> next 1 year OOS -> 1 year shift.
- Parameters are selected only from past data.
- OOS results are not used for parameter selection.


#### IC statistics
- Mean IC shows average factor predictive power.
- Std IC shows how unstable IC is over time.
- T-stat shows if average IC is statistically different from zero.
- IC > 0 shows how often factor has positive predictive power.
- IC autocorrelation shows if factor IC is persistent between periods.


IMPORTANT:
- Factor signal uses information available at t-1 and is evaluated against forward return from t.
- The historical test period has already been inspected during development.
- This part searches for factor candidates. Separate result validation belongs to the next part.

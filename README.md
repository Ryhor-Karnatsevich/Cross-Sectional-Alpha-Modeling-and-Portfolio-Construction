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
    - **pipeline.py**
    - config.py
    - data.py
    - data_quality.py
    - get_tickers.py
    - risk_free_rate.py
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
- The dataset ends on the latest repository snapshot instead of assuming an unknown index composition after that date.


IMPORTANT:
- All future features must be computed using data up to t-1
- daily returns represent t-1 -> t
- saved forward returns represent t -> t+21 trading days


### **config.py**:
- Contains paths to parquet and csv files. Also contains setup parameters for data preparing.
- Uses for saving and deleting.
- DATA_START_DATE = 2008-01-01
- SUSPICIOUS_ABS_DAILY_RETURN = 0.5
- MAX_ABS_DAILY_RETURN = 1.0
- ROUND_TRIP_RETURN_TOLERANCE = 0.25
- CONFIRMED_REAL_RETURN_EVENTS contains manually verified extreme market moves
- YAHOO_REUSED_TICKERS contains obsolete symbols whose Yahoo history belongs to another security


### **get_tickers.py**:
- Downloads point-in-time S&P500 component snapshots from:
- https://github.com/fja05680/sp500
- Uses `S&P 500 Historical Components & Changes (Updated).csv`.
- Validates dates, duplicates and the number of components in every snapshot.
- Extracts the union of all historical tickers since 2008.
- Can return the S&P500 component list for a requested date.
- Converts ticker symbols with a dot to yfinance format with a dash.
- Builds a daily membership matrix by carrying every snapshot forward until the next index change.
- Limits the final market dataset to the date of the latest confirmed membership snapshot.


### **data.py**:
- Calculate different metrics to create parquet and csv files for further factors analysis.


#### download_data
- Download data for all historical tickers since 2008 via yfinance in batches
- Download both raw close and adjusted close fields
- Retry tickers without prices individually after the batch download
- Use only explicit aliases for verified direct ticker changes
- Replace obsolete or reused Yahoo symbols with their explicit aliases
- Reject known reused Yahoo symbols when no verified continuous alias exists
- Report whether every ticker came from a batch, individual retry, alias or remained missing
- Merge batches into a unified panel
- Remove duplicated columns

#### get_price_matrix
- Extract adjusted close prices
- Sort by date and remove empty rows
- Keep missing prices as missing

#### get_volume_matrix
- Extract raw Yahoo volume
- Rescale volume by raw close / adjusted close, so adjusted price x stored volume equals raw close x raw volume
- Hide negative values (still exist)

#### compute_returns
- Compute daily returns from adjusted prices without clipping real market moves
- Exclude returns whose current or previous price failed data quality
- Aligned with prices

#### compute_liquidity
- Compute liquidity proxy:
  - adjusted price x compatible volume (= raw close x raw volume)
  - 20-day rolling mean
  - log(1 + x) transform

#### Other
- Create long prices dataset
- Compute forward returns
- Compute membership matrix for every trading date and ticker
- Flag daily returns with an absolute move of at least 50% for diagnostics
- Quarantine a ticker from the first unconfirmed move of at least 100% or isolated spike-reversal
- Keep manually verified real extreme-return events
- Apply data-quality checks to the complete downloaded price history
- Compute availability as price available AND membership AND data quality
- Store the Yahoo symbol and download method for every historical ticker in `universe.csv`
- Report whether a ticker has price observations during its actual membership period
- Never replace a suspicious return with an artificial capped value

#### Sanity check for prices and volume
- Checks:
  - index monotonicity
  - If there are at least 100 columns
  - price/volume alignment
  - Negative volume test
  - Duplicates test
  - Missing values

#### Universe check
- Drops days with less than 150 available S&P500 members. For prices and liquidity.

#### Gaps check
- Reveal gaps greater than 5 days. (technically 10 due to fill in previous part)

#### Saving
- Using **save_all** save files in directory.
- Raw files:
  - prices
  - volume
  - liquidity
  - historical component snapshots
  - universe report with membership dates, data coverage and inclusion status
- Processed files:
  - returns
  - forward returns
  - prices in long format
  - membership matrix
  - data-quality matrix
  - availability matrix

#### Combining all together
- Additional calculations:
  - Aligning volume based on prices
  - Keep every historical ticker instead of removing it using full-period statistics
  - Keep full-period coverage only as a diagnostic in the universe report
  - Apply data-quality decisions using only information known by each date

#### Pipeline logic
- If data exist then just return it.
- If data isn't complete or missing then execute building it.


### **delete.py**:
- Deletes all generated Data System files, including the risk-free rate.
- It could be useful to clean space for further data updating.


### **pipeline.py**
- Executes whole code - data importing / rebuilding.


### **risk_free_rate.py**
- Downloads the official daily DGS3MO three-month US Treasury yield from FRED.
- Saves it separately to `Data/Raw/dgs3mo.parquet`.
- The general pipeline downloads it only when the file is missing.
- Future macro input for result validation. It is not one of the 11 core equity datasets and is not used in the current Factor Layer.


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

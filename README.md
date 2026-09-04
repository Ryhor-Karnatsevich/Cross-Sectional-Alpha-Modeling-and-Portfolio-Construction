# WORK IN PROGRESS
Status: This project is currently under active development. Some features might not work as expected.

Start date: 16.04.2026

**Project Roadmap*

| stage                     | status            | 
|---------------------------|-------------------|
| Data System               | **in progress**   | 
| Factor Layer              | **research complete** |
| Alpha Engine              | not started: no validated alpha yet |
| Signal -> Expectation     |                   |
| Risk Model                |                   |
| Portfolio Construction    |                   |
| Portfolio Risk Control    |                   |
| Backtest Engine           |                   |
| Benchmark & Evaluation    |                   |
| Analysis & Interpretation |                   |

### Project Structure
src/


  - Data_System/
    - _init_.py 
    - **pipeline.py**
    - config.py
    - data.py
    - get_tickers.py
    - delete.py


  - Factors_Layer/
    - **pipeline.py**
    - factors.py
    - transforms.py
    - research.py
    - statistical_research.py
    - candidate_research.py
    - factor_independence.py
    - quantile_research.py
    - walk_forward.py
    - regime_research.py


  - Alpha/
    - Not started yet


  - Pipeline/
    - run.py (empty)
    


## Data [1]

During that stage the data has been downloaded and initially cleaned and prepared. 

At the end there are 10 datasets with different metrics and formats:
- Five **"Processed"** files.
- Five **"Raw"** files.


**Limitations**:
- Historical membership is taken from a community-maintained GitHub repository, not from official S&P data.
- yfinance may have missing or incomplete history for delisted stocks and old ticker symbols. Therefore survivorship bias is reduced but not fully removed.
- The dataset ends on the latest repository snapshot instead of assuming an unknown index composition after that date.


IMPORTANT:
- All future features must be computed using data up to t-1
- daily returns represent t-1 → t
- forward returns represent t → t+21 trading days


### **config.py**:
- Contains paths to parquet and csv files. Also contains setup parameters for data preparing. 
- Uses for saving and deleting.
- START_DATE = 2010-01-01
- MIN_COVERAGE = 0.8
- MAX_ABS_DAILY_RETURN = 1.0
- MAX_EXTREME_DAILY_RETURNS = 1


### **get_tickers.py**:
- Downloads point-in-time S&P500 component snapshots from:
- https://github.com/fja05680/sp500
- Uses `S&P 500 Historical Components & Changes (Updated).csv`.
- Validates dates, duplicates and the number of components in every snapshot.
- Extracts the union of all historical tickers since 2010.
- Can return the S&P500 component list for a requested date.
- Converts ticker symbols with a dot to yfinance format with a dash.
- Builds a daily membership matrix by carrying every snapshot forward until the next index change.
- Limits the final market dataset to the date of the latest confirmed membership snapshot.


### **data.py**:
- Calculate different metrics to create 7 parquet and 1 csv files for further factors analysis. 


####  download_data
- Download data for all historical tickers since 2010 via yfinance in batches
- Merge batches into a unified panel
- Remove duplicated columns

####  get_price_matrix
- Extract adjusted close prices
- Sort by date and remove empty rows
- Forward-fill missing values (max 5 days)

####  get_volume_matrix
- Extract volume data
- Hide negative values (still exist)

####  compute_returns
- Compute daily returns (with clipping [-50%, +50%])
- Aligned with prices

#### compute_liquidity
- Compute liquidity proxy:
  - price × volume
  - 20-day rolling mean
  - log(1 + x) transform

#### Other
- Create long prices dataset
- Compute forward returns with 21 days shift
- Compute membership matrix for every trading date and ticker
- Compute availability as price available AND stock is an S&P500 member on that date
- Detect repeated daily price jumps above 100% during index membership as probable ticker reuse or broken Yahoo history

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
  - availability matrix

#### Combining all together
- Additional calculations:
  - Aligning volume based on prices
  - Remove assets with price coverage <80% during their membership period
  - Remove assets with more than one daily price jump above 100% during membership

#### Pipeline logic
- If data exist then just return it.
- If data isn't complete or missing then execute building it. 


### **delete.py**:
- Runs a process of deleting 7 parquet and 1 csv files.
- It could be useful to clean space for further data updating.


### **pipeline.py**
- Executes whole code - data importing / rebuilding.








## Factor Layer

The goal of that stage is to bults four factors that will be used in alpha creating.

### Factors:
**Momentum**
- 252 days window (1 Year)
- With minimum 200 observations
- 21 days skip (1 Month)
- Winsorized and Normalized


**Low Volatility**
- 60 days window
- Lower volatility is better, so factor sign is negative.
- Winsorized and Normalized


**Trend**
- Price / SMA(50) - 1
- Measures how far price is from 50 days moving average.
- Winsorized and Normalized


### transforms.py
- Winsorize values cross-sectionally for every date.
- Default limits are 1% and 99%.
- Normalize factor values with cross-sectional z-score.
- After normalization factor mean is near 0 and standard deviation is near 1 for every date.


### Factor pipeline.py
- Loads returns, prices, availability and forward returns datasets.
- Builds Momentum, Low Volatility and Trend factors.
- Uses availability mask before factor transformation.
- Calculates IC for every 21 trading days.
- IC is Spearman correlation between factor score known at t-1 and forward return from t to t+21.
- Requires the stock to be an index member on the IC evaluation date.
- Dates with less than 30 valid assets are excluded from IC calculations.


### Factor research.py
- Uses the same sensitivity variants for train, validation, test and rolling IC analysis.
- Tests 30 Momentum, Low Volatility and Trend specifications.
- Uses one minimum data coverage rule: 80% of every effective factor window.
- Uses 2010-2017 for train, 2018-2021 for validation and 2022-2026 for test.
- Selection score uses train and validation only. Test results are not used for parameter selection.
- Calculates full rolling 36-observation Mean IC history for every specification.
- Checks equal-weight combinations of the selected factor specifications.


### Factor statistical_research.py
- Tests the three documented baseline factors without selecting new parameters.
- Calculates daily cross-sectional Spearman IC for 5, 21, 63 and 126 trading-day forward returns.
- Uses the factor signal from t-1 and index membership on evaluation date t.
- Reports Newey-West (HAC) t-statistics with horizon - 1 lags for overlapping forward returns.
- Reports 95% circular block-bootstrap confidence intervals for Mean IC.
- Tests all 21 possible monthly rebalance offsets instead of relying on one arbitrary start date.
- Calculates rolling 252-trading-day Mean IC with at least 126 observations.
- Applies Holm and Benjamini-Hochberg corrections to the 12 factor-horizon tests inside every period.
- Saves detailed tables, run configuration and charts to `Data/Factor_Research/statistical_stage`.


### Factor candidate_research.py
- Adds 8 economically motivated price and volume factor families with 26 predefined specifications.
- Tests Short-Term Reversal, Residual Momentum, Volatility-Scaled Momentum, High Proximity, Trend Slope, Risk-Adjusted Trend, Liquidity Change and Price-Volume Confirmation.
- Evaluates every specification on 5, 21, 63 and 126 trading-day forward returns with the same daily IC and robust statistics used in `statistical_research.py`.
- Applies Holm and Benjamini-Hochberg corrections to all 104 candidate-horizon tests inside every period.
- Selects one representative per family using train and validation only. Test does not enter the selection score.
- Saves only the 8 selected factor matrices as a local cache for the next research stage.
- Saves tables, run configuration and a Mean IC heatmap to `Data/Factor_Research/candidate_stage`.


### Factor factor_independence.py
- Compares the selected candidates with the three baseline factors.
- Calculates daily cross-sectional Spearman correlation between factor scores.
- Calculates correlation between daily 21-day IC series using one common horizon.
- Residualizes every selected candidate against baseline Momentum, Low Volatility and Trend using same-date cross-sectional OLS.
- Checks separately whether baseline Trend adds information after removing its Momentum exposure.
- Tests Low Volatility both as a return alpha and as a descriptor of future realized volatility.
- Applies Holm and Benjamini-Hochberg corrections to residual and Low Vol role tests.
- Saves detailed correlation, residual IC and Low Vol role results to `Data/Factor_Research/independence_stage`.


### Factor quantile_research.py
- Builds equal-weight Q1-Q5 portfolios using factor information from t-1 and membership at t.
- Uses non-overlapping holding periods equal to the selected factor horizon.
- Calculates Q5 - Q1 spread, quantile monotonicity, turnover, gross return and net return under 0, 5, 10 and 25 bps transaction-cost assumptions.
- Tests every unique calendar phase up to 21 offsets.
- Does not remove a stock using future availability. A missing endpoint uses the last observable price inside the completed holding period and is reported separately.
- This is a Factor Layer diagnostic, not the final Portfolio Construction engine.


### Factor walk_forward.py
- Combines the original 30 sensitivity variants and 26 new candidates into 56 factor specifications and 224 factor-horizon hypotheses.
- Uses a rolling 5-year research window followed by the next out-of-sample calendar year, starting in 2015.
- Purges the final h trading days from every training sample because their h-day forward return was not known at the selection date.
- Re-selects parameters separately for every factor family and every OOS year without using future results.
- Saves stitched OOS IC and annual Q5 - Q1 portfolio returns after 10 bps transaction costs.
- Closes the final position at every annual model reset and includes liquidation costs.
- Saves reusable factor and daily IC caches to `Data/Factor_Research/walk_forward_stage`.


### Factor regime_research.py
- Defines bull/bear, high/low market volatility and broad/narrow market breadth using only information available at t-1.
- Uses expanding historical medians for volatility and breadth thresholds.
- Tests regime differences with chronological HAC regressions.
- Reports both fixed-factor results and purged walk-forward OOS results.
- Applies Holm and Benjamini-Hochberg corrections to regime comparisons.
- Regimes are diagnostics and are not used to retrofit factor parameters or trading rules.


#### IC statistics
- Mean IC shows average factor predictive power.
- Std IC shows how unstable IC is over time.
- T-stat shows if average IC is statistically different from zero.
- IC > 0 shows how often factor has positive predictive power.
- IC autocorrelation shows if factor IC is persistent between periods.


IMPORTANT:
- Factor matrices are calculated in memory and not saved to separate files yet.
- Factor signal uses information available at t-1 and is evaluated against forward return from t.
- Current Factor Layer has three implemented factors. Size factor needs historical market cap data.
- The historical test period has already been inspected during development. It is a development backtest, not a pristine final holdout for newly proposed factors.
- The completed price and volume research has not produced a validated return alpha. Trend Slope remains a research watchlist candidate only.
- Historical volatility is validated as a descriptor of future realized volatility and belongs in the future Risk Model rather than the Alpha Engine.
- The final research decision is saved locally in `Data/Factor_Research/FINAL_FACTOR_RESEARCH.md`.



# WORK IN PROGRESS
Status: This project is currently under active development. Some features might not work as expected.

Start date: 16.04.2026

**Project Roadmap**:

| stage                     | status            | 
|---------------------------|-------------------|
| Data System               | **done**          | 
| Factor Layer              | <- here right now |
| Alpha Engine              |                   |
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


  - Alpha/
    - Not started yet


  - Pipeline/
    - run.py (empty)
    


## Data [1]

During that stage the data has been downloaded and initially cleaned and prepared. 

At the end there are 8 datasets with different metrics and formats:
- Four **"Processed"** files.
- Four **"Raw"** files.


**Limitations**:
- Survivors bias present: list of stocks is not dynamic but static, so there are only stocks that been successful during last history period.


IMPORTANT:
- All future features must be computed using data up to t-1
- daily returns represent t-1 → t
- forward returns represent t → t+21 trading days


### **config.py**:
- Contains paths to parquet and csv files. Also contains setup parameters for data preparing. 
- Uses for saving and deleting.
- START_DATE = 2010-01-01
- MIN_COVERAGE = 0.8


### **get_tickers.py**:
- Uses link to wikipedia to extract current S&P500 list of ticker names.
- https://en.wikipedia.org/wiki/List_of_S%26P_500_companies


### **data.py**:
- Calculate different metrics to create 7 parquet and 1 csv files for further factors analysis. 


####  download_data
- Download data via yfinance in batches
- Merge batches into a unified panel
- Remove duplicated columns

####  get_price_matrix
- Extract adjusted close prices
- Sort by date and remove empty rows
- Forward-fill missing values (max 5 days)
- Remove dates with <50% cross-sectional coverage

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
- Compute availability for prices

#### Sanity check for prices and volume
- Checks:
  - index monotonicity
  - If there are at least 100 columns
  - price/volume alignment
  - Negative volume test
  - Duplicates test
  - Missing values

#### Universe check
- Drops days with less than 150 records. For prices and liquidity.

#### Gaps check
- Reveal gaps greater than 5 days. (technically 10 due to fill in previous part)

#### Saving
- Using **save_all** save files in directory.

#### Combining all together
- Additional calculations:
  - Aligning volume based on prices
  - Remove assets with data coverage <80%

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
- Dates with less than 30 valid assets are excluded from IC calculations.


### Factor research.py
- Uses the same sensitivity variants for train, validation, test and rolling IC analysis.
- Tests 30 Momentum, Low Volatility and Trend specifications.
- Uses one minimum data coverage rule: 80% of every effective factor window.
- Uses 2010-2017 for train, 2018-2021 for validation and 2022-2026 for test.
- Selection score uses train and validation only. Test results are not used for parameter selection.
- Calculates full rolling 36-observation Mean IC history for every specification.
- Checks equal-weight combinations of the selected factor specifications.


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



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
    - **pipeline.py**
    


## Data [1]

### **config.py**:
- Contains paths to parquet and csv files. Also contains setup parameters for data preparing. 
- Uses for saving and deleting.


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
  - Remove assets with data coverage <70%


#### Pipeline logic
- If data exist then just return it.
- If data isn't complete or missing then execute building it. 


### **delete.py**:
- Runs a process of deleting 7 parquet and 1 scv files.
- It could be useful to clean space for further data updating.



IMPORTANT:
- All future features must be computed using data up to t-1
- returns represent t → t+1

### limitations
survivors bias present.
This project uses a static S&P500 universe and therefore suffers from survivorship bias.





## Factor Layer

The goal of that stage is to bults four factors that will be used in alpha creating.

### Factors:
**Momentum**
- 252 days window (1 Year)
- With minimum 200 observations
- 21 days skip (1 Month)
- Winsorized and Normalized



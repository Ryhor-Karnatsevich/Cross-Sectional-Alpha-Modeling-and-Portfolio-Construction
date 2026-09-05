# WORK IN PROGRESS
Status: This project is currently under active development. Some features might not work as expected.


**Project Roadmap*

| stage                     | status                | 
|---------------------------|-----------------------|
| Data System               | **completed**         | 
| Factor Layer              | **research complete** |


### Project Structure
src/


  - Data_System/
    - _init_.py 
    - **pipeline.py**
    - config.py
    - data.py
    - get_tickers.py
    - risk_free_rate.py
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
    - composite_alpha_research.py
    - market_opportunity_research.py
    - portfolio_implementation_research.py
    - trend_slope_conditional_research.py


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


### **risk_free_rate.py**:
- Downloads the official daily `DGS3MO` three-month US Treasury yield from FRED.
- Saves one additional on-demand macro file to `Data/Raw/dgs3mo.parquet`.
- Validates dates, duplicates, numeric values and data coverage.
- This file is used only by Market Opportunity Research and is not one of the 10 core equity datasets.


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


### Factor composite_alpha_research.py
- Tests whether several weak but economically different signals become useful when combined.
- Combines Trend Slope, Short-Term Reversal and Liquidity Change on one 21-day horizon.
- Selects every component specification separately for every OOS year using only the previous 5 years.
- Purges the final 21 trading days from every selection sample.
- Tests equal weights, residualized factors, shrunk historical IC weights and inverse-volatility scaling.
- A component that does not pass the training eligibility rule cannot receive an increased IC weight.
- Tests all 21 calendar rebalance phases and 10/25 bps transaction costs.
- Treats 2015-2026 as one stitched walk-forward development backtest and applies Holm and Benjamini-Hochberg corrections.
- Saves selections, weights, OOS IC, annual phase returns, statistics and a chart to `Data/Factor_Research/composite_stage`.


### Factor market_opportunity_research.py
- Tests if the fixed `raw_equal` composite works better in a favorable environment for cross-sectional alpha.
- Uses the DGS3MO risk-free rate, median single-stock volatility, cross-sectional return dispersion and an average stock-correlation proxy.
- Every market indicator is lagged by one trading day.
- Builds volatility, dispersion and correlation only from point-in-time available S&P500 members.
- Selects the composite components and market thresholds separately for every OOS year using only the previous 5 years.
- Purges the final 21 trading days from every threshold training sample.
- Uses a fixed 2% risk-free-rate threshold. Other thresholds come only from the historical training window.
- Compares an always-active portfolio with binary and 0/50/100% tiered exposure rules.
- Tests all 21 calendar phases and exact exposure-adjusted turnover under 0, 10 and 25 bps costs.
- Does not include cash carry. The rate is tested as a market condition, not added as free alpha.
- Saves indicators, annual thresholds, OOS states, conditional IC, regression results, portfolio results and charts to `Data/Factor_Research/market_opportunity_stage`.


#### Market Opportunity result
- Low correlation is the only individually useful condition. Conditional Mean IC difference is 0.0410 with FDR p-value 0.0467.
- In the multivariate OOS regression the standardized correlation coefficient is -0.0250 with HAC t-stat -2.92 and FDR p-value 0.0140.
- The binary rule is active around 33% of monthly holding periods.
- It reduces median dollar turnover from 2.39 to 1.01 per rebalance.
- Its median annual Q5-Q1 result after 10 bps costs is +1.55% with median Sharpe 0.29. The always-active result is -0.75% with median Sharpe -0.07.
- The binary rule is positive in 20 of 21 calendar phases, but the worst phase is still slightly negative.
- Portfolio returns and improvement versus the always-active strategy are not statistically significant after multiple-testing correction. This is conditional research evidence, not validated alpha.


### Factor portfolio_implementation_research.py
- Keeps the `raw_equal` composite and binary Market Opportunity rule frozen. It does not select a new signal or a more profitable market rule.
- Diagnoses annual long and short contributions, component IC in low-correlation periods, profit concentration and entry/exit effects.
- Tests 21, 42 and 63 trading-day holding periods across all 21 monthly calendar phases.
- Compares equal-weight tails, 20/30% position buffers, beta-neutral legs and a risk-controlled version.
- Estimates beta from a trailing 252-day window with at least 126 observations. The market proxy uses point-in-time S&P500 membership and all estimates are lagged by one day.
- The risk-controlled version targets 10% annualized spread volatility but does not lever above 2.0 gross exposure.
- Limits every individual position to 2% and total gross exposure to 2.0.
- Tests fixed 10/25 bps costs and a liquidity-dependent cost model using lagged 20-day average dollar volume and a reference AUM of $10 million.
- Saves the complete grid, annual decomposition, transition diagnostics, decision criteria and charts to `Data/Factor_Research/portfolio_implementation_stage`.


#### Portfolio implementation result
- The plain conditional 21-day portfolio has +2.09% median annual return after modeled liquidity costs, median Sharpe 0.42 and positive mean return in all 21 calendar phases.
- The 20/30% buffer reduces median turnover from 1.01 to 0.94 but also slightly reduces median annual return to +1.96%.
- Beta neutralization reduces estimated portfolio beta to approximately zero. It does not improve the 21-day result.
- The primary risk-controlled 21-day version reduces turnover to 0.76 and median gross exposure to 0.53. Its median annual net return is +1.26% and median Sharpe is 0.40.
- Only 45.5% of its 11 complete OOS calendar years are positive. The largest positive year supplies 49.6% of total positive profit.
- Trend Slope is the only composite component with a significant low-correlation IC improvement after FDR correction: difference 0.0666, HAC t-stat 2.69 and FDR p-value 0.0212.
- No portfolio implementation survives portfolio-level FDR correction. The minimum adjusted p-value is 0.3153.
- The stop/go decision is `research_candidate_only`: the result is suitable for untouched forward or paper validation, but it is not validated alpha.


#### Portfolio implementation limitations
- This is still the already inspected development sample. The 21 calendar phases overlap and are robustness checks, not 21 independent observations.
- The liquidity model is stylized. It does not model short borrow fees, borrow availability, taxes or broker-specific execution.
- Beta neutrality uses a rolling single-factor estimate, not a full covariance risk model or sector neutrality.
- Volatility targeting only reduces exposure and does not increase leverage above the baseline.
- The 42 and 63-day holding periods are sensitivity checks and must not be selected only because their historical result is better.


### Factor trend_slope_conditional_research.py
- Runs the final bounded research extension around the already discovered Trend Slope and low-correlation relationship.
- Selects one 21-day Trend Slope specification for every OOS year using only the previous 5 purged years.
- Compares four frozen market rules: always active, low correlation, low correlation with high dispersion and the existing binary Market Opportunity rule.
- Uses seven declared comparisons instead of a full Cartesian parameter search.
- Compares equal quintiles, controlled quintiles and controlled deciles. The controlled portfolios use a 20/30% or 10/20% entry/exit buffer, beta neutrality, 10% volatility targeting, 2% position caps and maximum 2.0 gross exposure.
- Uses 21 days as the primary holding period and 42 days only as a sensitivity check.
- Tests all 21 monthly calendar phases, fixed 10/25 bps costs and the lagged liquidity-dependent cost model.
- Saves selections, conditional IC, annual portfolios, diagnostics, decision criteria and charts to `Data/Factor_Research/trend_slope_conditional_stage`.


#### Conditional Trend Slope result
- The complete stitched Trend Slope Mean IC is 0.0110 and is not significant.
- During low-correlation periods Mean IC increases to 0.0386 with HAC t-stat 2.58 and FDR p-value 0.0196. Outside the regime Mean IC is -0.0280. The difference is 0.0666 with FDR p-value 0.0212.
- The primary low-correlation controlled quintile is active in 56.8% of monthly periods. Median annual return after modeled costs is +2.97%, median Sharpe is 0.51 and all 21 calendar phases have positive mean returns. The worst phase is +1.46%.
- The always-active version returns +4.14% with median Sharpe 0.52. Low-correlation filtering keeps a similar Sharpe with less exposure, but does not improve total return. The paired improvement FDR p-value is approximately 0.50.
- Return per unit of average gross exposure increases from 0.029 for always active to 0.036 for low correlation. This is descriptive capital efficiency, not independent alpha evidence.
- Adding high dispersion makes the result worse: +0.75% median annual return and median Sharpe 0.16.
- Controlled deciles do not improve controlled quintiles. The 42-day result is positive but remains a sensitivity observation and is not selected after seeing the output.
- Six of 11 complete OOS years are positive. The largest positive year supplies 29.0% of total positive profit.
- Only 3 of 11 annual Trend Slope selections pass the training rule requiring positive Mean IC in both halves of the training window.
- No portfolio return survives FDR correction. The minimum adjusted p-value is 0.3212.
- Final decision is `research_watchlist`. This is a useful narrow case study and a candidate for untouched paper validation, not validated alpha.


#### Conditional Trend Slope limitations
- Trend Slope and low correlation were identified after the development history had already been inspected.
- Calendar phases overlap and are robustness checks, not independent observations.
- The low-correlation filter improves conditional IC, but it does not statistically outperform the always-active Trend Slope portfolio.
- The test uses a stylized execution model and does not include borrow availability or borrow fees.
- The better historical 42-day sensitivity result must not be promoted to the primary rule without new untouched evidence.


#### IC statistics
- Mean IC shows average factor predictive power.
- Std IC shows how unstable IC is over time.
- T-stat shows if average IC is statistically different from zero.
- IC > 0 shows how often factor has positive predictive power.
- IC autocorrelation shows if factor IC is persistent between periods.


IMPORTANT:
- The three baseline factor matrices from `pipeline.py` are calculated in memory and are not saved as production datasets. The 56 research specifications are saved only as a local reusable cache in `Data/Factor_Research/walk_forward_stage/factor_cache`.
- Factor signal uses information available at t-1 and is evaluated against forward return from t.
- Current Factor Layer has three implemented factors. Size factor needs historical market cap data.
- The historical test period has already been inspected during development. It is a development backtest, not a pristine final holdout for newly proposed factors.
- The completed price and volume research has not produced a validated return alpha. Trend Slope remains a research watchlist candidate only.
- Combining Trend Slope, Short-Term Reversal and Liquidity Change does not produce a validated alpha either. The best equal-weight composite has Mean OOS IC 0.0101, but HAC t-stat 1.05 and FDR p-value 0.53.
- After 10 bps costs the best composite has median annual Q5-Q1 return -0.75% across 21 calendar phases. More complex residual, IC-weighted and inverse-volatility versions do not improve it.
- Market Opportunity conditioning improves the best composite economically, mainly in low-correlation periods, but the portfolio improvement is not statistically validated.
- Historical volatility is validated as a descriptor of future realized volatility and belongs in the future Risk Model rather than the Alpha Engine.
- The final research decision is saved locally in `Data/Factor_Research/FINAL_FACTOR_RESEARCH.md`.



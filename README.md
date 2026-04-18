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
    


### Data [1]

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



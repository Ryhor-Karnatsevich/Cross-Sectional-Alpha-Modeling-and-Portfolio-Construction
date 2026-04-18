# WORK IN PROGRESS
Status: This project is currently under active development. Some features might not work as expected.

Start date: 16.04.2026

**Project Roadmap**:
- Data System             **<- here right now**
- Factor Layer 
- Alpha Engine
- Signal -> Expectation
- Risk Model 
- Portfolio Construction 
- Portfolio Risk Control 
- Backtest Engine 
- Benchmark & Evaluation 
- Analysis & Interpretation



### Project Structure
- src/
  - Data
    - _init_.py 
    - **pipeline.py**
    - config.py
    - data.py
    - get_tickers.py
    - delete.py
  - Factors
    - **pipeline.py**


### Data [1]

IMPORTANT:
- All future features must be computed using data up to t-1
- returns represent t → t+1

### limitations
survivors bias present.


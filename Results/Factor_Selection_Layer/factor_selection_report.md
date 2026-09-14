# Factor Selection Report

## Scope

- Factor configurations analyzed: `56`.
- Factor-horizon hypotheses analyzed: `448`.
- Effect tests analyzed: `2240`.
- Full 448-hypothesis scope: `yes`.

## Pattern Classification

- `no_stable_structure`: 426 hypotheses.
- `lower_tail`: 14 hypotheses.
- `upper_tail`: 7 hypotheses.
- `positive_monotonic`: 1 hypotheses.

## Evidence Status

- `no_stable_structure`: 426 hypotheses.
- `rejected_by_multiple_testing`: 12 hypotheses.
- `time_unstable_pattern`: 10 hypotheses.

## Economic Pattern Cards

The table shows up to 50 detected patterns. Complete results remain in `hypothesis_cards.csv` and `effect_tests.csv`.

| Hypothesis | Pattern | Strongest economic effect | Q10-Q1 mean / t | Q10-middle mean / t | Middle-Q1 mean / t | Edges-middle mean / t | Status |
| --- | --- | --- | --- | --- | --- | --- | --- |
| short_term_reversal\|reversal_5d / h5 | upper_tail | q10_minus_middle | 0.0014 / 1.94 | 0.0011 / 2.30 | 0.0003 / 0.73 | 0.0004 / 1.61 | rejected_by_multiple_testing |
| liquidity_change\|liq_20d_252d / h252 | upper_tail | q10_minus_middle | 0.0002 / 0.02 | 0.0167 / 2.40 | -0.0164 / -1.11 | 0.0166 / 1.80 | rejected_by_multiple_testing |
| liquidity_change\|liq_20d_126d / h252 | upper_tail | q10_minus_middle | 0.0037 / 0.45 | 0.0139 / 2.41 | -0.0102 / -1.01 | 0.0121 / 1.70 | rejected_by_multiple_testing |
| low_volatility\|60d / h63 | lower_tail | edges_minus_middle | -0.0183 / -1.60 | -0.0014 / -0.26 | -0.0170 / -2.24 | 0.0078 / 2.55 | rejected_by_multiple_testing |
| low_volatility\|90d / h63 | lower_tail | edges_minus_middle | -0.0185 / -1.57 | -0.0017 / -0.32 | -0.0168 / -2.14 | 0.0075 / 2.39 | rejected_by_multiple_testing |
| trend\|SMA20 / h1 | lower_tail | middle_minus_q1 | -0.0004 / -1.78 | -0.0001 / -0.76 | -0.0003 / -2.06 | 0.0001 / 1.37 | rejected_by_multiple_testing |
| low_volatility\|90d / h42 | lower_tail | edges_minus_middle | -0.0123 / -1.53 | -0.0010 / -0.29 | -0.0113 / -2.13 | 0.0051 / 2.47 | rejected_by_multiple_testing |
| low_volatility\|60d / h42 | lower_tail | edges_minus_middle | -0.0122 / -1.56 | -0.0010 / -0.29 | -0.0111 / -2.16 | 0.0051 / 2.48 | rejected_by_multiple_testing |
| short_term_reversal\|reversal_5d / h1 | positive_monotonic | q10_minus_middle | 0.0004 / 2.03 | 0.0003 / 2.32 | 0.0001 / 0.90 | 0.0001 / 1.40 | rejected_by_multiple_testing |
| liquidity_change\|liq_20d_126d / h126 | upper_tail | edges_minus_middle | 0.0056 / 0.91 | 0.0103 / 2.04 | -0.0047 / -1.14 | 0.0075 / 2.17 | rejected_by_multiple_testing |
| low_volatility\|40d / h63 | lower_tail | edges_minus_middle | -0.0185 / -1.69 | -0.0025 / -0.50 | -0.0160 / -2.20 | 0.0068 / 2.31 | time_unstable_pattern |
| low_volatility\|180d / h63 | lower_tail | edges_minus_middle | -0.0208 / -1.70 | -0.0033 / -0.61 | -0.0176 / -2.14 | 0.0071 / 2.18 | time_unstable_pattern |
| short_term_reversal\|reversal_5d / h21 | upper_tail | q10_minus_middle | 0.0027 / 1.89 | 0.0023 / 1.97 | 0.0004 / 0.36 | 0.0010 / 1.15 | rejected_by_multiple_testing |
| low_volatility\|180d / h42 | lower_tail | edges_minus_middle | -0.0141 / -1.66 | -0.0021 / -0.56 | -0.0120 / -2.13 | 0.0049 / 2.25 | time_unstable_pattern |
| low_volatility\|90d / h5 | lower_tail | edges_minus_middle | -0.0016 / -1.49 | -0.0001 / -0.31 | -0.0014 / -2.04 | 0.0007 / 2.39 | time_unstable_pattern |
| residual_momentum\|resmom_9m_1m / h63 | upper_tail | q10_minus_middle | 0.0097 / 1.13 | 0.0093 / 2.12 | 0.0004 / 0.07 | 0.0044 / 1.21 | time_unstable_pattern |
| low_volatility\|120d / h63 | lower_tail | middle_minus_q1 | -0.0190 / -1.59 | -0.0034 / -0.64 | -0.0155 / -1.98 | 0.0061 / 1.96 | rejected_by_multiple_testing |
| liquidity_change\|liq_20d_252d / h126 | upper_tail | q10_minus_middle | 0.0083 / 0.98 | 0.0129 / 2.00 | -0.0045 / -0.78 | 0.0087 / 1.98 | time_unstable_pattern |
| low_volatility\|180d / h5 | lower_tail | middle_minus_q1 | -0.0018 / -1.64 | -0.0003 / -0.69 | -0.0015 / -2.02 | 0.0006 / 2.01 | time_unstable_pattern |
| low_volatility\|180d / h10 | lower_tail | middle_minus_q1 | -0.0033 / -1.63 | -0.0007 / -0.71 | -0.0026 / -1.98 | 0.0010 / 1.82 | time_unstable_pattern |
| low_volatility\|20d / h5 | lower_tail | middle_minus_q1 | -0.0017 / -1.78 | -0.0004 / -1.07 | -0.0013 / -1.96 | 0.0004 / 1.63 | time_unstable_pattern |
| low_volatility\|20d / h63 | lower_tail | middle_minus_q1 | -0.0172 / -1.68 | -0.0036 / -0.79 | -0.0136 / -2.02 | 0.0050 / 1.91 | time_unstable_pattern |

## IC and Stability Diagnostics

IC is reported as supporting information. It does not create a candidate when no economic pattern is detected.

| Horizon | Pattern | Direction | Mean IC | IC HAC t | Primary effect | Effect mean | Effect HAC t | Monthly stability | Annual stability | Rank autocorr | Status |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| liquidity_change\|liq_20d_126d / h126 | upper_tail | q10_above_middle | -0.0015 | -0.27 | q10_minus_middle | 0.0103 | 2.04 | 0.61 | 0.71 | 0.06 | rejected_by_multiple_testing |
| liquidity_change\|liq_20d_126d / h252 | upper_tail | q10_above_middle | 0.0029 | 0.54 | q10_minus_middle | 0.0139 | 2.41 | 0.61 | 0.62 | 0.15 | rejected_by_multiple_testing |
| liquidity_change\|liq_20d_252d / h126 | upper_tail | q10_above_middle | 0.0048 | 0.57 | q10_minus_middle | 0.0129 | 2.00 | 0.59 | 0.59 | 0.12 | time_unstable_pattern |
| liquidity_change\|liq_20d_252d / h252 | upper_tail | q10_above_middle | 0.0080 | 0.80 | q10_minus_middle | 0.0167 | 2.40 | 0.63 | 0.69 | 0.08 | rejected_by_multiple_testing |
| low_volatility\|120d / h63 | lower_tail | q1_above_middle | -0.0144 | -0.57 | middle_minus_q1 | -0.0155 | -1.98 | 0.56 | 0.71 | 0.89 | rejected_by_multiple_testing |
| low_volatility\|180d / h5 | lower_tail | q1_above_middle | -0.0015 | -0.20 | middle_minus_q1 | -0.0015 | -2.02 | 0.49 | 0.65 | 1.00 | time_unstable_pattern |
| low_volatility\|180d / h10 | lower_tail | q1_above_middle | -0.0046 | -0.43 | middle_minus_q1 | -0.0026 | -1.98 | 0.52 | 0.65 | 0.99 | time_unstable_pattern |
| low_volatility\|180d / h42 | lower_tail | q1_above_middle | -0.0122 | -0.58 | middle_minus_q1 | -0.0120 | -2.13 | 0.53 | 0.65 | 0.96 | time_unstable_pattern |
| low_volatility\|180d / h63 | lower_tail | q1_above_middle | -0.0125 | -0.49 | middle_minus_q1 | -0.0176 | -2.14 | 0.55 | 0.65 | 0.94 | time_unstable_pattern |
| low_volatility\|20d / h5 | lower_tail | q1_above_middle | -0.0013 | -0.20 | middle_minus_q1 | -0.0013 | -1.96 | 0.51 | 0.65 | 0.88 | time_unstable_pattern |
| low_volatility\|20d / h63 | lower_tail | q1_above_middle | -0.0088 | -0.41 | middle_minus_q1 | -0.0136 | -2.02 | 0.54 | 0.53 | 0.59 | time_unstable_pattern |
| low_volatility\|40d / h63 | lower_tail | q1_above_middle | -0.0098 | -0.43 | middle_minus_q1 | -0.0160 | -2.20 | 0.55 | 0.59 | 0.68 | time_unstable_pattern |
| low_volatility\|60d / h42 | lower_tail | q1_above_middle | -0.0106 | -0.53 | middle_minus_q1 | -0.0111 | -2.16 | 0.56 | 0.65 | 0.81 | rejected_by_multiple_testing |
| low_volatility\|60d / h63 | lower_tail | q1_above_middle | -0.0119 | -0.49 | middle_minus_q1 | -0.0170 | -2.24 | 0.57 | 0.65 | 0.73 | rejected_by_multiple_testing |
| low_volatility\|90d / h5 | lower_tail | q1_above_middle | -0.0013 | -0.18 | middle_minus_q1 | -0.0014 | -2.04 | 0.49 | 0.76 | 0.99 | time_unstable_pattern |
| low_volatility\|90d / h42 | lower_tail | q1_above_middle | -0.0126 | -0.62 | middle_minus_q1 | -0.0113 | -2.13 | 0.55 | 0.71 | 0.89 | rejected_by_multiple_testing |
| low_volatility\|90d / h63 | lower_tail | q1_above_middle | -0.0137 | -0.56 | middle_minus_q1 | -0.0168 | -2.14 | 0.57 | 0.71 | 0.84 | rejected_by_multiple_testing |
| residual_momentum\|resmom_9m_1m / h63 | upper_tail | q10_above_middle | 0.0139 | 0.90 | q10_minus_middle | 0.0093 | 2.12 | 0.58 | 0.59 | 0.59 | time_unstable_pattern |
| short_term_reversal\|reversal_5d / h1 | positive_monotonic | positive | 0.0075 | 2.61 | q10_minus_q1 | 0.0004 | 2.03 | 0.57 | 0.76 | 0.76 | rejected_by_multiple_testing |
| short_term_reversal\|reversal_5d / h5 | upper_tail | q10_above_middle | 0.0126 | 2.86 | q10_minus_middle | 0.0011 | 2.30 | 0.61 | 0.88 | -0.01 | rejected_by_multiple_testing |
| short_term_reversal\|reversal_5d / h21 | upper_tail | q10_above_middle | 0.0097 | 1.89 | q10_minus_middle | 0.0023 | 1.97 | 0.58 | 0.76 | -0.00 | rejected_by_multiple_testing |
| trend\|SMA20 / h1 | lower_tail | q1_above_middle | -0.0063 | -2.12 | middle_minus_q1 | -0.0003 | -2.06 | 0.59 | 0.76 | 0.91 | rejected_by_multiple_testing |

## Figures

### Hypothesis Overview

![hypothesis overview](Figures/hypothesis_overview.png)

### Short Term Reversal  Reversal 5D  Evidence Dashboard

![short term reversal  reversal 5d  evidence dashboard](Figures/short_term_reversal__reversal_5d__evidence_dashboard.png)

### Short Term Reversal  Reversal 5D  Quantile Curves

![short term reversal  reversal 5d  quantile curves](Figures/short_term_reversal__reversal_5d__quantile_curves.png)

### Short Term Reversal  Reversal 5D  Monthly Ic

![short term reversal  reversal 5d  monthly ic](Figures/short_term_reversal__reversal_5d__monthly_ic.png)

### Liquidity Change  Liq 20D 252D  Evidence Dashboard

![liquidity change  liq 20d 252d  evidence dashboard](Figures/liquidity_change__liq_20d_252d__evidence_dashboard.png)

### Liquidity Change  Liq 20D 252D  Quantile Curves

![liquidity change  liq 20d 252d  quantile curves](Figures/liquidity_change__liq_20d_252d__quantile_curves.png)

### Liquidity Change  Liq 20D 252D  Monthly Ic

![liquidity change  liq 20d 252d  monthly ic](Figures/liquidity_change__liq_20d_252d__monthly_ic.png)

### Liquidity Change  Liq 20D 126D  Evidence Dashboard

![liquidity change  liq 20d 126d  evidence dashboard](Figures/liquidity_change__liq_20d_126d__evidence_dashboard.png)

### Liquidity Change  Liq 20D 126D  Quantile Curves

![liquidity change  liq 20d 126d  quantile curves](Figures/liquidity_change__liq_20d_126d__quantile_curves.png)

### Liquidity Change  Liq 20D 126D  Monthly Ic

![liquidity change  liq 20d 126d  monthly ic](Figures/liquidity_change__liq_20d_126d__monthly_ic.png)

### Low Volatility  60D  Evidence Dashboard

![low volatility  60d  evidence dashboard](Figures/low_volatility__60d__evidence_dashboard.png)

### Low Volatility  60D  Quantile Curves

![low volatility  60d  quantile curves](Figures/low_volatility__60d__quantile_curves.png)

### Low Volatility  60D  Monthly Ic

![low volatility  60d  monthly ic](Figures/low_volatility__60d__monthly_ic.png)

### Low Volatility  90D  Evidence Dashboard

![low volatility  90d  evidence dashboard](Figures/low_volatility__90d__evidence_dashboard.png)

### Low Volatility  90D  Quantile Curves

![low volatility  90d  quantile curves](Figures/low_volatility__90d__quantile_curves.png)

### Low Volatility  90D  Monthly Ic

![low volatility  90d  monthly ic](Figures/low_volatility__90d__monthly_ic.png)

### Trend  Sma20  Evidence Dashboard

![trend  SMA20  evidence dashboard](Figures/trend__SMA20__evidence_dashboard.png)

### Trend  Sma20  Quantile Curves

![trend  SMA20  quantile curves](Figures/trend__SMA20__quantile_curves.png)

### Trend  Sma20  Monthly Ic

![trend  SMA20  monthly ic](Figures/trend__SMA20__monthly_ic.png)

### Low Volatility  40D  Evidence Dashboard

![low volatility  40d  evidence dashboard](Figures/low_volatility__40d__evidence_dashboard.png)

### Low Volatility  40D  Quantile Curves

![low volatility  40d  quantile curves](Figures/low_volatility__40d__quantile_curves.png)

### Low Volatility  40D  Monthly Ic

![low volatility  40d  monthly ic](Figures/low_volatility__40d__monthly_ic.png)

### Low Volatility  180D  Evidence Dashboard

![low volatility  180d  evidence dashboard](Figures/low_volatility__180d__evidence_dashboard.png)

### Low Volatility  180D  Quantile Curves

![low volatility  180d  quantile curves](Figures/low_volatility__180d__quantile_curves.png)

### Low Volatility  180D  Monthly Ic

![low volatility  180d  monthly ic](Figures/low_volatility__180d__monthly_ic.png)

### Residual Momentum  Resmom 9M 1M  Evidence Dashboard

![residual momentum  resmom 9m 1m  evidence dashboard](Figures/residual_momentum__resmom_9m_1m__evidence_dashboard.png)

### Residual Momentum  Resmom 9M 1M  Quantile Curves

![residual momentum  resmom 9m 1m  quantile curves](Figures/residual_momentum__resmom_9m_1m__quantile_curves.png)

### Residual Momentum  Resmom 9M 1M  Monthly Ic

![residual momentum  resmom 9m 1m  monthly ic](Figures/residual_momentum__resmom_9m_1m__monthly_ic.png)

### Low Volatility  120D  Evidence Dashboard

![low volatility  120d  evidence dashboard](Figures/low_volatility__120d__evidence_dashboard.png)

### Low Volatility  120D  Quantile Curves

![low volatility  120d  quantile curves](Figures/low_volatility__120d__quantile_curves.png)

### Low Volatility  120D  Monthly Ic

![low volatility  120d  monthly ic](Figures/low_volatility__120d__monthly_ic.png)

## Interpretation Rules

- `positive_monotonic` and `negative_monotonic` describe ordered Q1-Q10 relationships.
- `upper_tail` and `lower_tail` describe an effect concentrated in one extreme quantile.
- `both_tails_vs_middle` describes a U-shaped or inverted-U relationship.
- `no_stable_structure` means that no supported shape was identified by the current descriptive rules.
- A provisional candidate is not a validated trading strategy.

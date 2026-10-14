Fix Bull over-detection / Chop under-detection in HMM regime model

- Features: momentum is now a vol-scaled trend t-stat (40-day lookback), vol is Wilder ATR% z-scored over 1 year, and Kaufman's efficiency ratio is added to detect chop; data history goes from 2y to 5y
- Model: transitions are stickier (0.95/0.95/0.92), Chop is redefined as no trend + normal vol + low ER (was high vol), and dates are now aligned with probabilities; forward filtering.
- Result: regimes last ~16–20 days on average, and sideways, rally and decline periods are labelled correctly on ITC, MARUTI and TCS

Refs:
[1] Hamilton (1989), "A New Approach to the Economic Analysis of Nonstationary Time Series and the Business Cycle", Econometrica 57(2)
[2] Rabiner (1989), "A Tutorial on Hidden Markov Models and Selected Applications in Speech Recognition", Proc. IEEE 77(2)
[3] Moskowitz, Ooi & Pedersen (2012), "Time Series Momentum", Journal of Financial Economics 104(2)
[4] Wilder (1978), New Concepts in Technical Trading Systems
[5] Kaufman (1995), Smarter Trading; see also Kaufman, Trading Systems and Methods
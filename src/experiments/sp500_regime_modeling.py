import pandas as pd
import numpy as np
import yfinance
import matplotlib.pyplot as plt
import matplotlib

from hmmlearn import hmm
from src.models.ContinuousHMM import ContinuousHMM

matplotlib.use('TkAgg')

sp500 = yfinance.download (tickers = "^GSPC", start = "1994-01-07",
                              end = "2019-09-01", interval = "1d")

print(sp500)

plt.figure()
plt.plot(sp500["Close"])
plt.title("S&P500 Close")

df = pd.DataFrame()

df["close"] = sp500["Close"]
df.fillna(method='bfill', inplace=True)

df["log_return"] = np.log(sp500["Close"] / sp500["Close"].shift(1))
df["log_return"] = (df["log_return"] - df["log_return"].mean()) / df["log_return"].std()

df["rolling_mean"] = df["log_return"].rolling(60).mean().dropna()
df["rolling_mean"] = (df["rolling_mean"] - df["rolling_mean"].mean()) / df["rolling_mean"].std()

df["rolling_volatility"] = df["log_return"].rolling(60).std().dropna()
df["rolling_volatility"] = (df["rolling_volatility"] - df["rolling_volatility"].mean()) / df["rolling_volatility"].std()

df["rolling_skewness"] = df["log_return"].rolling(60).skew().dropna()
df["rolling_skewness"] = (df["rolling_skewness"] - df["rolling_skewness"].mean()) / df["rolling_skewness"].std()

df["rolling_kurtosis"] = df["log_return"].rolling(60).kurt().dropna()
df["rolling_kurtosis"] = (df["rolling_kurtosis"] - df["rolling_kurtosis"].mean()) / df["rolling_kurtosis"].std()

rolling_max = sp500["Close"].cummax()
df["drawdown"] = (sp500["Close"]-rolling_max)/rolling_max
df["drawdown"] = (df["drawdown"] - df["drawdown"].mean()) / df["drawdown"].std()

print(df.describe())

fig, axes = plt.subplots(2, 2, figsize=(16, 12),
                         gridspec_kw={'wspace':0.3, 'hspace':0.4})

axes[0,0].plot(df["rolling_mean"])
axes[0,0].set_title("S&P500 Rolling Mean")

axes[0,1].plot(df["rolling_volatility"])
axes[0,1].set_title("S&P500 Rolling Volatility")

axes[1,0].plot(df["rolling_skewness"])
axes[1,0].set_title("S&P500 Rolling Skewness")

axes[1,1].plot(df["rolling_kurtosis"])
axes[1,1].set_title("S&P500 Rolling Kurtosis")

for ax in axes.flat:
    ax.set_xlabel("Date")
    ax.set_ylabel("Value")

plt.figure()
plt.plot(df["drawdown"])
plt.title("S&P500 Drawdown")

df = df.dropna()

num_regimes = 3
num_iters = 2500

model1 = ContinuousHMM(num_regimes)
model1.fit(df, num_iters)
model1_regimes = model1.decode(df)

print("From Scratch Implementation")
for i in range(num_regimes):
    print(f"Regime {i} closing values: mean = {df[model1_regimes == i].mean()["log_return"]}, std = {df[model1_regimes == i].std()["log_return"]}")

model2 = hmm.GaussianHMM(
    n_components=num_regimes,
    n_iter=num_iters,
    covariance_type="full",
    init_params="stmc"
)
model2.fit(df)
model2_regimes = model2.predict(df)

print("HMMLearn Library")
for i in range(model2.n_components):
    print(f"Regime {i} closing values: mean = {model2.means_[i, 1]}, std = {np.sqrt(model2.covars_[i, 1, 1])}")

plt.figure(figsize=(14,6))
plt.scatter(x=df.index, y=df["close"], c=model1_regimes, cmap='viridis')
plt.xlabel("Date")
plt.ylabel("Close Price")
plt.title("S&P500 Regimes (From Scratch Implementation)")
plt.colorbar(label="Regime")

plt.figure(figsize=(14,6))
plt.scatter(x=df.index, y=df["close"], c=model2_regimes, cmap='viridis')
plt.xlabel("Date")
plt.ylabel("Close Price")
plt.title("S&P500 Regimes (HMMLearn Library)")
plt.colorbar(label="Regime")
plt.show()






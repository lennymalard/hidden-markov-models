# Hidden Markov Models (HMM) & Market Regime Detection

This project features a completely from-scratch, numerically stable implementation of Discrete and Continuous Hidden Markov Models (HMM) using NumPy. It includes comprehensive unit testing and a practical financial experiment applying the Continuous HMM to detect market regimes in the S&P 500.

## 📌 Features

* **Discrete HMM (`DiscreteHMM.py`)**: Implements standard Forward-Backward algorithms, Baum-Welch training, and Viterbi decoding for discrete observation spaces. Utilizes local scaling to maintain numerical stability.
* **Continuous HMM (`ContinuousHMM.py`)**: Implements Gaussian-emission HMMs. Utilizes robust `logsumexp` calculations to handle underflow issues common in continuous probabilistic modeling.
* **Market Regime Experiment (`sp500_regime_modeling.py`)**: A financial modeling script that:
    * Fetches historical S&P 500 data using `yfinance`.
    * Computes rolling statistical features (volatility, skewness, kurtosis, drawdown).
    * Trains the custom `ContinuousHMM` to identify distinct market environments (e.g., bull, bear, high-volatility).
    * """Benchmarks""" the from-scratch implementation against the industry-standard `hmmlearn` library.
    * Visualizes the identified regimes using `matplotlib`.
* **Robust Unit Testing**: Verifies large-scale numerical stability, forward-backward consistency, normalization vectors, Viterbi deterministic cycles, and monotonic convergence of the Baum-Welch algorithm.

## 📂 Project Structure

```text
.
└── src/
    ├── experiments/
    │   └── sp500_regime_modeling.py   # S&P 500 regime detection script
    ├── models/
    │   ├── ContinuousHMM.py           # Continuous HMM implementation
    │   ├── continuous_hmm_test.py     # Unit tests for Continuous HMM
    │   ├── DiscreteHMM.py             # Discrete HMM implementation
    │   └── discrete_hmm_test.py       # Unit tests for Discrete HMM
    └── utils/
        └── math.py                    # Log-Sum-Exp implementation for stability
```

## 🛠️ Prerequisites

To run this project, you will need Python 3.7+ and the following dependencies:

```bash
pip install numpy pandas matplotlib yfinance hmmlearn
```

## 🚀 Usage

### Running the S&P 500 Regime Experiment

The main experiment script downloads historical S&P 500 index data, standardizes the log returns and rolling features, and compares our custom `ContinuousHMM` against `hmmlearn.GaussianHMM`.

Ensure you are in the root directory of the project, then run:

```bash
python -m src.experiments.sp500_regime_modeling
```

*Note: The script will output the mean and standard deviation of log returns for each identified regime, and display several Matplotlib figures plotting the index alongside the colored regimes.*

### Using the Models in Your Own Code

You can easily import and use the models in your own scripts.

**Continuous HMM:**
```python
import numpy as np
from src.models.ContinuousHMM import ContinuousHMM

# Initialize a 3-state continuous HMM
model = ContinuousHMM(num_states=3)

# obs should be a numpy array of shape (seq_len, num_features)
obs = np.random.normal(0, 1, (100, 1))

# Fit the model (Baum-Welch)
model.fit(obs, num_iter=50)

# Decode the hidden states (Viterbi)
hidden_states = model.decode(obs)
```

## 🧪 Testing

The codebase includes standard Python `unittest` suites to ensure mathematical accuracy and numerical stability.

To run all unit tests from the root directory:

```bash
python -m unittest discover -s src/models -p "*_test.py"
```

Or run them individually:

```bash
python -m unittest src.models.continuous_hmm_test
python -m unittest src.models.discrete_hmm_test
```

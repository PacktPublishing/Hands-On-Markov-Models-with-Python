from __future__ import print_function


import datetime
import numpy as np
import pandas as pd
from matplotlib import cm, pyplot as plt
from matplotlib.dates import YearLocator, MonthLocator

# try:
#     from matplotlib.finance import quotes_historical_yahoo_ochl
# except ImportError:
#     # For Matplotlib prior to 1.5.
#     from matplotlib.finance import (
#         quotes_historical_yahoo as quotes_historical_yahoo_ochl
#         )

from hmmlearn.hmm import GaussianHMM


print(__doc__)

quotes = pd.read_csv('data/yahoofinance-INTC-19950101-20040412.csv',
                     index_col=0,
                     parse_dates=True,
                     infer_datetime_format=True)
# Unpack quotes
dates = np.array(quotes.index, dtype=int)
close_v = np.array(quotes.Close)
volume = np.array(quotes.Volume)[1:]

# Take diff of close value. Note that this makes
# ``len(diff) = len(close_t) - 1``, therefore, other quantities also
# need to be shifted by 1.
diff = np.diff(close_v)
dates = dates[1:]
close_v = close_v[1:]

# Pack diff and volume for training.
X = np.column_stack([diff, volume])

# Make an HMM instance and execute fit
model = GaussianHMM(n_components=4, covariance_type="diag",
n_iter=1000).fit(X)

# Predict the optimal sequence of internal hidden state
hidden_states = model.predict(X)

print("Transition matrix")
print(model.transmat_)
print()

print("Means and vars of each hidden state")
for i in range(model.n_components):
    print("{0}th hidden state".format(i))
    print("mean = ", model.means_[i])
    print("var = ", np.diag(model.covars_[i]))
    print()

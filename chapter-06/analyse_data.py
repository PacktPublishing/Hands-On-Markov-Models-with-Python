import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
from hmmlearn.hmm import GMMHMM
import warnings
import itertools
from tqdm import tqdm

warnings.filterwarnings("ignore")

plt.style.use('ggplot')

company = 'GOOGL'
data = pd.read_csv('data/company_data/{company}.csv'.format(company=company))


def plot_data(column):
    """
    Plot data for the corresponding column with respect to date

    :param column: column for which the data is going to be plotted
    :type column: str
    """
    fig = plt.figure()
    axes = fig.add_subplot(111)

    date = np.array(data['date'], dtype='datetime64[D]')
    column_data = np.array(data[column])

    axes.plot(date, column_data)
    axes.set_title('{company} {column}'.format(company=company, column=column))
    axes.set_xlabel('Date')
    axes.set_ylabel(column)

    # Limit the x-axis to show only years info
    years = mdates.YearLocator()
    months = mdates.MonthLocator()
    date_formatter = mdates.DateFormatter('%Y-%m-%d')
    axes.xaxis.set_major_locator(years)
    axes.xaxis.set_major_formatter(date_formatter)
    axes.xaxis.set_minor_locator(months)

    fig.autofmt_xdate()
    plt.show()


# Feature extraction
open_price = np.array(data['open'])
close_price = np.array(data['close'])
high_price = np.array(data['high'])
low_price = np.array(data['low'])

frac_change = (close_price - open_price) / open_price
frac_high = (high_price - open_price) / open_price
frac_low = (open_price - low_price) / open_price

training_data = np.column_stack((frac_change, frac_high, frac_low))[0:100, :]

n_mixture_components = 5
n_hidden_states = 4
n_latency_days = 10

hmm = GMMHMM(n_components=n_hidden_states, n_mix=n_mixture_components)
hmm.fit(training_data)

predicted_closing_prices = []

frac_change_range = np.linspace(-0.1, 0.1, 50)
frac_high_range = np.linspace(0, 0.1, 10)
frac_low_range = np.linspace(0, 0.1, 10)

possible_outcomes = np.array(list(itertools.product(frac_change_range, frac_high_range, frac_low_range)))


def get_most_probable_outcome(day_index):
    previous_data = training_data[day_index - n_latency_days: day_index, :]
    outcome_score = []
    for possible_outcome in possible_outcomes:
        total_data = np.row_stack((previous_data, possible_outcome))
        outcome_score.append(hmm.score(total_data))
    most_probable_outcome = possible_outcomes[np.argmax(outcome_score)]
    return most_probable_outcome


def get_closing_price(day_index):
    predicted_frac_change, _, _ = get_most_probable_outcome(day_index)
    return open_price[day_index] * (1 + predicted_frac_change)


for index in tqdm(range(n_latency_days, training_data.shape[0])):
    predicted_closing_price = get_closing_price(index)
    predicted_closing_prices.append(predicted_closing_price)

predicted_closing_prices = np.array(predicted_closing_prices)

date = np.array(data['date'], dtype='datetime64[D]')[n_latency_days:training_data.shape[0]:]
actual_closing_prices = np.array(data['close'])[n_latency_days:training_data.shape[0]:]

plt.plot(date, actual_closing_prices, c="red")
plt.plot(date, predicted_closing_prices, c="blue")
plt.show()




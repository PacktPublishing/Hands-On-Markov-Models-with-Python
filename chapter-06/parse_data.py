"""
Usage: parse_data.py --company=<company>
"""
import os
import tarfile
import pandas as pd
from pandas import errors as pd_errors
from functools import reduce
from docopt import docopt

args = docopt(doc=__doc__, argv=None,
              help=True, version=None,
              options_first=False)

years = [2015, 2016, 2017]
company = args['--company']


# Getting the data files list
data_files_list = []
for year in years:
    year_directory = 'data/{year}'.format(year=year)
    for file in os.listdir(year_directory):
        data_files_list.append('{year_directory}/{file}'.format(year_directory=year_directory, file=file))


def parse_data(file_name, company_symbol):
    """
    Returns data for the corresponding company

    :param file_name: name of the tar file
    :param company_symbol: company symbol
    :type file_name: str
    :type company_symbol: str
    :return: dataframe for the corresponding company data
    :rtype: pd.DataFrame
    """
    tar = tarfile.open(file_name)
    try:
        price_report = pd.read_csv(tar.extractfile('prices.csv'))
        company_price_data = price_report[price_report['symbol'] == company_symbol]
        return company_price_data
    except (KeyError, pd_errors.EmptyDataError):
        return pd.DataFrame()


# Getting the complete data for a given company
company_data = reduce(lambda df, file_name: df.append(parse_data(file_name, company)),
                      data_files_list,
                      pd.DataFrame())
company_data = company_data.sort_values(by=['date'])

# Create folder for company data if does not exists
if not os.path.exists('data/company_data'):
    os.makedirs('data/company_data')

# Write data to a CSV file
company_data.to_csv('data/company_data/{company}.csv'.format(company=company),
                    columns=['date', 'open', 'high', 'low', 'close', 'volume', 'adj_close'],
                    index=False)

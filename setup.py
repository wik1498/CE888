# Load, clean and prepare datasets

import os
import numpy as np
import pandas as pd
from sklearn import preprocessing


# load dataframe from csv file, ready for cleaning and preparation
def load_dataset(path):
    print("Loading dataset %s" % path)
    df = pd.read_csv(path)

    return df


# save prepared dataframe to separate csv file
def save_dataframe(df, outdir, outname):
    if not os.path.exists(outdir):
        os.mkdir(outdir)

    fullname = os.path.join(outdir, outname)
    print("Saving dataframe to %s" % fullname)

    df.to_csv(fullname)


# replace value that don't make sense with the mode
def replace_by_mode(df, value, col):
    df[col] = df[col].replace(value, np.nan)
    df = df.fillna(df[col].mode()[0])
    return df


# drop rows with values that don't make sense
def drop_rows(df, value, col):
    df = df[df[col] != value]
    return df


# load wines dataset and rename y column to 'class'
def prepare_wine_dataset(path):
    df = load_dataset(path)

    df = df.rename(columns={'quality': 'class'})

    # There are several options here; merge classes into sets {3, 4, 5}, {6, 7, 8, 9}, only look at classes 4 and 6, etc
    df = df[(df['class'] == 4) | (df['class'] == 6)]

    return df


# load weather dataset, alter erroneous values and rename y column to 'class'
def prepare_weather_dataset(path):
    df = pd.read_csv(path)

    # print(df.columns)

    # Must exclude variable RISK_MM in order to prevent it from leaking the answers into the model
    df = df.drop(columns=['RISK_MM'])

    # Convert date from integer to month:
    df['Date'] = pd.DatetimeIndex(df['Date']).month

    df = df.fillna(df['WindGustDir'].mode()[0])

    # These numerical, continuous columns contain an erroneous discrete value 'W'
    bad_columns = ['MinTemp', 'MaxTemp', 'Rainfall', 'Humidity9am', 'Humidity3pm', 'WindSpeed9am', 'WindSpeed3pm',
                   'Evaporation', 'Sunshine', 'WindGustSpeed', 'Pressure9am', 'Pressure3pm', 'Cloud9am', 'Cloud3pm',
                   'Temp9am', 'Temp3pm', 'RainToday']

    for i in bad_columns:
        # Either replace erroneous values by mode/mean, or delete rows containing them
        print('Preparing erroneous column: ', i)
        # df = replace_by_mode(df, 'W', i)
        df = drop_rows(df, 'W', i)

    # These discrete valued columns must have their values encoded into numerical labels
    for i in ['Date', 'Location', 'WindGustDir', 'WindDir9am', 'WindDir3pm', 'RainToday', 'RainTomorrow']:
        print('Encoding labels for column ', i)
        le = preprocessing.LabelEncoder()
        le.fit(df[i])
        df[i] = le.transform(df[i])

    df = df.rename(columns={'RainTomorrow': 'class'})

    return df


def prepare_breast_cancer_dataset(path):
    df = pd.read_csv(path)

    df = df.rename(columns={'diagnosis': 'class'})

    return df


def setup_datasets():
    # raw data location
    data_dir = "raw_data/"
    # output directory for cleaned dataframes
    df_dir = "data/"

    # dataset file names
    wine_data = "wine_data.csv"
    weather_aus_data = "weather_aus_data.csv"
    breast_cancer_data = "breast_cancer_data.csv"

    # for each dataset:
    #   load,
    #   fix erroneous values,
    #   rename target (y column) to 'class',
    #   save in dataframe directory.

    print('\npreparing wine dataset')
    wine_df = prepare_wine_dataset(data_dir + wine_data)
    save_dataframe(wine_df, df_dir, wine_data)

    print('\npreparing weather dataset')
    weather_df = prepare_weather_dataset(data_dir + weather_aus_data)
    save_dataframe(weather_df, df_dir, weather_aus_data)

    print('\npreparing breast cancer dataset')
    bc_df = prepare_breast_cancer_dataset(data_dir + breast_cancer_data)
    save_dataframe(bc_df, df_dir, breast_cancer_data)


if __name__ == "__main__":
    setup_datasets()

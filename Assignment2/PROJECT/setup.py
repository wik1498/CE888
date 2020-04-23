# Load, clean and prepare datasets

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import preprocessing


# load dataframe from csv file, ready for cleaning and preparation
from sklearn.preprocessing import MinMaxScaler


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

    df.to_csv(fullname, index=False)


# replace value that don't make sense with the mode
def replace_by_mode(df, value, col):
    df[col] = df[col].replace(value, np.nan)
    df = df.fillna(df[col].mode()[0])
    return df


# drop rows with values that don't make sense
def drop_rows(df, value, col):
    df = df[df[col] != value]
    return df


def scale(df):
    scaler = MinMaxScaler()
    cols = df.columns.drop(['class'])

    df[cols] = scaler.fit_transform(df[cols])
    return df


def show_plot(df, title):
    distribution = np.bincount(df['class'])
    plt.bar(range(0, distribution.size), distribution)
    plt.title(title)
    plt.xlabel('Class')
    plt.ylabel('Samples')
    plt.show()


# load wines dataset and rename y column to 'class'
def prepare_wine_dataset(path):
    df = load_dataset(path)

    df = df.rename(columns={'quality': 'class'})

    # There are several options here; merge classes into sets {3, 4, 5}, {6, 7, 8, 9}, only look at classes 4 and 6, etc
    # df = df[(df['class'] == 4) | (df['class'] == 6)]
    # df = df[(df['class'] == 5) | (df['class'] == 6)]

    # convert classes (4, 6) to 0, 1, 2 etc
    le = preprocessing.LabelEncoder()
    le.fit(df['class'])
    df['class'] = le.transform(df['class'])

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
    for i in ['RainTomorrow']:
        print('Encoding labels for column ', i)
        le = preprocessing.LabelEncoder()
        le.fit(df[i])
        df[i] = le.transform(df[i])

    # alternative method of converting categorical features into numeric values
    for col in ['Date', 'Location', 'WindGustDir', 'WindDir9am', 'WindDir3pm', 'RainToday']:
        print('Encoding labels for column ', i)
        dummies = pd.get_dummies(df[col], prefix=col)
        df = pd.concat([df, dummies], axis=1)
        df.drop(col, axis=1, inplace=True)

    df = df.rename(columns={'RainTomorrow': 'class'})

    # sample 10% of the data to increase execution time during testing
    df = df.sample(frac=0.1, random_state=1)

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
    #   convert discrete values to binary features
    #   use minmaxscaler on features
    #   rename target (y column) to 'class',
    #   save in dataframe directory.

    print('\npreparing wine dataset')
    wine_df = prepare_wine_dataset(data_dir + wine_data)
    scale(wine_df)
    save_dataframe(wine_df, df_dir, wine_data)

    print('\npreparing weather dataset')
    weather_df = prepare_weather_dataset(data_dir + weather_aus_data)
    scale(weather_df)
    save_dataframe(weather_df, df_dir, weather_aus_data)

    print('\npreparing breast cancer dataset')
    bc_df = prepare_breast_cancer_dataset(data_dir + breast_cancer_data)
    scale(bc_df)
    save_dataframe(bc_df, df_dir, breast_cancer_data)

    print('\nDisplaying dataframe graphs:')
    show_plot(wine_df, "Wine Dataset")
    show_plot(weather_df, "Weather Dataset")
    show_plot(bc_df, "Breast Cancer Dataset")


if __name__ == "__main__":
    setup_datasets()

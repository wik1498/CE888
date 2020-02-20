import os

import pandas as pd

from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
from sklearn.model_selection import cross_val_score, KFold


def load_dataset(path):
    df = pd.read_csv(path)
    # TODO: exception handling, e.g. data has not been cleaned? -> instruct to run setup.py

    return df


def describe(df):
    print("Rows: %d\tColumns: %d" % (df.shape[0], df.shape[1]))

    # report on class imbalance
    vc = df['class'].value_counts()

    imbalance_percentage = 100 * (1 - (vc.min() / vc.sum()))

    print("Imbalance: %.2f%%\t(%d / %d)" % (imbalance_percentage, vc.min(), vc.sum()))
    print("Class counts: ")
    print(vc)


def get_simple_KFold_score(classifier, X, y, n):
    scores = cross_val_score(classifier, X, y, cv=n)

    return scores.mean()


# Necessary because cross_val_score cannot return f1 score per class
def get_KFold_class_scores(classifier, df, n):
    kf = KFold(n_splits=n)

    X = df.drop(columns=['class'])
    y = df['class']

    res = [0, 0]

    for train_index, test_index in kf.split(df):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]

        classifier.fit(X_train, y_train)
        y_pred = classifier.predict(X_test)

        res = res + f1_score(y_test, y_pred, average=None)

    res = res / n

    return res


# utility function for formatting, separates sections of results
def print_separator():
    print('--------------------------------------')


def data_report():
    df_dir = "data/"

    # dataset file names
    datasets = ["wine_data.csv", "weather_aus_data.csv", "breast_cancer_data.csv"]

    for filename in datasets:
        df = load_dataset(os.path.join(df_dir, filename))

        print_separator()
        print('Dataset analysis: ', filename)

        # report on dimensions, class imbalance
        describe(df)

        print()

        print('Baseline classification: ')
        classifiers = [tree.DecisionTreeClassifier(), RandomForestClassifier()]

        for c in classifiers:
            print('\nClassifier: %s' % c.__class__)
            print('f1 scores per class:')
            print(get_KFold_class_scores(c, df, 5))


if __name__ == "__main__":
    data_report()

import os
import util

import pandas as pd
import matplotlib.pyplot as plt

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

    res = []

    for train_index, test_index in kf.split(df):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]

        classifier.fit(X_train, y_train)
        y_pred = classifier.predict(X_test)

        res.append(f1_score(y_test, y_pred, average='weighted'))

    return res


def data_report():
    df_dir = "data/"

    # dataset file names
    datasets = ["breast_cancer_data.csv", "wine_data.csv", "weather_aus_data.csv"]

    results = []
    for filename in datasets:
        df = load_dataset(os.path.join(df_dir, filename))

        util.print_separator()
        print('Dataset analysis: ', filename)

        # report on dimensions, class imbalance
        describe(df)

        print('Baseline classification: ')
        classifiers = [RandomForestClassifier()]  # [tree.DecisionTreeClassifier(), RandomForestClassifier()]

        for c in classifiers:
            print('\nClassifier: %s' % c.__class__)
            print('f1 scores per class:')
            res = (get_KFold_class_scores(c, df, 5))
            print(res)
            results.append(res)

    # Creates two subplots and unpacks the output array immediately
    fig, axes = plt.subplots(1, 3, sharey=True)

    for r, ax, lb in zip(results, axes, ['Cancer', 'Wine', 'Weather']):
        ax.boxplot(r , labels=[lb])

    plt.title('Baseline Random Forest')
    plt.xlabel('Dataset')
    plt.ylabel('F1 score')
    plt.ylim(0.4,1.0)
    plt.show()



if __name__ == "__main__":
    data_report()

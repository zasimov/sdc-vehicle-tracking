"""Train classifier and save it to `svc.pickle`"""

import argparse
import datetime
import pickle
import time

import h5py
from matplotlib import pyplot as plt
import numpy as np
from sklearn import svm
from sklearn import cross_validation
from sklearn.model_selection import learning_curve
from sklearn.model_selection import ShuffleSplit

from vt import journal


class Timer:

    def __enter__(self):
        self.start = time.clock()
        return self

    def __exit__(self, *args):
        self.end = time.clock()
        self.interval = self.end - self.start


def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None,
                        n_jobs=1, train_sizes=np.linspace(.1, 1.0, 5)):
    """
    Generate a simple plot of the test and training learning curve.

    Parameters
    ----------
    estimator : object type that implements the "fit" and "predict" methods
        An object of that type which is cloned for each validation.

    title : string
        Title for the chart.

    X : array-like, shape (n_samples, n_features)
        Training vector, where n_samples is the number of samples and
        n_features is the number of features.

    y : array-like, shape (n_samples) or (n_samples, n_features), optional
        Target relative to X for classification or regression;
        None for unsupervised learning.

    ylim : tuple, shape (ymin, ymax), optional
        Defines minimum and maximum yvalues plotted.

    cv : int, cross-validation generator or an iterable, optional
        Determines the cross-validation splitting strategy.
        Possible inputs for cv are:
          - None, to use the default 3-fold cross-validation,
          - integer, to specify the number of folds.
          - An object to be used as a cross-validation generator.
          - An iterable yielding train/test splits.

        For integer/None inputs, if ``y`` is binary or multiclass,
        :class:`StratifiedKFold` used. If the estimator is not a classifier
        or if ``y`` is neither binary nor multiclass, :class:`KFold` is used.

        Refer :ref:`User Guide <cross_validation>` for the various
        cross-validators that can be used here.

    n_jobs : integer, optional
        Number of jobs to run in parallel (default 1).
    """
    plt.figure()
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.grid()

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Cross-validation score")

    plt.legend(loc="best")
    return plt, test_scores_mean[-1]


def learn_fast(args, X, y, random_state, classifier):
    X_train, X_test, y_train, y_test = cross_validation.train_test_split(
        X, y, test_size=args.test_size, random_state=random_state)

    classifier.fit(X_train, y_train)

    return classifier.score(X_test, y_test)


def learn_slow(args, X, y, random_state, classifier):
    cv = ShuffleSplit(n_splits=10, test_size=args.test_size, random_state=random_state)

    _, accuracy = plot_learning_curve(classifier, svc.__class__.__name__, X, y, (0.7, 1.01), cv)
    plt.show()

    return accuracy


def default_output(C):
    return 'svc-%s.pickle' % round(C)


if __name__ == '__main__':
    parser = argparse.ArgumentParser('python train_svc.py')
    parser.add_argument('--dataset', default='dataset.h5')
    parser.add_argument('--params-file', default='params.pickle')
    parser.add_argument('--test-size', default=0.2, type=float)
    parser.add_argument('--journal', default='journal.csv')
    parser.add_argument('-C', type=float, default=1.0, help='SVM C parameter')
    parser.add_argument('--output')
    parser.add_argument('--slow', default=False, action='store_true')

    args = parser.parse_args()

    with h5py.File(args.dataset) as dataset:
        X = np.array(dataset['features'])
        y = np.array(dataset['targets'])

    #
    # Train
    #
    random_state = np.random.randint(0, 100)

    C = args.C
    svc = svm.LinearSVC(C=C)

    with Timer() as train_time:
        if not args.slow:
            accuracy = learn_fast(args, X, y, random_state, svc)
        else:
            accuracy = learn_slow(args, X, y, random_state, svc)

    if args.slow:
        exit(0)

    #
    # Save trained model
    #
    output = default_output(C) if not args.output else args.output
    with open(output, 'wb') as out_file:
        pickle.dump(svc, out_file)

    #
    # Make journal record
    #
    with open(args.params_file, 'rb') as params_file:
        dataset_params = pickle.load(params_file)

    journal.log(args.journal,
                dataset_params,
                train_time.interval,
                svc.__class__.__name__,
                random_state,
                C,
                accuracy)

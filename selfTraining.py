#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch
import numpy as np
import math


class SelfTraining(torch.nn.Module):

    def __init__(self, classifier, max_iter, p=0.95):
        """
        Self-training algorithm

        Parameters
        ----------
        classifier : rumour classifier
        max_iter : integer
        min confidence threshold: float [0, 1]
            At iteration k, keep label were the confidence is higher than the threshold p
        """
        super().__init__()
        self.classifier = classifier
        self.max_iter = max_iter
        self.p = p


    def fit(self, X, X_no_label, Y):
        """
        Fit the model

        Parameters
        ----------
        X : ndarray shape (n_labeled_samples, n_features)
            Labeled data
        X_no_label : ndarray shape (n_unlabeled_samples, n_features)
            Unlabeled data
        Y : ndarray shape (n_labeled_samples,)
            Labels
        """
        iter_no = 0
        self.final_classifier = self.classifier

        if self.max_iter == 1:
            print('Launching supervised learning only', end='...')
        else:
            print('Launching self-training algorithm ...\n')

        while (X_no_label.shape[0] > 0) & (iter_no < self.max_iter):
            self.final_classifier = self.final_classifier.fit(X, Y)
            # Get confidence scores on unlabeled data
            scores = self.final_classifier.predict_proba(X_no_label).max(axis=1)
            ix = np.isin(range(len(scores)), np.where(scores > self.p)[0])
            # keep best ones and update data
            best_pred = self.final_classifier.predict(X_no_label)[ix]
            X = np.concatenate([X, X_no_label[ix]], axis=0)
            Y = np.concatenate([Y, best_pred])
            X_no_label = X_no_label[~ix]

            # Compute distance and add an interation
            iter_no += 1

            # Monitoring progress
            if self.max_iter == 1:
                print('done')
            else:
                print(' | '.join([("%d" % iter_no).rjust(11), ("%d" % sum(ix)).rjust(11)]))


    def predict(self, X):
        """
        Self-training algorithm

        Parameters
        ----------
        X : ndarray shape (n_samples, n_features)


        Returns
        ---------
        predictions: ndarray shape (n_samples, )
        """
        return self.final_classifier.predict(X)


    def accuracy(self, Xtest, Ytrue):
        """
        Parameters
        ----------
        Xtest : ndarray shape (n_samples, n_features)
            Test data
        Ytrue : ndarray shape (n_samples, )
            Test labels
        """
        predictions = self.predict(Xtest)
        accuracy = sum(predictions == Ytrue) / len(predictions)
        print('Accuracy: {}%'.format(round(accuracy * 100, 2)))

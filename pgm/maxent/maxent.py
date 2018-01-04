# -*- coding: utf-8 -*-

from .classifier import Classifier
from scipy.misc import logsumexp
from scipy import exp
from collections import Counter
import random
import numpy as np


class MaxEnt(Classifier):

    def get_model(self):
        return self.seen

    def set_model(self, model):
        self.seen = model

    model = property(get_model, set_model)

    def train(self, instances, dev_instances=None):
        """Construct a statistical model from labeled instances."""
        self.train_sgd(instances, dev_instances, 0.001, 100)

    def train_sgd(self, train_instances, dev_instances, learning_rate, batch_size):
        """Train MaxEnt model with Minibatch Stochastic Gradient."""

        all_instances = train_instances + dev_instances

        # create set of features limited to just the most common 750
        fc = Counter([f for inst in all_instances for f in inst.features()])
        features = set([feature for (feature, count) in fc.most_common(750)])
        features.add('<bias>')

        # create set of labels
        labels = set([instance.label for instance in all_instances])

        # initialize model to all zero parameters and dictionaries to remember
        # which indices to use for which labels and features
        parameters = np.zeros((len(labels), len(features)))
        feature_indices = {feature: i for i, feature in enumerate(features)}
        label_indices = {label: i for i, label in enumerate(labels)}
        self.model = (parameters, feature_indices, label_indices)

        # cache feature vector in Documents
        for inst in all_instances:
            inst.feature_vector = [feature_indices[f] for f in inst.features()
                                   if f in feature_indices]
            inst.feature_vector.append(feature_indices['<bias>'])

        # create minibatches
        random.shuffle(train_instances)
        batches = [train_instances[i:i + batch_size] for i in
                   range(0, len(train_instances), batch_size)]

        # initialize variables for training
        converged = False
        total_epochs = 0
        epochs_wo_change = 0
        best_params = parameters
        max_acc = self.accuracy(dev_instances)
        max_loglikelihood = self.loglikelihood(dev_instances)

        # print statement to track progress of training
        print('Epoch=' + str(total_epochs) + ' Acc=' + str(max_acc) +
              ' LogLike=' + str(max_loglikelihood))

        # train over each minibatch until acc does not improve
        while not converged:
            total_epochs += 1
            epochs_wo_change += 1
            for batch in batches:
                gradient = self.compute_gradient(batch)
                parameters += gradient * learning_rate
                self.model = (parameters, feature_indices, label_indices)
            acc = self.accuracy(dev_instances)
            loglikelihood = self.loglikelihood(dev_instances)

            # print statements to track progress of training
            print('Epoch=' + str(total_epochs) + ' Acc=' + str(acc) +
                  ' LogLike=' + str(loglikelihood))

            if acc > max_acc:
                epochs_wo_change = 0
                max_acc = acc
                best_params = parameters
            if epochs_wo_change >= 5:
                converged = True
        # update model
        self.model = (best_params, feature_indices, label_indices)

    def classify(self, instance):
        parameters, feature_indices, label_indices = self.model
        instance.feature_vector = [feature_indices[feature]
                                   for feature in instance.features()
                                   if feature in feature_indices]
        instance.feature_vector.append(feature_indices['<bias>'])
        d = self.posterior_probs(instance)
        if d != {}:
            return max(d, key=lambda x: d[x])
        else:
            return random.choice(label_indices.keys())

    def compute_gradient(self, batch):
        return self.observed_values(batch) - self.expected_values(batch)

    def expected_values(self, batch):
        parameters, feature_indices, label_indices = self.model
        values = np.zeros((len(label_indices), len(feature_indices)))
        for instance in batch:
            probs = self.posterior_probs(instance)
            for j in instance.feature_vector:
                for label in probs:
                    i = label_indices[label]
                    values[i, j] += probs[label]
        return values

    def observed_values(self, batch):
        parameters, feature_indices, label_indices = self.model
        values = np.zeros((len(label_indices), len(feature_indices)))
        for instance in batch:
            i = label_indices[instance.label]
            for j in instance.feature_vector:
                values[i, j] += 1
        return values

    def posterior_probs(self, instance):
        """returns a dictionary of P(label|features) for all labels"""
        parameters, feature_indices, label_indices = self.model
        probs = {}
        m = [parameters[label_indices[label], feature_index]
             for feature_index in instance.feature_vector
             for label in label_indices]
        for label in label_indices:
            l = [parameters[label_indices[label], feature_index]
                 for feature_index in instance.feature_vector]
            probs[label] = exp(sum(l) - logsumexp(m))
        return probs

    def loglikelihood(self, instances):
        likelihood = 0.0
        for instance in instances:
            probs = self.posterior_probs(instance)
            if probs[instance.label] != 0:
                likelihood += np.log(probs[instance.label])
        return likelihood

    def accuracy(self, test):
        correct = [self.classify(x) == x.label for x in test]
        return float(sum(correct)) / len(correct)
# -*- coding: utf-8 -*-

from abc import ABCMeta, abstractmethod, abstractproperty
from _pickle import dump, load, HIGHEST_PROTOCOL as HIGHEST_PICKLE_PROTOCOL


class Classifier(object):
    """An abstract text classifier.
    Subclasses must provide training and classification methods, as well as
    an implementation of the model property. The internal representation of
    a classifier's model is entirely up to the subclass, but the read/write
    model property must return/accept a single object (e.g., a list of
    probability distributions)."""

    __metaclass__ = ABCMeta

    def __init__(self, model=None):
        if isinstance(model, str):
            self.load_model(model)
        else:
            self.model = model

    def get_model(self):
        return None

    def set_model(self, model):
        pass
    model = abstractproperty(get_model, set_model)

    def save(self, file):
        """Save the current model to the given file."""
        if isinstance(file, str):
            with open(file, "wb") as file:
                self.save(file)
        else:
            dump(self.model, file, HIGHEST_PICKLE_PROTOCOL)

    def load(self, file):
        """Load a saved model from the given file."""
        if isinstance(file, str):
            with open(file, "rb") as file:
                self.load(file)
        else:
            self.model = load(file)

    @abstractmethod
    def train(self, instances):
        """Construct a statistical model from labeled instances."""
        pass

    @abstractmethod
    def classify(self, instance):
        """Classify an instance and return the expected label."""
        return None
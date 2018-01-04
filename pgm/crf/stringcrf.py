import numpy as np
from numpy import fromiter, int32
from arsenal.alphabet import Alphabet
from .crf import CRF


def build_domain(data):
    """
    Do feature extraction to determine the set of *supported* featues, i.e.
    those active in the ground truth configuration and active labels. This
    function will each features and label an integer.
    """
    L = Alphabet()
    A = Alphabet()
    for x in data:
        L.add_many(x.truth)
        A.add_many(f for token in x.sequence for f in token.attributes)
    # domains are now ready
    L.freeze()
    A.stop_growth()
    return L, A


class StringCRF(CRF):
    """
    Conditional Random Field (CRF) for linear-chain structured models with
    string-valued labels and features.
    This implementation of StringCRF differs from stringcrf.StringCRF in
    that it encodes features in a more memory efficient fashion; instead
    of computing the feature_table for all (t,yp,p) pairs we take use the
    following trick:
       feature_table[t,yp,y] => x[t].attributes + y*|A|
    This is basically the math used to index a 2d array.
    """

    def __init__(self, label_alphabet, feature_alphabet):
        self.label_alphabet = label_alphabet
        self.feature_alphabet = feature_alphabet
        CRF.__init__(self, len(self.label_alphabet), len(self.feature_alphabet))

    def __call__(self, x):
        return self.label_alphabet.lookup_many(CRF.__call__(self, x))

    def preprocess(self, data):
        """
        preprocessing hook which caches the ``feature_table`` and ``target_features``
        attributes of a Instance.
        """
        A = self.feature_alphabet
        L = self.label_alphabet

        size = (len(A) + len(L))*len(L)
        if self.W.shape[0] != size:
            print('reallocating weight vector.')
            self.W = np.zeros(size)

        for x in data:
            # cache feature_table
            if x.feature_table is None:
                x.feature_table = FeatureVectorSequence(x, A, L)
            # cache target_features
            if x.target_features is None:
                x.target_features = self.path_features(x, list(L.map(x.truth)))


class FeatureVectorSequence(object):
    def __init__(self, instance, A, L):
        self.sequence = [fromiter(A.map(t.attributes), dtype=int32) for t in instance.sequence]
        self.A = len(A)
        self.L = len(L)

    def __getitem__(self, item):
        (t,yp,y) = item
        token = self.sequence[t]
        if yp is not None:
            return np.append(token, yp) + y * self.A
        else:
            return token + y*self.A


class Instance(object):
    def __init__(self, s, truth=None):
        self.sequence = list(s)
        self.truth = truth
        self.N = len(s)
        # CRF will cache data here
        self.feature_table = None
        self.target_features = None

    def F(self, t, yp, y):
        """ Create the vector of active indicies for this label setting. """
        # label-label-token; bigram features
        #for f in self.sequence[t].attributes:
        #    yield '[%s,%s,%s]' % (yp,y,f)
        # label-token; emission features
        for f in self.sequence[t].attributes:
            yield '[%s,%s]' % (y,f)
        # label "prior" feature
        yield '[%s]' % y
        # label-label; transition feature
        yield '[%s,%s]' % (yp,y)
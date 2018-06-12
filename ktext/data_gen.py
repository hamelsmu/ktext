import numpy as np
from typing import ClassVar, Tuple
from keras.utils import Sequence
import logging


class Neg_Sampling_Data_Gen(Sequence):
    """
    Keras custom data generator that allows for negative sampling per batch.

    This object creates a data generator for Keras that is useful when you have
    parallel corpuses of text and want to build a model to predict which pairs of
    text are matching pairs.

    Inspiration taken from: https://stanford.edu/~shervine/blog/keras-how-to-generate-data-on-the-fly.html
    """
    def __init__(self,
                 vectorized_text1: np.ndarray,
                 vectorized_text2: np.ndarray,
                 bs: int,
                 shuffle: bool = True) -> ClassVar:
        """
        vectorized_text1 and vectorized_text2 are complimentary pieces of text that
        you want to train a neural network to match.

        vectorized_text1 : np.ndarray[int]
            This is the vectorized version of 1 body of text.
        vectorized_text2 : np.ndarray[int]
            This is the vectorized version of a complimentary body of text that has the
            same number of rows of `vectorized_text1`
        bs : int
            batch size.  A cartesian product is done within batch of all pairs for
            negative sampling, so effective batch size fed to model will be bs ** 2.
        shuffle : bool
            Whether or not you want to shuffle the data after each epoch.  True by default.
        """
        # checks
        assert vectorized_text1.shape[0] == vectorized_text2.shape[0], 'Number of rows in vectorized_text{1,2} should be equivalent.'
        assert vectorized_text1.ndim == 2, 'Numpy array `vectorized_text1` should only have 2 dimensions.'
        assert vectorized_text2.ndim == 2, 'Numpy array `vectorized_text2` should only have 2 dimensions.'
        logging.warning(f'Effective batch size will be {bs**2:,} because of negative sampling (bs^2).')

        self.nrows = vectorized_text1.shape[0]
        self.text1 = vectorized_text1
        self.text2 = vectorized_text2
        self.bs = bs
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.nrows) / self.bs))

    def __getitem__(self, index: int) -> Tuple:
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.bs:(index+1)*self.bs]

        # Generate data
        X, y = self.__negative_sampling(indexes)

        return X, y

    def on_epoch_end(self) -> None:
        'Updates indexes after each epoch for shuffling'
        self.indexes = np.arange(self.nrows)
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def __negative_sampling(self, indexes: np.ndarray) -> Tuple:
        # cartesian product of indexes paired with itself
        idxs = np.array(np.meshgrid(indexes, indexes))

        # get the indexes for each part of the two pairs (t1, t2)
        t1_idxs = idxs[0].ravel()
        t2_idxs = idxs[1].ravel()

        # use the indexes to retrieve the data
        text1 = self.text1[t1_idxs]
        text2 = self.text2[t2_idxs]

        # label = 1 when indices are equal, otherwise 0
        labels = np.equal(t1_idxs, t2_idxs) * 1  # convert bool to int

        return (text1, text2), labels

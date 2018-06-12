import numpy as np
from typing import ClassVar, Tuple, List
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
                 neg_sample_per_pair: int,
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
            batch size.
        neg_sample_per_pair : int
            The number of negative samples you want to generate per pair of matching texts. This
            will cause your effective batch size to increase to bs * neg_sample_per_pair
        shuffle : bool
            Whether or not you want to shuffle the data after each epoch.  True by default.
        """
        # checks
        max_neg_sample = bs - 1
        self.eff_bs = (bs * neg_sample_per_pair) + bs
        assert vectorized_text1.shape[0] == vectorized_text2.shape[0], 'Number of rows in vectorized_text{1,2} should be equivalent.'
        assert bs <= vectorized_text1.shape[0], f'Batch size cannot exceed number of rows in data: {vectorized_text1.shape[0]:,}'
        assert vectorized_text1.ndim == 2, 'Numpy array `vectorized_text1` should only have 2 dimensions.'
        assert vectorized_text2.ndim == 2, 'Numpy array `vectorized_text2` should only have 2 dimensions.'
        assert neg_sample_per_pair <= max_neg_sample, f'`negative_samples_per_pair` cannot exceed {max_neg_sample:,}. Based upon batch size of {bs:,}'
        logging.warning(f'Effective batch size is {bs} + ({bs}*{neg_sample_per_pair}) = {self.eff_bs:,}')

        self.nrows = vectorized_text1.shape[0]
        self.text1 = vectorized_text1
        self.text2 = vectorized_text2
        self.bs = bs
        self.shuffle = shuffle
        self.on_epoch_end()
        self.k_neg = neg_sample_per_pair * bs

    def __len__(self):
        'Denotes the number of batches per epoch'
        return self.nrows // self.bs

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

    def __negative_sampling(self,
                            indexes: np.ndarray) -> Tuple[List, np.ndarray]:

        # cartesian product of indexes paired with itself
        idxs = np.array(np.meshgrid(indexes, indexes))

        # get the indexes for each part of the two pairs (t1, t2)
        t1_idxs = idxs[0].ravel()
        t2_idxs = idxs[1].ravel()
        match_flag = np.equal(t1_idxs, t2_idxs)

        # get negative indices
        neg_exs = np.stack([t1_idxs[~match_flag], t2_idxs[~match_flag]], axis=1)
        neg_idxs = neg_exs[np.random.choice(neg_exs.shape[0], self.k_neg, replace=False)]

        # get positive indices
        pos_idxs = np.stack([t1_idxs[match_flag], t2_idxs[match_flag]], axis=1)
        assert pos_idxs.shape[0] == self.bs, 'Number of positive examples: {pos_idxs.shape[0]:,} must equal batch size: {self.bs:,}.'

        # combine negative and positive indices
        final_idxs = np.vstack([pos_idxs, neg_idxs])
        np.random.shuffle(final_idxs)

        # use the indexes to retrieve the data
        text1 = self.text1[final_idxs[:, 0]]
        text2 = self.text2[final_idxs[:, 1]]

        # label = 1 when indices are equal, otherwise 0
        labels = np.equal(final_idxs[:, 0], final_idxs[:, 1]) * 1  # convert to int

        # checks
        assert text1.shape[0] == self.eff_bs, f'Num of rows returned from text1 {text1.shape[0]:,} does not match expected value of {self.eff_bs:,}.'
        assert text2.shape[0] == self.eff_bs, f'Num of rows returned from text2 {text2.shape[0]:,} does not match expected value of {self.eff_bs:,}.'
        assert text1.shape[0] == text2.shape[0], f'Num of rows returned by text1: {text1.shape[0]:,} is not equal to text2: {text2.shape[0]:,}.'
        assert labels.shape[0] == text1.shape[0], f'Num of rows in label: {labels.shape[0]:,} must be equal to rows in text: {text1.shape[0]:,}.'

        return [text1, text2], labels

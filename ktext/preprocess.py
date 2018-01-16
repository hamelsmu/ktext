"""
Utilities for cleaning, tokenizing and vectorizing text.
Author: Hamel Husain
"""
from pathos.multiprocessing import Pool, cpu_count
import numpy as np
from more_itertools import chunked
from typing import List, Callable, Union, Any
from math import ceil
from textacy.preprocess import preprocess_text
from textacy.corpus import Corpus
from gensim.corpora.dictionary import Dictionary
from keras.preprocessing.sequence import pad_sequences
import pandas as pd
import spacy
import logging
from itertools import chain
from collections import Counter
import timeit

spacyen_default = spacy.load('en')

def get_time():
    return timeit.default_timer()

def time_diff(start):
    return int(get_time() - start)

# Helper functions that need to be pickled so they live outside the class"
def count_len(tkn_docs: List[List[str]], round_factor=-1) -> List[int]:
    """Build histogram of length of documents (# of tokens) in increments of 10."""
    return Counter([round(len(doc), round_factor) for doc in tkn_docs])

def build_corpus(documents: List[List[str]]) -> Dictionary:
    """Given a List of token sequences, return a gensim Dictionary object."""
    gd = Dictionary(documents=documents)
    return gd


def textacy_cleaner(text: str) -> str:
    """
    Defines the default function for cleaning text.

    This function operates over a list.
    """
    return preprocess_text(text,
                           fix_unicode=True,
                           lowercase=True,
                           transliterate=True,
                           no_urls=True,
                           no_emails=True,
                           no_phone_numbers=True,
                           no_numbers=True,
                           no_currency_symbols=True,
                           no_punct=False,
                           no_contractions=False,
                           no_accents=True)


def spacy_tokenizer(text: str, spacylang=None) -> List[str]:
    """
    Tokenize a string.
    ex:  'Hello World' -> ['Hello', 'World']
    """
    if spacylang is None:
        spacylang = spacyen_default
    return [tok.text for tok in spacylang.tokenizer(text)]


def apply_parallel(data: List[Any], func: Callable) -> List[Any]:
    """
    Apply function to list of elements.

    Automatically determines the chunk size.
    """
    cpu_cores = cpu_count()

    try:
        chunk_size = ceil(len(data) / cpu_cores)
        pool = Pool(cpu_cores)
        transformed_data = pool.map(func, chunked(data, chunk_size), chunksize=1)
    finally:
        pool.close()
        pool.join()
        return transformed_data

class processor_base(object):
    """
    Utility for preprocessing of text for deep learning.
    """

    def __init__(self,
                 cleaner: Callable = None,
                 tokenizer: Callable = None) -> None:
        """
        cleaner: function that takes as input string and outputs a string
        tokenizer: function that takes as input a list of strings and outputs
                   a list of lists of tokens.
        """
        if cleaner is None:
            self.cleaner = textacy_cleaner

        if tokenizer is None:
            self.tokenizer = spacy_tokenizer

    def set_tokenizer(self, func: Callable) -> None:
        """
        Set the tokenizer you wish to use.

        This is a function that f(str) -> List(str)
        """
        raise NotImplementedError
        self.tokenizer = func

    def set_cleaner(self, func: Callable) -> None:
        """
        Set the cleaner you wish to use.

        This is a function that f(str) -> str
        """
        raise NotImplementedError
        self.cleaner = func


class processor(processor_base):
    """
    Pre-process text in memory.

    Includes utilities for cleaning, tokenization, and vectorization in parallel.
    """
    def __init__(self,
                 hueristic_pct_padding: float = .90,
                 append_indicators: bool = False,
                 keep_n: int = 150000,
                 padding: str = 'pre',
                 padding_maxlen: Union[int, None] = None,
                 truncating: str = 'post'
                 ):
        """
        Parameters:
        ----------
        hueristic_pct_padding: float
            This parameter is only used if `padding_maxlen` = None.  A histogram
            of documents is calculated, and the maxlen is set hueristic_pct_padding.
        append_indicators: bool
            If True, will append the tokens '_start_' and '_end_' to the beginning
            and end of your tokenized documents.  This can be useful when training
            seq2seq models.
        keep_n: int = 150000
            This is the maximum size of your vocabulary (unique number of words
            allowed).  Consider limiting this to a reasonable size based upon
            your corpus.
        padding : str
            'pre' or 'post', pad either before or after each sequence.
        padding_maxlen : int or None
            Maximum sequence length, longer sequences are truncated and shorter
            sequences are padded with zeros at the end.  Note if this is specified,
            the `hueristic_pct_padding` is ignored.
        truncating : str
            'pre' or 'post', remove values from sequences larger than padding_maxlen
            either in the beginning or in the end of the sequence.

        See https://keras.io/preprocessing/sequence/

        Attributes:
        -----------
        vocabulary : gensim.corpora.dictionary.Dictionary
            This is a gensim object that is built after parsing all the tokens
            in your corpus.
        n_tokens : int
            The total number of tokens in the corpus.  Will be less than or
            equal to keep_n
        id2token : dict
            dict with { int : str} ex: {'the': 2, 'cat': 3}
            this is used for converting tokens to integers.
        token2id : dict
            dict with {str: int} ex: {2: 'the', 3: 'cat'}
            this is used for decoding predictions back to tokens
        document_length_stats : pandas.DataFrame
            histogram of document lengths.  Can be used to decide padding_maxlen.
        """
        super().__init__()
        self.hueristic_pct = hueristic_pct_padding
        self.append_indicators = append_indicators
        self.keep_n = keep_n
        self.padding = padding
        self.padding_maxlen = padding_maxlen
        self.truncating = truncating

        # These are placeholders for data that will be collected or calculated
        self.vocabulary = Dictionary()
        self.n_tokens = None
        self.id2token = None
        self.token2id = None
        self.document_length_histogram = Counter()
        self.document_length_stats = None
        self.doc_length_huerestic = None

        # These values are 'hardcoded' for now
        self.padding_value = 0.0
        self.padding_dtype = 'int32'
        self.start_tok = '_start_'
        self.end_tok = '_end_'
        self.keep_tokens = [self.start_tok, self.end_tok]

    def process_text(self, text: List[str]) -> List[List[str]]:
        """Combine the cleaner and tokenizer."""
        return self.__apply_tokenizer(self.__apply_cleaner(text))

    def __apply_cleaner(self, data: List[str]) -> List[str]:
        """Apply the cleaner over a list."""
        return [self.cleaner(doc) for doc in data]

    def __apply_tokenizer(self, data: List[str]) -> List[List[str]]:
        """Apply the tokenizer over a list."""
        if self.append_indicators:
            tmp = [[self.start_tok] + self.tokenizer(doc) + [self.end_tok] for doc in data]
            return tmp
        else:
            return [self.tokenizer(doc) for doc in data]

    def parallel_process_text(self, data: List[str]) -> List[List[str]]:
        """Apply cleaner -> tokenizer."""
        return apply_parallel(data, self.process_text)

    def generate_doc_length_stats(self):
        """Analyze document length statistics for padding strategy"""
        hueristic = self.hueristic_pct
        histdf = (pd.DataFrame([(a, b) for a, b in self.document_length_histogram.items()],
                               columns=['bin', 'doc_count'])
                  .sort_values(by='bin'))
        histdf['cumsum_pct'] = histdf.doc_count.cumsum() / histdf.doc_count.sum()

        self.document_length_stats = histdf
        self.doc_length_huerestic = histdf.query(f'cumsum_pct >= {hueristic}').bin.head(1).values[0]
        logging.warning(' '.join(["Setting maximum document length to",
                                  f'{self.doc_length_huerestic} based upon',
                                  f'hueristic of {hueristic} percentile.\n',
                                  'See full histogram by insepecting the',
                                  "`document_length_stats` attribute."]))
        self.padding_maxlen = self.doc_length_huerestic

    def fit(self,
            data: List[str],
            return_tokenized_data: bool = False,
            no_below: int = 100,
            no_above: float = .9) -> Union[None, List[List[str]]]:
        """
        TODO: update docs

        Apply cleaner and tokenzier to raw data and build vocabulary.

        Parameters
        ----------
        data : List[str]
            These are raw documents, which are a list of strings. ex:
            [["The quick brown fox"], ["jumps over the lazy dog"]]
        return_tokenized_data : bool
            Return the tokenized strings.  This is primarly used for debugging
            purposes.
        no_below : int
            See below explanation
        no_above : float
            See below explanation

        When tokenizing documents, filter tokens according to these rules:
        1. occur less than `no_below` documents (absolute number) or
        2. occur more than `no_above` documents (fraction of total corpus size, not absolute number).
        3. after (1), and (2), keep only the first keep_n most frequent tokens.

        Returns
        -------
        None or List[List[str]]
            if return_tokenized_data=True then will return tokenized documents,
            otherwise will not return anything.

        This method heavily leverages gensim https://radimrehurek.com/gensim/corpora/dictionary.html
        """
        now = get_time()
        logging.warning(f'....tokenizing data')
        tokenized_data = list(chain.from_iterable(self.parallel_process_text(data)))

        if not self.padding_maxlen:
            document_len_counters = apply_parallel(tokenized_data, count_len)

            for doc_counter in document_len_counters:
                self.document_length_histogram.update(doc_counter)
            self.generate_doc_length_stats()

        # chunk the data manually for corpus build adnd pass to build corpus method
        logging.warning(f'(1/3) done. {time_diff(now)} sec')
        logging.warning(f'....building corpus')
        now = get_time()
        corpus = build_corpus(tokenized_data)

        # Merge the corpuses from each thread together, this is like a "reduce" step
        logging.warning(f'(2/3) done. {time_diff(now)} sec')
        logging.warning(f'....consolidating corpus')
        now = get_time()
        self.vocabulary.merge_with(corpus)

        # # get rid of rare tokens from corpus such that they will get the same id
        self.vocabulary.filter_extremes(no_below,
                                        no_above,
                                        self.keep_n,
                                        keep_tokens=self.keep_tokens)

        # compactify the ids for each word
        self.vocabulary.compactify()

        # Build Dictionary accounting For 0 padding, and reserve 1 for unknown and rare Words
        self.token2id = dict([(k, v + 2) for k, v in self.vocabulary.token2id.items()])
        self.id2token = dict([(v, k) for k, v in self.token2id.items()])
        self.n_tokens = len(self.id2token.keys())

        # logging
        logging.warning(f'(3/3) done. {time_diff(now)} sec')
        logging.warning(f'Finished parsing {self.vocabulary.num_docs:,} documents.')

        if return_tokenized_data:
            return tokenized_data

    def token_count_pandas(self):
        """ See token counts as pandas dataframe"""
        freq_df = pd.DataFrame([b for a, b in self.vocabulary.dfs.items()],
                               index=[a for a, b in self.vocabulary.dfs.items()],
                               columns=['count'])

        id2tokens = [(b, a) for a, b in self.vocabulary.token2id.items()]

        token_df = pd.DataFrame([b for a, b in id2tokens],
                                index=[a for a, b in id2tokens],
                                columns=['token'])

        return freq_df.join(token_df).sort_values('count', ascending=False)

    def fit_transform(self,
                      data: List[str],
                      no_below: int = 25,
                      no_above: float = 0.8) -> List[List[int]]:
        """
        Apply cleaner and tokenzier to raw data, build vocabulary and return
        transfomred dataset that is a List[List[int]].  This will use
        process-based-threading on all available cores.

        ex:
        >>> data = [["The quick brown fox"], ["jumps over the lazy dog"]]
        >>> pp = preprocess(maxlen=5, no_below=0)
        >>> pp.fit_transform(data)
        # 0 padding is applied
        [[0, 2, 3, 4, 5], [6, 7, 2, 8, 9]]

        Parameters
        ----------
        data : List[str]
            These are raw documents, which are a list of strings. ex:
            [["The quick brown fox"], ["jumps over the lazy dog"]]
        no_below : int
            See below explanation
        no_above : float
            See below explanation

        When tokenizing documents, filter tokens according to these rules:
        1. occur less than `no_below` documents (absolute number) or
        2. occur more than `no_above` documents (fraction of total corpus size, not absolute number).
        3. after (1), and (2), keep only the first keep_n most frequent tokens.

        Returns
        -------
        numpy.array with shape (number of documents, max_len)


        This method leverages gensim https://radimrehurek.com/gensim/corpora/dictionary.html
        """
        tokdata = self.fit(data,
                           return_tokenized_data=True,
                           no_below=no_below,
                           no_above=no_above)

        logging.warning(f'...fit is finished, beginning transform')
        now = get_time()
        vec_data = self.vectorize_parallel(tokdata)
        logging.warning(f'done. {time_diff(now)} sec')
        return vec_data

    def transform(self, data: List[str]) -> List[List[int]]:
        """
        Transform List of documents into List[List[int]]
        If transforming a large number of documents consider using the method
        `transform_parallel` instead.

        ex:
        >> pp = processor()
        >> pp.fit(docs)
        >> new_docs = [["The quick brown fox"], ["jumps over the lazy dog"]]
        >> pp.transform(new_docs)
        [[1, 2, 3, 4], [5, 6, 1, 7, 8]]
        """
        return self.vectorize(self.process_text(data))

    def transform_parallel(self, data: List[str]) -> List[List[int]]:
        """
        Transform List of documents into List[List[int]].  Uses process based
        threading on all available cores.  If only processing a small number of
        documents ( < 10k ) then consider using the method `transform` instead.

        ex:
        >> pp = processor()
        >> pp.fit(docs)
        >> new_docs = [["The quick brown fox"], ["jumps over the lazy dog"]]
        >> pp.transform_parallel(new_docs)
        [[1, 2, 3, 4], [5, 6, 1, 7, 8]]
        """
        return np.vstack(apply_parallel(data, self.transform))

    def get_idx(self, token: str) -> int:
        """Get integer index from token."""
        # return the index for index or if not foudn return out of boundary index which is 1
        return self.token2id.get(token, 1)

    def __vec_one_doc(self, doc: List[str]) -> List[int]:
        """
        Vectorize a single tokenized document.
        ex: ['hello', 'world']
        """
        return [self.get_idx(tok) for tok in doc]

    def vectorize(self, docs: List[List[str]]) -> List[List[int]]:
        """
        Vectorize and apply padding on a set of tokenized doucments
        ex: [['hello, 'world'], ['goodbye', 'now']]

        """
        # First apply indexing on all the rows then pad_sequnces (i found this
        # faster than trying to do these steps on each row
        return pad_sequences(list(map(self.__vec_one_doc, docs)),
                             maxlen=self.padding_maxlen,
                             dtype=self.padding_dtype,
                             padding=self.padding,
                             truncating=self.truncating,
                             value=self.padding_value)

    def vectorize_parallel(self,
                           data: List[List[str]]) -> np.array:
        """
        Apply idx-> token mappings in parallel and apply padding.

        Arguments:
        data: List of List of strings
        """
        indexed_data = apply_parallel(data, self.vectorize)
        # concatenate list of arrays vertically
        return np.vstack(indexed_data)

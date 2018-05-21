"""
Utilities for cleaning, tokenizing and vectorizing text.
Author: Hamel Husain
"""
from pathos.multiprocessing import Pool, cpu_count
from more_itertools import chunked
from typing import List, Callable, Union, Any
from math import ceil
from textacy.preprocess import preprocess_text
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import text_to_word_sequence, Tokenizer
from collections import OrderedDict
import pandas as pd
import logging
from itertools import chain
from collections import Counter
import timeit

def flattenlist(listoflists):
    return list(chain.from_iterable(listoflists))

def get_time():
    return timeit.default_timer()


def time_diff(start):
    return int(get_time() - start)


def myround(x, base=5):
    """Custom rounding for histogram."""
    return int(base * round(float(x) / base))


def count_len(tkn_doc: List[str]) -> List[int]:
    """Build histogram of length of documents (# of tokens) in increments of 5."""
    return myround(len(tkn_doc))


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
                           no_punct=True,
                           no_contractions=False,
                           no_accents=True)


def apply_parallel(func: Callable,
                   data: List[Any],
                   cpu_cores: int = None) -> List[Any]:
    """
    Apply function to list of elements.

    Automatically determines the chunk size.
    """
    if not cpu_cores:
        cpu_cores = cpu_count()

    try:
        chunk_size = ceil(len(data) / cpu_cores)
        pool = Pool(cpu_cores)
        transformed_data = pool.map(func, chunked(data, chunk_size), chunksize=1)
    finally:
        pool.close()
        pool.join()
        return transformed_data


def process_text_constructor(cleaner: Callable,
                             tokenizer: Callable,
                             append_indicators: bool,
                             start_tok: str,
                             end_tok: str):
    """Generate a function that will clean and tokenize text."""
    def process_text(text):
        if append_indicators:
            return [[start_tok] + tokenizer(cleaner(doc)) + [end_tok] for doc in text]
        return [tokenizer(cleaner(doc)) for doc in text]

    return process_text


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
            self.tokenizer = text_to_word_sequence

    def set_tokenizer(self, func: Callable) -> None:
        """
        Set the tokenizer you wish to use.

        This is a function that f(str) -> List(str)
        """
        self.tokenizer = func

    def set_cleaner(self, func: Callable) -> None:
        """
        Set the cleaner you wish to use.

        This is a function that f(str) -> str
        """
        self.cleaner = func


class processor(processor_base):
    """
    Pre-process text in memory.

    Includes utilities for cleaning, tokenization, and vectorization in parallel.
    """
    def __init__(self,
                 heuristic_pct_padding: float = .90,
                 append_indicators: bool = False,
                 keep_n: int = 150000,
                 padding: str = 'pre',
                 padding_maxlen: Union[int, None] = None,
                 truncating: str = 'post'
                 ):
        """
        Parameters:
        ----------
        heuristic_pct_padding: float
            This parameter is only used if `padding_maxlen` = None.  A histogram
            of documents is calculated, and the maxlen is set heuristic_pct_padding.
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
            the `heuristic_pct_padding` is ignored.
        truncating : str
            'pre' or 'post', remove values from sequences larger than padding_maxlen
            either in the beginning or in the end of the sequence.

        See https://keras.io/preprocessing/sequence/

        Attributes:
        -----------
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
        self.heuristic_pct = heuristic_pct_padding
        self.append_indicators = append_indicators
        self.keep_n = keep_n
        self.padding = padding
        self.padding_maxlen = padding_maxlen
        self.truncating = truncating

        # These are placeholders for data that will be collected or calculated
        self.indexer = None
        self.n_tokens = None
        self.id2token = None
        self.token2id = None
        self.document_length_histogram = Counter()
        self.document_length_stats = None
        self.doc_length_huerestic = None
        self.num_cores = cpu_count()

        # These values are 'hardcoded' for now
        self.padding_value = 0.0
        self.padding_dtype = 'int32'
        self.start_tok = '_start_'
        self.end_tok = '_end_'

    def __clear_data(self):
        self.indexer = None
        self.document_length_histogram = None
        self.document_length_stats = None

    def set_num_processes(self, n):
        """Set the number of processes for process bases threading."""
        self.num_cores = min(int(n), cpu_count())

    def process_text(self, text: List[str]) -> List[List[str]]:
        """Combine the cleaner and tokenizer."""
        process_text = process_text_constructor(cleaner=self.cleaner,
                                                tokenizer=self.tokenizer,
                                                append_indicators=self.append_indicators,
                                                start_tok=self.start_tok,
                                                end_tok=self.end_tok)
        return process_text(text)

    def parallel_process_text(self, data: List[str]) -> List[List[str]]:
        """Apply cleaner -> tokenizer."""
        process_text = process_text_constructor(cleaner=self.cleaner,
                                                tokenizer=self.tokenizer,
                                                append_indicators=self.append_indicators,
                                                start_tok=self.start_tok,
                                                end_tok=self.end_tok)
        n_cores = self.num_cores
        return flattenlist(apply_parallel(process_text, data, n_cores))

    def generate_doc_length_stats(self):
        """Analyze document length statistics for padding strategy"""
        heuristic = self.heuristic_pct
        histdf = (pd.DataFrame([(a, b) for a, b in self.document_length_histogram.items()],
                               columns=['bin', 'doc_count'])
                  .sort_values(by='bin'))
        histdf['cumsum_pct'] = histdf.doc_count.cumsum() / histdf.doc_count.sum()

        self.document_length_stats = histdf
        self.doc_length_huerestic = histdf.query(f'cumsum_pct >= {heuristic}').bin.head(1).values[0]
        logging.warning(' '.join(["Setting maximum document length to",
                                  f'{self.doc_length_huerestic} based upon',
                                  f'heuristic of {heuristic} percentile.\n',
                                  'See full histogram by insepecting the',
                                  "`document_length_stats` attribute."]))
        self.padding_maxlen = self.doc_length_huerestic

    def fit(self,
            data: List[str],
            return_tokenized_data: bool = False) -> Union[None, List[List[str]]]:
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

        Returns
        -------
        None or List[List[str]]
            if return_tokenized_data=True then will return tokenized documents,
            otherwise will not return anything.
        """
        self.__clear_data()
        now = get_time()
        logging.warning(f'....tokenizing data')
        tokenized_data = self.parallel_process_text(data)

        if not self.padding_maxlen:
            # its not worth the overhead to parallelize document length counts
            length_counts = map(count_len, tokenized_data)
            self.document_length_histogram = Counter(length_counts)
            self.generate_doc_length_stats()

        # Learn corpus on single thread
        logging.warning(f'(1/2) done. {time_diff(now)} sec')
        logging.warning(f'....building corpus')
        now = get_time()
        self.indexer = custom_Indexer(num_words=self.keep_n)
        self.indexer.fit_on_tokenized_texts(tokenized_data)

        # Build Dictionary accounting For 0 padding, and reserve 1 for unknown and rare Words
        self.token2id = self.indexer.word_index
        self.id2token = {v: k for k, v in self.token2id.items()}
        self.n_tokens = max(self.indexer.word_index.values())

        # logging
        logging.warning(f'(2/2) done. {time_diff(now)} sec')
        logging.warning(f'Finished parsing {self.indexer.document_count:,} documents.')

        if return_tokenized_data:
            return tokenized_data

    def token_count_pandas(self):
        """ See token counts as pandas dataframe"""
        freq_df = pd.DataFrame.from_dict(self.indexer.word_counts, orient='index')
        freq_df.columns = ['count']
        return freq_df.sort_values('count', ascending=False)

    def fit_transform(self,
                      data: List[str]) -> List[List[int]]:
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

        Returns
        -------
        numpy.array with shape (number of documents, max_len)

        """
        tokenized_data = self.fit(data, return_tokenized_data=True)

        logging.warning(f'...fit is finished, beginning transform')
        now = get_time()
        indexed_data = self.indexer.tokenized_texts_to_sequences(tokenized_data)
        logging.warning(f'...padding data')
        final_data = self.pad(indexed_data)
        logging.warning(f'done. {time_diff(now)} sec')
        return final_data

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
        tokenized_data = self.process_text(data)
        indexed_data = self.indexer.tokenized_texts_to_sequences(tokenized_data)
        return self.pad(indexed_data)

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
        logging.warning(f'...tokenizing data')
        tokenized_data = self.parallel_process_text(data)
        logging.warning(f'...indexing data')
        indexed_data = self.indexer.tokenized_texts_to_sequences(tokenized_data)
        logging.warning(f'...padding data')
        return self.pad(indexed_data)

    def pad(self, docs: List[List[int]]) -> List[List[int]]:
        """
        Vectorize and apply padding on a set of tokenized doucments
        ex: [['hello, 'world'], ['goodbye', 'now']]
        """
        # First apply indexing on all the rows then pad_sequnces (i found this
        # faster than trying to do these steps on each row
        return pad_sequences(docs,
                             maxlen=self.padding_maxlen,
                             dtype=self.padding_dtype,
                             padding=self.padding,
                             truncating=self.truncating,
                             value=self.padding_value)


class custom_Indexer(Tokenizer):
    """
    Text vectorization utility class.
    This class inherits keras.preprocess.text.Tokenizer but adds methods
    to fit and transform on already tokenized text.

    Parameters
    ----------
    num_words : int
        the maximum number of words to keep, based
        on word frequency. Only the most common `num_words` words will
        be kept.
    """

    def __init__(self, num_words):
        self.num_words = num_words
        self.word_counts = OrderedDict()
        self.word_docs = {}
        self.document_count = 0

    def fit_on_tokenized_texts(self, tokenized_texts):
            self.document_count = 0
            for seq in tokenized_texts:
                self.document_count += 1

                for w in seq:
                    if w in self.word_counts:
                        self.word_counts[w] += 1
                    else:
                        self.word_counts[w] = 1
                for w in set(seq):
                    if w in self.word_docs:
                        self.word_docs[w] += 1
                    else:
                        self.word_docs[w] = 1

            wcounts = list(self.word_counts.items())
            wcounts.sort(key=lambda x: x[1], reverse=True)
            sorted_voc = [wc[0] for wc in wcounts][:self.num_words]
            # note that index 0 and 1 are reserved, never assigned to an existing word
            self.word_index = dict(list(zip(sorted_voc, list(range(2, len(sorted_voc) + 2)))))

    def tokenized_texts_to_sequences(self, tok_texts):
        """Transforms tokenized text to a sequence of integers.
        Only top "num_words" most frequent words will be taken into account.
        Only words known by the tokenizer will be taken into account.
        # Arguments
            tokenized texts:  List[List[str]]
        # Returns
            A list of integers.
        """
        res = []
        for vect in self.tokenized_texts_to_sequences_generator(tok_texts):
            res.append(vect)
        return res

    def tokenized_texts_to_sequences_generator(self, tok_texts):
        """Transforms tokenized text to a sequence of integers.
        Only top "num_words" most frequent words will be taken into account.
        Only words known by the tokenizer will be taken into account.
        # Arguments
            tokenized texts:  List[List[str]]
        # Yields
            Yields individual sequences.
        """
        for seq in tok_texts:
            vect = []
            for w in seq:
                # if the word is missing you get oov_index
                i = self.word_index.get(w, 1)
                vect.append(i)
            yield vect

[![GitHub license](https://img.shields.io/github/license/hamelsmu/ktext.svg)](https://github.com/hamelsmu/ktext/blob/master/LICENSE) 
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/Django.svg)](https://github.com/hamelsmu/ktext)


# Utilities for pre-processing text for deep learning in [Keras](https://keras.io/).  

`ktext` performs common pre-processing steps associated with deep learning (cleaning, tokenization, padding, truncation).  Most importantly, `ktext` allows you to perform these steps using process-based threading in parallel.  If you don't think you might benefit from parallelization, consider using the text preprocessing utilities [in keras](https://keras.io/preprocessing/text/) instead.

`ktext` helps you with the following:

1.  **Cleaning** You may want to clean your data to remove items like phone numbers and email addresses and replace them with generic tags, or remove HTML.  This step is optional, but can help remove noise in your data.

2.  **Tokenization** Take a raw string, ex "Hello World!" and tokenize it so it looks like ['Hello', 'World', '!']

3. **Generating Vocabulary and a {Token -> index} mapping** Map each unique token in your corpus to an integer value.  This usually stored as a dictionary.  For example {'Hello': 2, 'World':3, '!':4} might be a valid mapping from tokens to integers.  You usually want to reserve an integer for rare or unseen words (`ktext` uses `1`) and another integer for padding (`ktext` uses `0`).  You can set a threshold for rare words (see documentation).

4.  **Truncating and Padding** While it is not necessary, it can be much easier if all your documents are the same length.  The way we can accomplish this is through truncating and padding.  For all documents below the desired length we can pad the document with 0's and  documents above the desired length can be truncated.  This utility allows you to build a histogram of your document lengths and choose a sensible document length for your corpus.

This utility accomplishes all of the above using process-based threading for speed.  Sklearn style `fit`, `transform`, and `fit_transform` interfaces are provided (but not directly compatible with sklearn yet).  Pull requests and comments are welcome.  

Note: This utility is useful if all of your data can fit into memory on a single node.  Otherwise, if your data cannot fit into memory, consider using distributing computing paradigms such as Hive, Spark or Dask.  

## Documentation
[This notebook](./notebooks/Tutorial.ipynb) contains a tutorial on how to use this library.

## Installation

> $ pip install ktext

# nltk_utils_lib

`nltk_utils_lib` is a utility library for natural language processing (NLP) using NLTK.

## Description

This library provides functions for tokenizing sentences, stemming words, and creating a bag of words. It simplifies the process of text preprocessing for machine learning and NLP tasks.

## Installation

To install the library, run the following command:

```bash
pip install nltk_utils_lib

## Usage
```bash
from nltk_utils_lib.nltk_utils import tokenize, stem, bag_of_words

# Example sentence
sentence = "Hello, how are you?"

# Tokenize the sentence
tokens = tokenize(sentence)
print(tokens)  # ['Hello', ',', 'how', 'are', 'you', '?']

# Stem a word
word = "running"
stemmed_word = stem(word)
print(stemmed_word)  # run

# Create a bag of words
all_words = ["hello", "how", "are", "you", "run"]
bag = bag_of_words(tokens, all_words)
print(bag)  # [1, 1, 1, 1, 0]

## Dependencies
- 'nltk'
- 'numpy'

Ensure that you have all dependencies installed before using the library:
```bash
pip install nltk numpy
## License
- This project is licensed under the MIT License.
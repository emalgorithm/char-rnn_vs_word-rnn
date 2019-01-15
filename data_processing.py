import numpy as np
from nltk.tokenize import TweetTokenizer
from collections import Counter


def char_vectorization(text, maxlen=40):
    print('total non-unique chars: {}'.format(len(text)))
    chars = sorted(list(set(text)))
    print('total unique chars:', len(chars))
    char_indices = dict((c, i) for i, c in enumerate(chars))
    indices_char = dict((i, c) for i, c in enumerate(chars))

    # cut the text in semi-redundant sequences of maxlen characters
    step = 3
    sentences = []
    next_chars = []
    for i in range(0, len(text) - maxlen, step):
        sentences.append(text[i: i + maxlen])
        next_chars.append(text[i + maxlen])
    print('training sequences:', len(sentences))
    
    print('Vectorization...')
    x = np.zeros((len(sentences), maxlen, len(chars)), dtype=np.bool)
    y = np.zeros((len(sentences), len(chars)), dtype=np.bool)
    for i, sentence in enumerate(sentences):
        for t, char in enumerate(sentence):
            x[i, t, char_indices[char]] = 1
        y[i, char_indices[next_chars[i]]] = 1
    
    return x, y, chars, char_indices, indices_char


def word_vectorization(text, maxlen, max_words=None, step=3):
    tknzr = TweetTokenizer()
    
    text = tknzr.tokenize(text)
    print('total non-unique words: {}'.format(len(text)))
    print('total unique words before limit: {}'.format(len(set(text))))
    if max_words is None:
        max_words = len(set(text))
    cnt = Counter(text)
    top_words = [key for key, count in cnt.most_common(max_words)]
    text = [w for w in text if w in top_words]
    
    words = sorted(list(set(text)))
    print('total unique words after limit:', len(words))
    word_indices = dict((w, i) for i, w in enumerate(words))
    indices_word = dict((i, w) for i, w in enumerate(words))
    
    # cut the text in semi-redundant sequences of maxlen words
    sentences = []
    next_words = []
    for i in range(0, len(text) - maxlen, step):
        sentences.append(text[i: i + maxlen])
        next_words.append(text[i + maxlen])
    print('training sequences:', len(sentences))
    
    print('Vectorization...')
    x = np.zeros((len(sentences), maxlen, len(words)), dtype=np.bool)
    y = np.zeros((len(sentences), len(words)), dtype=np.bool)
    for i, sentence in enumerate(sentences):
        for t, word in enumerate(sentence):
            x[i, t, word_indices[word]] = 1
        y[i, word_indices[next_words[i]]] = 1
    
    return x, y, words, word_indices, indices_word


def word_vectorization_for_embedding(text, maxlen, max_words=None, step=3):
    tknzr = TweetTokenizer()
    
    text = tknzr.tokenize(text)
    print('total non-unique words: {}'.format(len(text)))
    print('total unique words before limit: {}'.format(len(set(text))))
    if max_words is None:
        max_words = len(set(text))
    cnt = Counter(text)
    top_words = [key for key, count in cnt.most_common(max_words)]
    text = [w for w in text if w in top_words]
    
    words = sorted(list(set(text)))
    print('total unique words after limit:', len(words))
    word_indices = dict((w, i) for i, w in enumerate(words))
    indices_word = dict((i, w) for i, w in enumerate(words))
    
    # cut the text in semi-redundant sequences of maxlen words
    sentences = []
    next_words = []
    for i in range(0, len(text) - maxlen, step):
        sentences.append(text[i: i + maxlen])
        next_words.append(text[i + maxlen])
    print('training sequences:', len(sentences))
    
    print('Vectorization...')
    x = np.zeros((len(sentences), maxlen), dtype=np.bool)
    y = np.zeros((len(sentences), len(words)), dtype=np.bool)
    for i, sentence in enumerate(sentences):
        for t, word in enumerate(sentence):
            x[i, t] = word_indices[word]
        y[i, word_indices[next_words[i]]] = 1
    
    return x, y, words, word_indices, indices_word

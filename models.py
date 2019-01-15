from keras.models import Sequential
from keras.callbacks import ModelCheckpoint
from keras.layers import Dense, Dropout, LSTM
from keras.layers import CuDNNLSTM
from keras.layers.embeddings import Embedding
import numpy as np


def create_char_rnn(maxlen, num_chars):
    model = Sequential()
    model.add(CuDNNLSTM(256, return_sequences=True, input_shape=(maxlen, num_chars)))
    model.add(Dropout(0.2))
    model.add(CuDNNLSTM(256, return_sequences=True))
    model.add(Dropout(0.2))
    model.add(CuDNNLSTM(256))
    model.add(Dropout(0.2))
    model.add(Dense(num_chars, activation='softmax'))

    return model


def create_word_rnn(maxlen, num_words, word_embedding=False):
    model = Sequential()
    if word_embedding:
        model.add(Embedding(num_words, 100, input_length=maxlen))
    model.add(CuDNNLSTM(256, return_sequences=True, input_shape=(maxlen, num_words)))
    model.add(Dropout(0.2))
    model.add(CuDNNLSTM(256, return_sequences=True))
    model.add(Dropout(0.2))
    model.add(CuDNNLSTM(256))
    model.add(Dropout(0.2))
    model.add(Dense(num_words, activation='softmax'))

    return model


def load_trained_char_rnn(weights_path, maxlen, num_tokens):
    model = create_char_rnn(maxlen, num_tokens)
    model.load_weights(weights_path)
    return model


def load_trained_word_rnn(weights_path, maxlen, num_tokens, word_embedding=False):
    model = create_word_rnn(maxlen, num_tokens, word_embedding)
    model.load_weights(weights_path)
    return model


def sample(preds, temperature=1.0):
    # helper function to sample an index from a probability array
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)


def generate(model, seed, num_chars, maxlen, char_indices, indices_char, logger, diversities=[0.2, 0.5, 1.0, 1.2]):
    for diversity in [0.2, 0.5, 1.0, 1.2]:
        sentence = seed
        logger.write('----- diversity: {} \n'.format(diversity))

        generated = ''
        generated += sentence
        logger.write('----- Generating with seed: "' + sentence + '" \n')
        logger.write(generated)

        for i in range(400):
            x_pred = np.zeros((1, maxlen, num_chars))
            for t, char in enumerate(sentence):
                x_pred[0, t, char_indices[char]] = 1.

            preds = model.predict(x_pred, verbose=0)[0]
            next_index = sample(preds, diversity)
            next_char = indices_char[next_index]

            generated += next_char
            sentence = sentence[1:] + next_char

            logger.write(next_char)
            logger.flush()
        logger.write('\n')


def generate_words(model, seed, num_chars, maxlen, char_indices, indices_char, logger, diversities=[0.2, 0.5, 1.0, 1.2]):
    for diversity in [0.2, 0.5, 1.0, 1.2]:
        sentence = seed
        logger.write('----- diversity: {} \n'.format(diversity))

        generated = []
        generated += sentence
        logger.write('----- Generating with seed: "' + sentence + '" \n')
        logger.write(generated)

        for i in range(400):
            x_pred = np.zeros((1, maxlen, num_chars))
            for t, char in enumerate(sentence):
                x_pred[0, t, char_indices[char]] = 1.

            preds = model.predict(x_pred, verbose=0)[0]
            next_index = sample(preds, diversity)
            next_char = indices_char[next_index]

            generated.append(next_char)
            sentence = sentence[1:] + [next_char]

            logger.write(next_char)
            logger.flush()
        logger.write('\n')
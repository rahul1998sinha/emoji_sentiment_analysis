import numpy as np
import pandas as pd
import emoji

from keras.models import Sequential
from keras.layers import Dense, Input, Dropout, SimpleRNN, LSTM, Activation
from keras.utils import np_utils


class Model(object):

    def __init__(self):
        print("Reading Data....")

    # function to create emoji dictionary
    def emoji_dept(self):
        emoji_dict = {0: ":heart:", 1: ":baseball:", 2: ":smile:", 3: ":disappointed:", 4: ":fork_and_knife:"}
        for ix in emoji_dict.keys():
            print(ix, end=" ")
            print(emoji.emojize(emoji_dict[ix], use_aliases=True))
        return emoji_dict

    # data preprocessing
    def data_modeling(self):
        # Creating training and testing data
        train = pd.read_csv('train_emoji.csv', header=None)
        test = pd.read_csv('test_emoji.csv', header=None)
        x_train = train[0]
        y_train = train[1]
        x_test = test[0]
        y_test = test[1]
        # Splitting the train data from sentences to words
        for ix in range(x_train.shape[0]):
            x_train[ix] = x_train[ix].split()

        # Splitting the test data from sentences to words
        for ix in range(x_test.shape[0]):
            x_test[ix] = x_test[ix].split()

        # Converting labels into categorical form
        y_train = np_utils.to_categorical(y_train)

        # To check maximum length of sentence in training data
        np.unique(np.array([len(ix) for ix in x_train]), return_counts=True)

        # To check maximum length of sentence in testing data
        np.unique(np.array([len(ix) for ix in x_test]), return_counts=True)

        return x_train, y_train, x_test, y_test

    def universal_glove_file_read(self):

        embeddings_index = {}
        f = open('glove.6B.50d.txt', encoding="utf8")
        for line in f:
            values = line.split()
            word = values[0]
            coefs = np.asarray(values[1:], dtype='float32')
            embeddings_index[word] = coefs
        f.close()
        return embeddings_index

    # Creating  embeddings dictionary with key = word and value = list of words in glove vector

    def glove_file(self, x_train, x_test, embeddings_index):
        # Filling the embedding matrix
        embedding_matrix_train = np.zeros((x_train.shape[0], 10, 50))
        embedding_matrix_test = np.zeros((x_test.shape[0], 10, 50))

        for ix in range(x_train.shape[0]):
            for ij in range(len(x_train[ix])):
                embedding_matrix_train[ix][ij] = embeddings_index[x_train[ix][ij].lower()]

        for ix in range(x_test.shape[0]):
            for ij in range(len(x_test[ix])):
                embedding_matrix_test[ix][ij] = embeddings_index[x_test[ix][ij].lower()]
        return embedding_matrix_train, embedding_matrix_test

    def model_LSTM(self, embedding_mat_train, embedding_mat_test, emoji_d, train, testset):
        # A simple LSTM network
        model = Sequential()
        model.add(LSTM(128, input_shape=(10, 50), return_sequences=True))
        model.add(Dropout(0.5))
        model.add(LSTM(128, return_sequences=False))
        model.add(Dropout(0.5))
        model.add(Dense(5))
        model.add(Activation('softmax'))
        # Setting Loss ,Optimiser for model
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        # Training model
        hist = model.fit(embedding_mat_train, train, epochs=50, batch_size=32, shuffle=True)
        pred = model.predict_classes(embedding_mat_test)
        # Printing the sentences with the predicted and the labelled emoji
        test = pd.read_csv('test_emoji.csv', header=None)
        for ix in range(embedding_mat_test.shape[0]):

            if pred[ix] != testset[ix]:
                print(ix)
                print(test[0][ix], end=" ")
                print(emoji.emojize(emoji_d[pred[ix]], use_aliases=True), end=" ")
                print(emoji.emojize(emoji_d[testset[ix]], use_aliases=True))

    def glove_query_convert(self, s, embeddings_index):
        s = s.split()
        np.unique(np.array([len(s)]), return_counts=True)
        matrix_test = np.zeros((1, 10, 50))
        for ij in range(len(s)):
            matrix_test[0][ij] = embeddings_index[s[ij].lower()]
        return matrix_test, s

    def single_query_model_LSTM(self, embedding_mat_train, train):
        model_LSTM = Sequential()
        model_LSTM.add(LSTM(128, input_shape=(10, 50), return_sequences=True))
        model_LSTM.add(Dropout(0.5))
        model_LSTM.add(LSTM(128, return_sequences=False))
        model_LSTM.add(Dropout(0.5))
        model_LSTM.add(Dense(5))
        model_LSTM.add(Activation('softmax'))
        # Setting Loss ,Optimiser for model
        model_LSTM.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        # Training model
        hist = model_LSTM.fit(embedding_mat_train, train, epochs=100, batch_size=32, shuffle=True)
        return model_LSTM

    def single_query_model_RNN(self, embedding_mat_train, train):
        # A simple RNN network to classify the emoji class from an input Sentence

        model_RNN = Sequential()
        model_RNN.add(SimpleRNN(64, input_shape=(10, 50), return_sequences=True))
        model_RNN.add(Dropout(0.5))
        model_RNN.add(SimpleRNN(64, return_sequences=False))
        model_RNN.add(Dropout(0.5))
        model_RNN.add(Dense(5))
        model_RNN.add(Activation('softmax'))

        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        # Training of the model and Setting hyperparameters for the model
        hist = model.fit(embedding_mat_train, train, epochs=50, batch_size=32, shuffle=True)
        return model_RNN

    def predict_res(self, s, model_deep, matrix_test, emoji_d):
        predict = model_deep.predict_classes(matrix_test)
        print(s, end=" ")
        # print(predict[0])
        print(emoji.emojize(emoji_d[predict[0]], use_aliases=True), end="")
        print("\nDone..")
        # return predicate[0]
        # can return index of emoji,use emoji_dict


if __name__ == '__main__':
    print("Training Model------")
    model = Model()
    emoji_dic = model.emoji_dept()
    train_x, train_y, test_x, test_y = model.data_modeling()
    embed_index = model.universal_glove_file_read()
    embedding_matrix_train, embedding_matrix_test = model.glove_file(train_x, test_x, embed_index)

    LSTM = model.single_query_model_LSTM(embedding_matrix_train, train_y)
    print("Training Model Done-----")
    c = "y"
    while c == "y":
        input_string = input(" ENTER THE QUERY")
        res_matrix, query = model.glove_query_convert(input_string, embed_index)
        model.predict_res(query, LSTM, res_matrix, emoji_dic)
        c = input("Do you want to provide query[y/n]")
    print(" Thank You")

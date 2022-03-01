import random
import json
import pickle
import numpy as np

import nltk
from nltk.stem import WordNetLemmatizer

from tensorflow.keras.models import load_model

lemmatizer = WordNetLemmatizer()
intents = json.loads(open('intents.json').read())

words = pickle.load(open('words.pkl', 'rb'))
classes = pickle.load(open('classes.pkl', 'rb'))
model = load_model('chatbot_model.model')


def clean_up_sentence(sentence):
    '''
    :param sentence: input sentence
    :return: list of lemmatized words, that are in the sentence
    '''
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word) for word in sentence_words]
    return sentence_words

def bag_of_words(sentence):
    '''
    :param sentence: input sentence
    :return: a "bag" for the sentence --> is word in sentence? 1|0
    '''
    sentence_words = clean_up_sentence(sentence)
    bag = [0] * len(words)
    for w in sentence_words:
        for i, word in enumerate(words):
            if word == w:
                bag[i] = 1
    return np.array(bag)


def predict_class(sentence):
    '''
    :param sentence: input sentece
    :return: list of (intents, probabilities) , sorted by probability high -> low
    '''
    bow = bag_of_words(sentence)
    res = model.predict(np.array([bow]))[0]
    ERROR_THRESHOLD = 0.25
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]

    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append({'intents': classes[r[0]], 'probability': str(r[1])})
    return return_list


def get_response(intents_list, intents_json):
    '''
    :param intents_list: predicted classes based on input
    :param intents_json: list of classes stored in json
    :return: a random result appropriate to predicted class
    '''
    tag = intents_list[0]['intents']
    list_of_intents = intents_json['intents']
    result = "I do not understand."     # if you cant recognize a class: which might be impossible but whatever

    for i in list_of_intents:
        if i['tag'] == tag:
            result = random.choice(i['responses'])
            break
    return result


if __name__ == '__main__':
    print("Ask me questions:")

    while True:
        message = input("")
        if message == "exit":
            print("Closing...")
            quit()

        ints = predict_class(message)
        res = get_response(ints, intents)
        print(res)
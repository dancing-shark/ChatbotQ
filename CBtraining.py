import random
import json
import pickle
import numpy as np
from nltk.stem import WordNetLemmatizer
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Dropout
from tensorflow.keras.optimizers import SGD
from configparser import ConfigParser
from master import KI

# config.ini
config_object = ConfigParser()
config_object.read("config.ini")
cb = config_object["CB"]

lemmatizer = WordNetLemmatizer()
ki = KI()

intents = json.loads(open("intents.json").read())

words = []
classes = []
documents = []
ignore_letters = ["?","!",".",","]

for intent in intents["intents"]:
    for pattern in intent["patterns"]:
        word_list = ki._removingStopWords(ki._lemmatizing(ki._partOfSpreechTagging(ki._sentence_tokenize(pattern))),ignore_letters)
        words.extend(word_list)
        documents.append((word_list, intent["tag"]))
        if intent["tag"] not in classes:
            classes.append(intent["tag"])

words = sorted(set(words))
classes = sorted(set(classes))

pickle.dump(words, open(cb["words"],"wb"))
pickle.dump(classes, open(cb["classes"],"wb"))

training = []
output_empty = [0]* len(classes)

for document in documents:
    bag =[]
    for word in words:
        bag.append(1) if word in document[0] else bag.append(0)
    output_row = list(output_empty)
    output_row[classes.index(document[1])] =1
    training.append([bag, output_row])

random.shuffle(training)

training = np.array(training)

train_x = list(training[:,0])
train_y = list(training[:,1])
print(train_x, train_y)

model = Sequential()
model.add(Dense(128, input_shape=(len(train_x[0]),), activation="relu"))
model.add(Dropout(0.5))
model.add(Dense(64, activation="relu"))
model.add(Dropout(0.5))
model.add(Dense(64, activation="relu"))
model.add(Dropout(0.5))
model.add(Dense(64, activation="relu"))
model.add(Dropout(0.5))
model.add(Dense(len(train_y[0]), activation="softmax"))

sgd = SGD(lr=0.01,decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss="categorical_crossentropy", optimizer=sgd, metrics=["accuracy"])

hist = model.fit(np.array(train_x),np.array(train_y), epochs = 100, batch_size=5, verbose=1)
model.save(cb["model"], hist)

print("Done")

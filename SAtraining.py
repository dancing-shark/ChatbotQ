from nltk.corpus import twitter_samples, stopwords
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import SGD
from tensorflow.keras import layers,utils
from master import KI
from configparser import ConfigParser
import re, string, random, pickle
import numpy as np


positive_tweets = twitter_samples.strings('positive_tweets.json')# Array aus Stings mit Tweets
negative_tweets = twitter_samples.strings('negative_tweets.json')# Array aus Stings mit Tweets

stop_words = stopwords.words('english')# Array aus Stings mit Wörtern

ki = KI() # NLP funktionen

# config.ini
config_object = ConfigParser()
config_object.read("config.ini")
sa = config_object["SA"]

classes = ["Positive","Negative"]
words = []
documents = []

# Data vorbereiten 
positive_cleaned_tokens_list = []
negative_cleaned_tokens_list = []

for tweet in positive_tweets: # 2D-Array [["token","token"],["token"]]
    tokens = ki._sentence_tokenize(tweet)
    cleaned_tokens = []
    for token in tokens:
        token = re.sub('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+#]|[!*\(\),]|'\
                      '(?:%[0-9a-fA-F][0-9a-fA-F]))+','', token)
        token = re.sub("(@[A-Za-z0-9_]+)","", token)
        token = re.sub("http|","", token)
        cleaned_tokens.append(token)
    cleaned_tokens = ki._lemmatizing(ki._partOfSpreechTagging(cleaned_tokens))
    tok = []
    for token in cleaned_tokens:
        if len(token) > 0 and token not in string.punctuation and token.lower() not in stop_words:
            tok.append(token.lower())
    positive_cleaned_tokens_list.append(tok)
    
for tweet in negative_tweets: # 2D-Array [["token","token"],["token"]]
    tokens = ki._sentence_tokenize(tweet)
    cleaned_tokens = []
    for token in tokens:
        token = re.sub('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+#]|[!*\(\),]|'\
                      '(?:%[0-9a-fA-F][0-9a-fA-F]))+','', token)
        token = re.sub("(@[A-Za-z0-9_]+)","", token)
        token = re.sub("http|","", token)
        cleaned_tokens.append(token)
    cleaned_tokens = ki._lemmatizing(ki._partOfSpreechTagging(cleaned_tokens))
    tok = []
    for token in cleaned_tokens:
        if len(token) > 0 and token not in string.punctuation and token.lower() not in stop_words:
            tok.append(token.lower())
    negative_cleaned_tokens_list.append(tok)
    
for tweet in positive_cleaned_tokens_list+negative_cleaned_tokens_list:
    words.extend(tweet)

words = sorted(set(words)) # Vocabulary building
pickle.dump(words, open(sa["words"], "wb"))
pickle.dump(classes, open(sa["classes"], "wb"))

positive_dataset = [(tweet_dict, "Positive") for tweet_dict in positive_cleaned_tokens_list]
negative_dataset = [(tweet_dict, "Negative") for tweet_dict in negative_cleaned_tokens_list]
documents = positive_dataset + negative_dataset

training = []
output_empty = [0]* len(classes)

for document in documents: #Encodeing
    bag =[]
    word_patterns = document[0]
    for word in words:
        bag.append(1) if word in word_patterns else bag.append(0)
    
    output_row = list(output_empty)
    output_row[classes.index(document[1])] =1
    training.append([bag, output_row])

random.shuffle(training)

training = np.array(training)

train_x = list(training[:,0])
train_y = list(training[:,1])

model = Sequential()
model.add(layers.Dense(128, input_shape=(len(train_x[0]),), activation="relu"))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(64, activation="relu"))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(64, activation="relu"))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(64, activation="relu"))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(len(train_y[0]), activation="softmax"))

sgd = SGD(lr=0.01,decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss="categorical_crossentropy", optimizer=sgd, metrics=["accuracy"])

hist = model.fit(np.array(train_x),np.array(train_y), epochs = 100, batch_size=5, verbose=1)

model.save(sa["model"], hist)
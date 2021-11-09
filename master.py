import nltk
from nltk.corpus import wordnet as wn
import re
import numpy as np
from tensorflow.keras.models import load_model
import pickle
import json
import random

class KI:
    def __init__(self, config=None):
        if config is not None:
            self.model = load_model(config["model"])
            self.words = pickle.load(open(config["words"], "rb"))
            self.classes = pickle.load(open(config["classes"], "rb"))
        
    def _sentence_tokenize(self,sentence):
        return nltk.word_tokenize(sentence, language='english')

    def _partOfSpreechTagging(self,tokens):
        return nltk.pos_tag(tokens)

    def _lemmatizing(self,tags):
        lemmatizer = nltk.WordNetLemmatizer()
        lemmata = []
        for (word,pos) in tags:
            if pos in ['JJ', 'JJR', 'JJS']:
                pos = wn.ADJ
            elif pos in ['NN', 'NNS', 'NNP', 'NNPS']:
                pos =  wn.NOUN
            elif pos in ['RB', 'RBR', 'RBS']:
                pos =  wn.ADV
            elif pos in ['VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ']:
                pos =  wn.VERB
            else:
                pos =  None

            if pos == None:
                lemmata.append(word)
            else:
                lemmata.append(lemmatizer.lemmatize(word,pos))
        return lemmata

    def _removingStopWords(self, lemmata, stopwords=[]):
        stop_words = nltk.corpus.stopwords.words('english')
        stop_words.extend(stopwords)
        return [word for word in lemmata if word not in stop_words]

    def bag_of_words(self, sentence): # Encoding - Text to Number Array
        sentence_words = self._removingStopWords(self._lemmatizing(self._partOfSpreechTagging(self._sentence_tokenize(sentence))))
        bag = [0]* len(self.words)
        for w in sentence_words:
            for i, word in enumerate(self.words):
                if word == w:
                    bag[i] = 1
        return np.array(bag)

    def run(self, sentence, var): # main methode
        """Possible return values: op, opin, all
        Args:
            sentence (str): sentence to be processed
            var (str): return value type
        """
        bow = self.bag_of_words(sentence)
        res = self.model.predict(np.array([bow]))[0]
        ERROR_THRESHOLD = 0.25
        results = [[i,r] for i,r in enumerate(res) if r > ERROR_THRESHOLD]
        results.sort(key=lambda x: x[1], reverse=True)
        retrun_list = []
        for r in results:
            retrun_list.append({"intent": self.classes[r[0]], "probability":str(r[1])})
        if var=="op":
            retrun_list = retrun_list[0]["intent"]
        elif var=="opin":
            retrun_list = retrun_list[0]
        elif var=="all":
            retrun_list = retrun_list
        return retrun_list


class SentimentAnalyzer(KI):
    def __init__(self, config):
        super().__init__(config)

    def message(self, sentence, var): # sentence processing ...
        return self.run(sentence, var)
    

class ConversationKI(KI):
    def __init__(self, config):
        super().__init__(config)
        self.intents = json.loads(open("intents.json").read())

    def message(self, sentence, var): # sentence processing ...
        tags = self._partOfSpreechTagging(self._sentence_tokenize(sentence))
        nppTags = [i[0] for i in tags if i[1]=="NNP"]
        if len(nppTags)==1:
            sentence = sentence.replace(str(nppTags[0]),"xxx")
        elif len(nppTags)==2:
            sentence = sentence.replace(str(nppTags[0])+" ","")
            sentence = sentence.replace(nppTags[1],"xxx")

        return self.run(sentence, var), nppTags
    
    def get_responses(self, tag): # get all responses from tag
        for i in self.intents["intents"]:
            if i["tag"]==tag:
                return [i[1] for i in i["responses"]]

    def get_response_by_sentimentfactor(self, tag, sentimentfactor): # get one respons from tag with matching factor
        list_of_resposes = []
        for i in self.intents["intents"]:
            if i["tag"]==tag:
                list_of_resposes.extend(i["responses"])
        
        results = []
        for i in list_of_resposes:
            if i[0]==sentimentfactor:
                results.append(i[1])
        return random.choice(results)
        


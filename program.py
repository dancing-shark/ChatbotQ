from master import ConversationKI, SentimentAnalyzer
from configparser import ConfigParser
import datetime
import csv
import pyjokes
import wikipedia
import random

class Program:
    def __init__(self):
        # config.ini
        self.config_object = ConfigParser()
        self.config_object.read("config.ini")
    
        self.cKI = ConversationKI(dict(self.config_object["CB"]))
        self.sKI = SentimentAnalyzer(dict(self.config_object["SA"]))

    def message(self, message):
        cat, nppTags = self.cKI.message(message, "op") # get category by ConversationAI
        sa = self.sKI.message(message, "op") # get sentiment by SentimentAI
        
        # additional functions
        if cat=="password":
            self.reset_password()
        elif cat=="ticket":
            self.create_ticket()
        elif cat=="db":
            self.search_in_db()
        elif cat=="joke":
            return "Okay I tell you one:\n"+self.tell_a_joke()
        elif cat=="time":
            return "The time is "+self.tell_the_time()
        elif cat=="wiki":
            wiki = self.tell_the_wiki(nppTags)
            return ["No Page like "+" ".join(nppTags)+" found. Try another one." if wiki==None else random.choice(self.cKI.get_responses(cat))+"\n"+wiki][0]

        response = str(self.cKI.get_response_by_sentimentfactor(cat,sa))
        self.protocol(message, response, cat, sa)
        return response+"\n 🅚🅐🅣: "+str(cat)+", 🅢🅐: "+str(sa)

    def protocol(self, message, response, tag, sentimentfactor):
        protresoult = [str(datetime.datetime.now()),message,response,tag,sentimentfactor]
        with open("protocol.csv","a",newline='', encoding='UTF8') as prot:
            writer = csv.writer(prot)
            writer.writerow(protresoult)

    def reset_password(self):
        pass

    def create_ticket(self):
        pass

    def search_in_db(self):
        pass

    def tell_a_joke(self):
        return pyjokes.get_joke(language="en", category="all")

    def tell_the_time(self):
        return datetime.datetime.now().strftime("%H:%M:%S")

    def tell_the_wiki(self, nppTags):
        try:
            return wikipedia.summary(" ".join(nppTags), sentences=1)
        except:
            return None


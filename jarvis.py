#!/usr/bin/env python
# -*- coding: utf-8 -*-

# jarvis.py
# [yzhang27]

import websocket
import pickle
import json
import urllib
import requests
import sqlite3
import sklearn # you can import other sklearn stuff too!
# FILL IN ANY OTHER SKLEARN IMPORTS ONLY
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import Pipeline

import botsettings # local .py, do not share!!
TOKEN = botsettings.API_TOKEN
DEBUG = False


def debug_print(*args):
    if DEBUG:
        print(*args)


try:
    conn = sqlite3.connect("jarvis.db")
except:
    debug_print("Can't connect to sqlite3 database...")


def post_message(message_text, channel_id):
    requests.post("https://slack.com/api/chat.postMessage?token={}&channel={}&text={}&as_user=true".format(TOKEN,channel_id,message_text))


class Jarvis():
    
    def __init__(self): # initialize Jarvis
        self.JARVIS_MODE = None
        self.ACTION_NAME = None
        
        # SKLEARN STUFF HERE:
        self.BRAIN = Pipeline([('vect', CountVectorizer()),
                               ('tfidf', TfidfTransformer()),
                               ('clf', MultinomialNB()),])
    
        self.BRAIN_SVM = Pipeline([('vect', CountVectorizer()),
                               ('tfidf', TfidfTransformer()),
                               ('clf', SGDClassifier()),])
   
    def extract_text_lable(self, c):
        text = []
        lable = []
        for row in c.execute("SELECT * from training_data"):
            text.append(row[1])
            lable.append(row[2])
        return text, lable
    
    def on_message(self, ws, message):
        m = json.loads(message)
        debug_print(m, self.JARVIS_MODE, self.ACTION_NAME)

        # only react to Slack "messages" not from bots (me):
        if m['type'] == 'message' and 'bot_id' not in m:
            
            c = conn.cursor()            
            
            if m['text'] == 'done':
                if self.JARVIS_MODE == 'trainingMode': 
                    post_message("Ok, I'm finished training.", m['channel'])
                elif self.JARVIS_MODE == 'testingMode':
                    post_message("Ok, I'm finished testing.", m['channel'])
                self.JARVIS_MODE = None 
                self.ACTION_NAME = None
                
            elif 'training time' in m['text']:
                post_message("Ok, I'm ready for training. What NAME should this ACTION be?", m['channel'])
                self.JARVIS_MODE = 'trainingMode'
                
            elif 'testing time' in m['text']:
                post_message("Ok, I'm ready for testing. Write me something and I will figure it out.", m['channel'])
                self.JARVIS_MODE = 'testingMode'
            
            elif self.JARVIS_MODE == 'trainingMode':
                if self.ACTION_NAME == None:
                    self.ACTION_NAME = m['text']
                    post_message("OK, let's call this action `{}`. Give me some training text now!".format(self.ACTION_NAME), m['channel'])
                else:
                    # write training text into database
                    msg_txt = m['text']
                    action  = self.ACTION_NAME
                    c.execute("INSERT INTO training_data (txt,action) VALUES (?, ?)", (msg_txt, action,))
                    conn.commit()
                    post_message("OK, I've got it. What else?", m['channel'])
                    
            elif self.JARVIS_MODE == 'testingMode':
                msg_txt = [m['text']]
                text, lable = self.extract_text_lable(c)
                self.BRAIN.fit(text, lable)
                predicted_lable = self.BRAIN.predict(msg_txt)
                post_message("OK, I think the action you mean is `{}`".format(predicted_lable[0]), m['channel'])
                post_message("Write me something else and I will figure it out.", m['channel'])
                
            elif 'export' in m['text']:
                pickle.dump(self.BRAIN, open('jarvis_brain.pkl', 'wb'))
                pickle.dump(self.BRAIN_SVM, open('jarvis_brain_svm.pkl', 'wb'))
                

        
def start_rtm():
    """Connect to Slack and initiate websocket handshake"""
    r = requests.get("https://slack.com/api/rtm.start?token={}".format(TOKEN), verify=False)
    r = r.json()
    r = r["url"]
    return r


def on_error(ws, error):
    print("SOME ERROR HAS HAPPENED", error)


def on_close(ws):
    conn.close()
    print("Web and Database connections closed")


def on_open(ws):
    print("Connection Started - Ready to have fun on Slack!")


r = start_rtm()
jarvis = Jarvis()
ws = websocket.WebSocketApp(r, on_message=jarvis.on_message, on_error=on_error, on_close=on_close)
ws.run_forever()



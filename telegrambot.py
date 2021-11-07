from configparser import ConfigParser
from telegram.ext import Updater, CommandHandler, MessageHandler, Filters
import json
from program import Program

# config.ini
config_object = ConfigParser()
config_object.read("config.ini")
token = config_object["Bot"]["token"] # get Telegram Bot token

# read content.json
json_content=dict
with open('content.json', 'r') as j:
    json_content = json.load(j) # get list of answers

pro = Program() # start main program

def start(update, context): # runs if command == start
    update.message.reply_text(json_content["start"])

def help(update, context): # runs if command == help
    update.message.reply_text(json_content["help"])

def contact(update, context): # runs if command == contact
    update.message.reply_text(json_content["contact"])

def content(update, context): # runs if command == content
    update.message.reply_text(json_content["content"])

def message(update, context): # runs by incoming message
    update.message.reply_text(pro.message(update.message.text))

updater = Updater(token) # create the bot

dispatcher = updater.dispatcher # get the dispatcher to register handlers

# different commands - answer in Telegram
dispatcher.add_handler(CommandHandler("start", start)) # reply by first incoming message
dispatcher.add_handler(CommandHandler("help", help)) # reply help
dispatcher.add_handler(CommandHandler("contact", contact)) # reply contact
dispatcher.add_handler(CommandHandler("content", content)) # reply content
dispatcher.add_handler(MessageHandler(Filters.text, message)) # reply message

updater.start_polling() # start the bot

updater.idle() # run the Bot until you press Ctrl-C
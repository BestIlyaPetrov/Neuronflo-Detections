"""
Developed by Sialoi Taa 
8/15/2023

Telegram bot object

To use this class, initialize it with a valid API token
and give it a name. After initialized, you can use the 
4 methods and 3 attributes to manipulate the telegram
bot object to any way desired.

Another way you can use this file is to use the 
individual functions. 
"""

import requests

class teleBot:
    def __init__(self, API_TOKEN, name):
        self.API_TOKEN = API_TOKEN
        self.Name = name
        self.CHAT_ID = self.teleChatID()
        if self.CHAT_ID == None:
            print(f"{self.Name} has an invalid chat ID. Please review the instructions again and make sure API token is correct.")

    def teleMessage(self, message):
        """
        This function takes in any text message as input and will send it to any chat with
        a valid chat ID. Make sure that the token and chat ID are valid and you are using
        the correct one!
        """

        if self.CHAT_ID == None:
            print(f"Invalid chat ID when sending a message! - {self.Name}")
            return self.CHAT_ID

        send_text = 'https://api.telegram.org/bot' + str(self.API_TOKEN) + '/sendMessage?chat_id=' + str(self.CHAT_ID) + \
                    '&parse_mode=MarkdownV2&text=' + str(message)
        response = requests.get(send_text)
        return response.json()

    def teleImage(self, file_path):
        """
        This method will take the file pathway for any image and be able to send to any
        Telegram chat with a valid API token and chat ID. Make sure that the bot is added to the chat
        before you try to send anything through the bot.
        """
        if self.CHAT_ID == None:
            return f"Invalid chat ID when sending an image! - {self.Name}"

        files = {'photo':open(file_path, 'rb')}    
        resp = requests.post('https://api.telegram.org/bot' + str(self.API_TOKEN) + '/sendPhoto?chat_id=' + str(self.CHAT_ID), files=files)
        status = str(resp.status_code)

        if status == '200':
            status = 'Image sent successfully!'
        else:
            status = "Something went wrong with the image!"
        
        return status

    def teleChatID(self):
        """
        This method allows the telegram bot to find and return the
        chat ID of the specified bot.
        """

        UPDATE_URL = 'https://api.telegram.org/bot' + str(self.API_TOKEN) + '/getUpdates'
        response = requests.get(UPDATE_URL, verify=False)
        
        if response.status_code == 200: # HTTP 200 means the request was successful.
            try:
                resp = response.json()
                if resp['ok'] == False:
                    return None
                chatID = resp['result'][0]['message']['chat']['id']
                return chatID
            except:
                print("Invalid JSON received:", response.text)
        else:
            print("Request failed with status code:", response.status_code)
            print("Response:", response.text)

        
       
    
    def change_API(self, API):
        self.API_TOKEN = API
        self.CHAT_ID = self.teleChatID()
        if self.CHAT_ID == None:
            print(f"{self.Name} has an invalid chat ID. Please review the instructions again and make sure API token is correct.")
            return False
        return True

def teleMessage(API_TOKEN, message, Name):
    """
    This function takes in any text message as input, with API key and name, and will 
    send it to any chat with a valid chat ID. Make sure that the token and chat ID are 
    valid and you are using the correct one!
    """

    CHAT_ID = teleChatID(API_TOKEN)

    if CHAT_ID == None:
        return f"Invalid chat ID when sending a message! - {Name}"

    send_text = 'https://api.telegram.org/bot' + str(API_TOKEN) + '/sendMessage?chat_id=' + str(CHAT_ID) + \
                '&parse_mode=MarkdownV2&text=' + str(message)
    response = requests.get(send_text)
    return response.json()

def teleChatID(API_TOKEN):
    """
    This function allows the telegram bot to find and return the
    chat ID of the specified bot.
    """

    UPDATE_URL = 'https://api.telegram.org/bot' + str(API_TOKEN) + '/getUpdates'
    response = requests.get(UPDATE_URL)
    resp = response.json()
    if resp['ok'] == False:
        return None
    chatID = resp['result'][0]['message']['chat']['id']
    return chatID

def teleImage(Name, API_TOKEN, file_path):
    """
    This function will take the file pathway for any image and be 
    able to send to any Telegram chat with a valid API token and 
    chat ID. Make sure that the bot is added to the chat before 
    you try to send anything through the bot.
    """

    CHAT_ID = teleChatID(API_TOKEN)

    if CHAT_ID == None:
        print(f"Invalid chat ID when sending an image! - {Name}")
        return CHAT_ID

    files = {'photo':open(file_path, 'rb')}    
    resp = requests.post('https://api.telegram.org/bot' + str(API_TOKEN) + '/sendPhoto?chat_id=' + str(CHAT_ID), files=files)
    status = resp.status_code

    if status == '200':
        status = 'Image sent successfully!'
    else:
        status = "Something went wrong with the image!"
    
    return status

# Example code
if __name__ == '__main__': 
    api = '6323749554:AAEAA_qF1dDE-UWlTr9nxlqlj_pmZbNOqSY'
    Turry = teleBot(API_TOKEN=api, name="Turry")
    resp = Turry.teleMessage(message=f"Object Testing 3 from {Turry.Name}")
    print(resp)

"""
Below is an example of what response.json could look like.
{"ok":true,
"result":[{"update_id":503695529,
            "message":{"message_id":23,
                    "from":{"id":5985749609,
                            "is_bot":false,
                            "first_name":"John",
                            "last_name":"Fields",
                            "language_code":"en"
                            },
                    "chat":{"id":-876132137,
                            "title":"NeuronFlo",
                            "type":"group",
                            "all_members_are_administrators":true
                            },
                    "date":1692149482,
                    "text":"/start",
                    "entities":[{"offset":0,
                                    "length":6,
                                    "type":"bot_command"
                                }]
                    }
        }]
}
"""
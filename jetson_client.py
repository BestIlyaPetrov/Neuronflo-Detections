import paho.mqtt.publish as publish
import random
import time
import datetime
import pytz
import json


# Define the username and password for the connection
username = "ilya-jetson"
password = "jetson-ai"
hostname = "142.93.72.136"
topic = "jetson/data"

# n = 3

# for i in range(1,n+1): #sends 3 messages
#     tz = pytz.timezone('UTC')  # set the desired timezone (e.g., UTC)
#     started = tz.localize(datetime.datetime.now())  # make the current datetime object timezone-aware
#     ended = started + datetime.timedelta(minutes=1)

#     started_str = started.strftime('%Y-%m-%dT%H:%M:%SZ')  # format as string
#     ended_str = ended.strftime('%Y-%m-%dT%H:%M:%SZ')  # format as string
    


#     data = {
#         "station": 1,
#         "started": started_str,
#         "ended": ended_str,
#         "duration": "01:00:00",
#         "video": "null",
#         "ppe_missing": "['hard hat', 'safety glasses']"
#         }

#     json_data = json.dumps(data)
#     print("Sending:")
#     print(json_data)
#     print("")

#     publish.single("jetson/data", json_data, hostname=hostname, auth={'username':username, 'password':password})
#     time.sleep(10)

# print(f"Sent {n+1} messages")


def send_update_MQTT(state):
    tz = pytz.timezone('UTC')  # set the desired timezone (e.g., UTC)
    timestamp = tz.localize(datetime.datetime.now())  # make the current datetime object timezone-aware
    timestamp_str = timestamp.strftime('%Y-%m-%dT%H:%M:%SZ')

    data = {
        "station": 1,
        "timestamp": timestamp_str,
        "video": "null",
        "state": state
        }
    client_send(data)

    return

        
        



def client_send(data):
    json_data = json.dumps(data)
    try:
        publish.single(topic, json_data, hostname=hostname, auth={'username':username, 'password':password})
        print("Sent the following to the server:")
        print(json_data)
        return
    except: 
        print("Couldn't publish data, something went wrong")
        return

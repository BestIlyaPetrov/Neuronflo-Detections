"""
Class for Jetson Nano to integrate all its functions
"""

import os
import sys
import datetime
import pytz
import json
import dotenv
import paho.mqtt.publish as publish
import Jetson.GPIO as GPIO
import constants as c

# Load credentials from .env file
dotenv.load_dotenv()

class Jetson:
    def __init__(self) -> None:
        GPIO.setmode(GPIO.BOARD)

        self.red = c.RED_LED
        self.green = c.GREEN_LED
        self.blue = c.BLUE_LED
        self.colors = {"red" : self.red, "green" : self.green, "blue" : self.blue}

        self.username = os.getenv("user_name")
        self.password = os.getenv("password")
        self.hostname = os.getenv("host_name")
        self.topic = os.getenv("topic")

        # Setup the pins as outputs
        GPIO.setup(self.red, GPIO.OUT)
        GPIO.setup(self.green, GPIO.OUT)
        GPIO.setup(self.blue, GPIO.OUT)

    def send_update_MQTT(self, state) -> None:
        """
        Sends an MQTT message to the broker with the current state
        """
        try:
            tz = pytz.timezone('UTC')  # set the desired timezone (e.g., UTC)
            timestamp = tz.localize(datetime.datetime.now())  # make the current datetime object timezone-aware
            timestamp_str = timestamp.strftime('%Y-%m-%dT%H:%M:%SZ')

            data = {
                "station": 1,
                "timestamp": timestamp_str,
                "video": "null",
                "state": state
                }
            self.client_send(data)
        except Exception as e:
            print(f"Unable to send MQTT message, Error: {e}")

    def client_send(self, data) -> None:
        """
        Sends the data to the broker
        """
        try:
            json_data = json.dumps(data)
            print("Sending:")
            print(json_data)
            print("")

            publish.single(self.topic, json_data, hostname=self.hostname, auth={'username':self.username, 'password':self.password})
        except Exception as e:
            print(f"Unable to send MQTT message, Error: {e}")

    def light_up_RGB_LED(self, color) -> None:
        """
        Turns on the RGB LED to the specified color
        """
        try:
            if color in self.colors:
                GPIO.output(self.colors[color], GPIO.HIGH)
                [GPIO.output(self.colors[col], GPIO.LOW) for col in self.colors if col != color]
            else:
                [GPIO.output(self.colors[col], GPIO.LOW) for col in self.colors]

        except Exception as e:
            print(f"Unable to light up LEDs, Error: {e}")

    

if __name__ == '__main__':
    # Check if the user has provided a color
    if len(sys.argv) != 2:
        print("Usage: python jetson.py <color>")
        sys.exit(1)

    jetson = Jetson()
    jetson.light_up_RGB_LED(sys.argv[1])
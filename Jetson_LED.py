import Jetson.GPIO as GPIO
import sys
# Pin Definitions
red_pin = 33  # board pin number
green_pin = 35
blue_pin = 37

# Set the pin numbering scheme
GPIO.setmode(GPIO.BOARD)

# Setup the pins as outputs
GPIO.setup(red_pin, GPIO.OUT)
GPIO.setup(green_pin, GPIO.OUT)
GPIO.setup(blue_pin, GPIO.OUT)

def light_up_RGB_LED(color):
    if color == "red":
        # Turn on the red LED
        GPIO.output(red_pin, GPIO.HIGH)
        # Turn off the green and blue LEDs
        GPIO.output(green_pin, GPIO.LOW)
        GPIO.output(blue_pin, GPIO.LOW)
    elif color == "green":
        # Turn on the green LED
        GPIO.output(green_pin, GPIO.HIGH)
        # Turn off the red and blue LEDs
        GPIO.output(red_pin, GPIO.LOW)
        GPIO.output(blue_pin, GPIO.LOW)
    elif color == "blue":
        # Turn on the blue LED
        GPIO.output(blue_pin, GPIO.HIGH)
        # Turn off the green and red LED
        GPIO.output(red_pin, GPIO.LOW)
        GPIO.output(green_pin, GPIO.LOW)
    else:
        GPIO.output(blue_pin,GPIO.LOW)
        GPIO.output(red_pin,GPIO.LOW)
        GPIO.output(green_pin,GPIO.LOW)

# use this if u want to run it from command line
if __name__ == '__main__':
    light_up_RGB_LED(sys.argv[1])
    #globals()(sys.argv[1])sys.argv[2])

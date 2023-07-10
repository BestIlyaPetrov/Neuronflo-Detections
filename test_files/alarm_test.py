import Jetson.GPIO as GPIO
import sys
import time
# Pin Definitions
pin = 29  # board pin number

# Set the pin numbering scheme
GPIO.setmode(GPIO.BOARD)

# Setup the pins as outputs
GPIO.setup(pin, GPIO.OUT)

def test_alarm(qty):
    for i in range(0, qty):
        GPIO.output(pin, GPIO.HIGH)
        time.sleep(0.25)
        GPIO.output(pin, GPIO.LOW)
        time.sleep(0.25)
    # Clean up the GPIO channel
    GPIO.cleanup()
   

# use this if u want to run it from command line
if __name__ == '__main__':
    qty = int(sys.argv[1])
    test_alarm(qty)
    # alarm_manual(sys.argv[1])
    #globals()(sys.argv[1])sys.argv[2])

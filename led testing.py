python3 -c "
import RPi.GPIO as GPIO
import time

GPIO.setmode(GPIO.BCM)
GPIO.setup(16, GPIO.OUT)
GPIO.setup(26, GPIO.OUT)

print('Testing RED LED (GPIO 16)...')
GPIO.output(16, True)
time.sleep(2)
GPIO.output(16, False)

print('Testing GREEN LED (GPIO 26)...')
GPIO.output(26, True)
time.sleep(2)
GPIO.output(26, False)

GPIO.cleanup()
print('Done!')
"

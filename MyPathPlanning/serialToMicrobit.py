
import serial
import time

ser = serial.Serial('/dev/tty.usbmodem1302', 115200)



while 1:
    #Sending number
    ser.write(b'$GPS:174,')
    # print(ser.readline())
    time.sleep(0.5)
    # Sending String
    ser.write(b'$Direction:180,')
    # print(ser.readline())
    time.sleep(0.5)


ser.close()

# import serial
# import time

# ser = serial.Serial('/dev/tty.usbmodem1302', 115200)



# while 1:
#     #Sending number
#     ser.write(b'$GPS:273,')
#     # print(ser.readline())
#     time.sleep(0.1)
#     # Sending String
#     ser.write(b'$Direction:270,')
#     # print(ser.readline())
#     time.sleep(0.01)


# ser.close()




import serial
import time

ser = serial.Serial('/dev/tty.usbmodem1302', 115200)
from pymavlink import mavutil

# Start a connection listening on a UDP port
the_connection = mavutil.mavlink_connection('udpin:localhost:14445')

# Wait for the first heartbeat 
#   This sets the system and component ID of remote system for the link
the_connection.wait_heartbeat()
print("Heartbeat from system (system %u component %u)" % (the_connection.target_system, the_connection.target_component))



while 1:
    msg = the_connection.recv_match(type='GLOBAL_POSITION_INT', blocking=True)
    #Print the compass value
    print("Global Position: %s" % int(msg.hdg/100))
    
    #Sending number
    ser.write(b'$GPS:' + str(int(msg.hdg/100)).encode() + b',')
    # print(ser.readline())
    time.sleep(0.1)
    # Sending String
    ser.write(b'$Direction:270,')
    # print(ser.readline())
    time.sleep(0.01)


ser.close()
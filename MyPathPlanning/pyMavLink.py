
import time

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
    time.sleep(0.5)
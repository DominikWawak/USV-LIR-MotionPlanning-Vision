import paho.mqtt.client as mqtt
# pip install paho-mqtt
import json

# Callback when the connection to the broker is established
def on_connect(client, userdata, flags, rc):
    print("Connected with result code " + str(rc))
    # Subscribe to a channel
    client.subscribe("test/res")

# Callback when a message is received
def on_message(client, userdata, msg):
    print("Received message on channel " + msg.topic + ": " + str(msg.payload))
    m_decode = str(msg.payload.decode("utf-8", "ignore"))
    m_decode = m_decode.replace(",", ':')
    print("data Received", m_decode.split(":")[1])

    

# Create a new MQTT client
client = mqtt.Client()
client.on_connect = on_connect
client.on_message = on_message

# Connect to the Beebotte broker
client.username_pw_set("9h1rq3Hf5cxTeQcb2yTYK3N6", "m33rs3IoJWxT9eX01hoxTLIDfLtq3EWN")
client.connect("mqtt.beebotte.com", 1883, 60)


# Publish the payload to a channel
client.publish("test/res", 'Hello World')

# Wait for incoming messages
client.loop_forever()
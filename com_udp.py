import socket
import time

# Global dictionary to keep track of the last time a message with a certain prefix was sent
last_sent_time = {}

# Data of secondary LED strip and smoker:
UDP_LED_COUNT = 45
UDP_IP_ADDRESS = "192.168.1.152"  # musictolight-led1
UDP_PORT = 4210


def send_udp_message(ip_address, port, message):
    """Sends a UDP message to the given IP address and port."""
    # global last_sent_time

    # Ensure message is in bytes if it's not already
    if isinstance(message, str):
        message = message.encode('utf-8')

    # Create a UDP socket
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    try:
        # Send the message
        sock.sendto(message, (ip_address, port))
    except OSError as e:
        # print error message and continue
        print(f"Could not send the message: {e}")

    # Close the socket
    sock.close()

    # Update the last sent time for this message type
    # last_sent_time[message_prefix] = current_time

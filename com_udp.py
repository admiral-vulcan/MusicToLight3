import socket
import time

# Global dictionary to keep track of the last time a message with a certain prefix was sent
last_sent_time = {}


def send_udp_message(ip_address, port, message):
    """Sends a UDP message to the given IP address and port.

    Args:
        ip_address (str): The recipient's IP address.
        port (int): The recipient's port.
        message (str): The message to send.
    """
    global last_sent_time

    # Extract the first 4 characters to identify the message type
    message_prefix = message[:3]

    # Get the current time
    current_time = time.time()

    # Check if we can send this message now
    if message_prefix not in last_sent_time or (current_time - last_sent_time.get(message_prefix, 0)) >= 0.1:
        # print(message)
        # Create a UDP socket
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

        # Send the message
        sock.sendto(bytes(message, "utf-8"), (ip_address, port))

        # Close the socket
        sock.close()

        # Update the last sent time for this message type
        last_sent_time[message_prefix] = current_time

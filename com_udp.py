import socket
import threading
from functools import wraps

# Wörterbuch, das Locks für jede IP-Adresse speichert
ip_locks = {}

# Data of secondary LED strip and smoker:
UDP_LED_COUNT = 45
UDP_IP_ADDRESS = "192.168.1.152"  # musictolight-led1
UDP_PORT = 4210


def synchronized_ip(func):
    @wraps(func)
    def wrapper(ip_address, *args, **kwargs):
        # Stelle sicher, dass für jede IP ein Lock existiert
        if ip_address not in ip_locks:
            ip_locks[ip_address] = threading.Lock()

        with ip_locks[ip_address]:
            # Nachricht senden
            result = func(ip_address, *args, **kwargs)
        return result

    return wrapper


@synchronized_ip
def send_udp_message_prepare(ip_address, port, message):
    """Sends a UDP message to the given IP address and port."""
    if isinstance(message, str):
        message = message.encode('utf-8')

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        sock.sendto(message, (ip_address, port))
    except OSError as e:
        print(f"Could not send the message: {e}")
    finally:
        sock.close()


# Optional: Funktion zum Senden der Nachricht in einem Thread
def send_udp_message(ip_address, port, message):
    thread = threading.Thread(target=send_udp_message_prepare, args=(ip_address, port, message))
    thread.start()

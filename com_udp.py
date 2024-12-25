# MusicToLight3  Copyright (C) 2024  Felix Rau.
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

import socket
import threading
import struct
from functools import wraps

# Wörterbuch, das Locks für jede IP-Adresse speichert
ip_locks = {}

# Data of secondary LED strip and smoker:
UDP_LED_COUNT = 45
UDP_IP_ADDRESS_LED1 = "192.168.1.152"  # musictolight-led1 (Beamer)
UDP_IP_ADDRESS_LED2 = "192.168.1.154"  # musictolight-led2 (SpectrumAnalyzer)
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


def send_spectrum_analyzer_data(ip_address, mode, intensity, color_start, color_end, num_leds_list):
    # Stelle sicher, dass num_leds_list genau 12 Werte enthält
    if len(num_leds_list) != 12:
        raise ValueError("num_leds_list muss 12 Werte enthalten.")

    # Erstellen des Datenpakets
    # Verwende 'B' für einen einzelnen Byte-Wert und '12B' für die 12 Byte-Werte der num_leds_list
    message = struct.pack('BBBB12B', mode, intensity, color_start, color_end, *num_leds_list)

    # Debug-Ausgabe, um zu sehen, wie das Paket aussieht
    # print("Gesendetes Paket: ", list(message))

    # Senden des UDP-Pakets
    send_udp_message(ip_address, UDP_PORT, message)



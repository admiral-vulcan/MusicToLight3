# MusicToLight3  Copyright (C) 2023  Felix Rau.
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

import redis
import json
from helpers import *

# Setting up communication with web server via Redis
redis_client = redis.StrictRedis(host='localhost', port=6379, db=0)
redis_client.set('strobe_mode', 'auto')
redis_client.set('smoke_mode', 'auto')
redis_client.set('panic_mode', 'off')
redis_client.set('play_videos_mode', 'auto')

strobe_mode = (redis_client.get('strobe_mode') or b'').decode('utf-8')
smoke_mode = (redis_client.get('smoke_mode') or b'').decode('utf-8')
panic_mode = (redis_client.get('panic_mode') or b'').decode('utf-8')
play_videos = (redis_client.get('play_videos_mode') or b'').decode('utf-8')

# Set master colors (TODO should later be changeable via web interface)
st_color_name = "blue"
nd_color_name = "red"
st_prim_color = get_rgb_from_color_name(st_color_name)
nd_prim_color = get_rgb_from_color_name(nd_color_name)

# Translate to RGB-Bytes
st_r, st_g, st_b = st_prim_color
nd_r, nd_g, nd_b = nd_prim_color

# Calculate the average of the two colors
average_color = tuple((a + b) // 2 for a, b in zip(st_prim_color, nd_prim_color))

# Find the highest value in the average
max_val = max(average_color)

# Calculate the factor to scale the highest value to 255
if max_val != 0:  # Prevent division by zero
    factor = 255 / max_val
else:
    factor = 0

# Multiply each color value in the average by the factor
secondary_color = tuple(int(val * factor) for val in average_color)

redis_client.set('st_prim_color', json.dumps(st_prim_color))
redis_client.set('nd_prim_color', json.dumps(nd_prim_color))
redis_client.set('secondary_color', json.dumps(secondary_color))


def redis_get_colors():
    """
    Retrieve the color values from Redis and assign them to the global variables.
    """
    global st_prim_color
    global nd_prim_color
    global secondary_color
    global st_r, st_g, st_b
    global nd_r, nd_g, nd_b

    st_prim_color = tuple(json.loads(redis_client.get('st_prim_color')))
    nd_prim_color = tuple(json.loads(redis_client.get('nd_prim_color')))
    secondary_color = tuple(json.loads(redis_client.get('secondary_color')))

    # Translate to RGB-Bytes
    st_r, st_g, st_b = st_prim_color
    nd_r, nd_g, nd_b = nd_prim_color


def get_gui_commands():
    """
    Retrieve GUI commands from Redis and return them.

    Returns:
        dict: A dictionary containing the GUI commands.
    """
    redis_get_colors()
    gui_commands = {
        'strobe_mode': (redis_client.get('strobe_mode') or b'').decode('utf-8'),
        'smoke_mode': (redis_client.get('smoke_mode') or b'').decode('utf-8'),
        'panic_mode': (redis_client.get('panic_mode') or b'').decode('utf-8'),
        'play_videos': (redis_client.get('play_videos_mode') or b'').decode('utf-8'),
    }
    return gui_commands


# MusicToLight3  Copyright (C) 2025  Felix Rau.
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

from flask import Flask, render_template, request, jsonify
import os
import subprocess
import re
import redis

app = Flask(__name__, static_folder='static')
redis_client = redis.StrictRedis(host='localhost', port=6379, db=0)

def is_running(process):
    s = subprocess.Popen(["ps", "axw"],stdout=subprocess.PIPE)
    for x in s.stdout:
        if re.search(process, x.decode()):
            return True
    return False

@app.route('/')
def home():
    return render_template('home.html', strobe_mode=redis_client.get('strobe_mode').decode('utf-8') if redis_client.get('strobe_mode') else 'offline')

@app.route('/status')
def status():
    running = is_running('main.py')
    return jsonify(running=running)

@app.route('/start', methods=['POST'])
def start():
    subprocess.Popen(['sudo', '-u', 'felix', 'env', 'DISPLAY=:0', 'python3', '/musictolight/main.py'])
    return '', 204

@app.route('/stop', methods=['POST'])
def stop():
    subprocess.Popen(['sudo', 'pkill', '-2', '-f', 'main.py'])
    return '', 204

@app.route('/shutdown', methods=['POST'])
def shutdown():
    subprocess.Popen(['sudo', '/usr/sbin/shutdown', 'now'])
    return '', 204

@app.route('/reboot', methods=['POST'])
def reboot():
    subprocess.Popen(['sudo', '/usr/sbin/reboot', 'now'])
    return '', 204

@app.route('/calibrate/<mode>', methods=['POST'])
def calibrate(mode):
    valid_modes = ['on', 'off']
    if mode in valid_modes:
        redis_client.set('calibrate', mode)
        return jsonify(calibrate=mode)
    else:
        return jsonify(error="Invalid mode"), 400

@app.route('/set_panic_mode/<mode>', methods=['POST'])
def set_panic_mode(mode):
    valid_modes = ['on', 'off']
    if mode in valid_modes:
        redis_client.set('panic_mode', mode)
        return jsonify(panic_mode=mode)
    else:
        return jsonify(error="Invalid mode"), 400

@app.route('/get_panic_mode', methods=['GET'])
def get_panic_mode():
    current_mode = redis_client.get('panic_mode').decode('utf-8') if redis_client.get('panic_mode') else 'offline'
    return jsonify(panic_mode=current_mode)

@app.route('/set_prim_color/<color>', methods=['POST'])
def set_prim_color(color):
    valid_colors = ['white', 'red', 'yellow', 'purple', 'green', 'orange', 'blue', 'pink', 'cyan']
    if color in valid_colors:
        redis_client.set('st_color_name', color)
        return jsonify(st_color_name=color)
    else:
        return jsonify(error="Invalid color"), 400

@app.route('/get_prim_color', methods=['GET'])
def get_prim_color():
    current_color = redis_client.get('st_color_name').decode('utf-8') if redis_client.get('st_color_name') else 'blue'
    return jsonify(st_color_name=current_color)

@app.route('/set_seco_color/<color>', methods=['POST'])
def set_seco_color(color):
    valid_colors = ['white', 'red', 'yellow', 'purple', 'green', 'orange', 'blue', 'pink', 'cyan']
    if color in valid_colors:
        redis_client.set('nd_color_name', color)
        return jsonify(nd_color_name=color)
    else:
        return jsonify(error="Invalid color"), 400

@app.route('/get_seco_color', methods=['GET'])
def get_seco_color():
    current_color = redis_client.get('nd_color_name').decode('utf-8') if redis_client.get('nd_color_name') else 'red'
    return jsonify(nd_color_name=current_color)

@app.route('/set_chill_mode/<mode>', methods=['POST'])
def set_chill_mode(mode):
    valid_modes = ['on', 'off']
    if mode in valid_modes:
        redis_client.set('chill_mode', mode)
        return jsonify(chill_mode=mode)
    else:
        return jsonify(error="Invalid mode"), 400

@app.route('/get_chill_mode', methods=['GET'])
def get_chill_mode():
    current_mode = redis_client.get('chill_mode').decode('utf-8') if redis_client.get('chill_mode') else 'offline'
    return jsonify(chill_mode=current_mode)

@app.route('/set_play_videos_mode/<mode>', methods=['POST'])
def set_play_videos_mode(mode):
    valid_modes = ['auto', 'off']
    if mode in valid_modes:
        redis_client.set('play_videos_mode', mode)
        return jsonify(play_videos_mode=mode)
    else:
        return jsonify(error="Invalid mode"), 400

@app.route('/get_play_videos_mode', methods=['GET'])
def get_play_videos_mode():
    current_mode = redis_client.get('play_videos_mode').decode('utf-8') if redis_client.get('play_videos_mode') else 'offline'
    return jsonify(play_videos_mode=current_mode)

@app.route('/set_strobe_mode/<mode>', methods=['POST'])
def set_strobe_mode(mode):
    valid_modes = ['on', 'auto', 'off']
    if mode in valid_modes:
        redis_client.set('strobe_mode', mode)
        return jsonify(strobe_mode=mode)
    else:
        return jsonify(error="Invalid mode"), 400

@app.route('/get_strobe_mode', methods=['GET'])
def get_strobe_mode():
    current_mode = redis_client.get('strobe_mode').decode('utf-8') if redis_client.get('strobe_mode') else 'offline'
    return jsonify(strobe_mode=current_mode)

@app.route('/set_smoke_mode/<mode>', methods=['POST'])
def set_smoke_mode(mode):
    valid_modes = ['on', 'auto', 'off']
    if mode in valid_modes:
        redis_client.set('smoke_mode', mode)
        return jsonify(smoke_mode=mode)
    else:
        return jsonify(error="Invalid mode"), 400

@app.route('/get_smoke_mode', methods=['GET'])
def get_smoke_mode():
    current_mode = redis_client.get('smoke_mode').decode('utf-8') if redis_client.get('smoke_mode') else 'offline'
    return jsonify(smoke_mode=current_mode)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8472)

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

# Required Libraries
import os
import random
import time
import cv2
import pygame
import numpy as np
from helpers import *
from pygame.math import Vector2

# Globale Variablen für den Video-Mode
video_playing = False
video_start_time = 0.0
minimum_video_duration = 6  # Mindestdauer in Sekunden
video_list = []
video_position = None
current_video_file = None
last_switch_time = None
autoplay = True  # An die GUI anheften!
wait_for_new_video = False
switch_interval = 5 * 60  # Intervall in Minuten, wann ein neues Video ausgewählt werden soll
last_flip = 0
video_stopped = 0


def get_video_list(directory):
    # Liste alle Dateien im Verzeichnis auf
    files = [f for f in os.listdir(directory) if f.endswith('.mp4')]
    # Mische die Liste zufällig
    random.shuffle(files)
    return files


# Screen resolution settings
screen_width = 1280
screen_height = 720
screen = pygame.display.set_mode((screen_width, screen_height))

# colors = [(255, 0, 0), (255, 0, 255), (0, 0, 255)]
font = 0
zero_color = (64, 64, 64)


def init_hdmi():
    """Initialize pygame and the display settings."""
    global font
    pygame.init()
    pygame.mouse.set_visible(False)

    # Schriftart und Größe festlegen
    font = pygame.font.SysFont(None, 72)


def quit_hdmi():
    pygame.quit()


def adjust_color(color):
    """Adjust color values using random values."""
    adjusted = []
    for value in color:
        if value == 0:
            adjusted.append(min(255, value + random.randint(0, 120)))
        elif value == 255:
            adjusted.append(max(0, value - random.randint(1, 120)))
        else:
            adjusted.append(value)
    return tuple(adjusted)


def hdmi_draw_black():
    """Fill the screen with black color."""
    global screen, last_flip
    hdmi_video_stop(True)
    time.sleep(0.005)
    screen.fill((0, 0, 0))
    pygame.display.flip()
    last_flip = time.time()


def hdmi_draw_centered_text(text):
    """Draw text centered on the screen."""
    global screen, font, last_flip

    lines = text.split('\n')
    total_height = sum([font.size(line)[1] for line in lines])
    start_y = (screen.get_height() - total_height) // 2
    screen.fill((0, 0, 0))
    for line in lines:
        text_surface = font.render(line, True, (255, 255, 255))
        center_x = (screen.get_width() - text_surface.get_width()) // 2
        screen.blit(text_surface, (center_x, start_y))
        start_y += font.size(line)[1]
    pygame.display.flip()
    last_flip = time.time()


text_cache = {}  # Dictionary to store pre-rendered text surfaces


@hdmi_in_thread
def hdmi_draw_matrix(matrix, top=(255, 0, 0), low=(0, 0, 255), mid=(255, 0, 255), glitch_mode = "off"):
    """Draw a matrix of numbers with glitch effects and overlays depending on glitch_mode."""
    """glitch_mode can be "off", "medium", or "maximum_chaos"""
    global screen, font, text_cache, last_flip
    if len(matrix) != 15 or any(len(row) != 9 for row in matrix):
        raise ValueError("Matrix must be 15x9.")

    colors = [low, mid, top]
    border_x = screen_width * 0.09
    border_y = screen_height * 0.11
    spacing_x = (screen_width - 2 * border_x) / 14.3
    spacing_y = (screen_height - 2 * border_y) / 7.8

    screen.fill((0, 0, 0))

    # Draw the matrix
    for x, column in enumerate(matrix):
        for y, number in enumerate(column):
            base_color = colors[y // 3]
            color = adjust_color(base_color)

            if number == 0:
                color = (75, 75, 75)

            text_key = (number, color)
            if text_key not in text_cache:
                text_cache[text_key] = font.render(str(number), True, color)

            text = text_cache[text_key]

            pos_x = border_x + x * spacing_x + random.randint(-3, 3)
            pos_y = screen_height - border_y - y * spacing_y + random.randint(-3, 3)
            screen.blit(text, (pos_x, pos_y))

    # --- Image distortion (displacement of stripes) ---
    if glitch_mode in ["medium", "maximum_chaos"] and random.random() < 1:
        num_stripes = random.randint(1, 3 if glitch_mode == "medium" else 8)
        for _ in range(num_stripes):
            y = random.randint(0, screen_height - 1)
            max_height = screen_height - y
            height = min(random.randint(5, 20 if glitch_mode == "medium" else 60), max_height)
            shift_x = random.randint(-10, 10 if glitch_mode == "medium" else 40)

            if shift_x == 0 or height <= 0:
                continue

            try:
                glitch_slice = screen.subsurface((0, y, screen_width, height)).copy()
                screen.blit(glitch_slice, (shift_x, y))
            except Exception as e:
                print(f"Glitch draw failed at y={y}, h={height}: {e}")

    # --- Overlay interference (dashes, blocks) ---
    if glitch_mode in ["medium", "maximum_chaos"] and random.random() < (0.4 if glitch_mode == "medium" else 1):
        glitch_overlay = pygame.Surface((screen_width, screen_height), pygame.SRCALPHA)
        num_blocks = random.randint(1, 3 if glitch_mode == "medium" else 6)
        for _ in range(num_blocks):
            glitch_type = random.choice(["stripe", "block", "shift"])
            if glitch_type == "stripe":
                y = random.randint(0, screen_height)
                h = random.randint(2, 10)
                alpha = random.randint(40, 90)
                pygame.draw.rect(glitch_overlay, (200, 200, 255, alpha), (0, y, screen_width, h))
            elif glitch_type == "block":
                w = random.randint(30, 100)
                h = random.randint(10, 50)
                x = random.randint(0, screen_width - w)
                y = random.randint(0, screen_height - h)
                alpha = random.randint(30, 80)
                pygame.draw.rect(glitch_overlay, (255, 255, 255, alpha), (x, y, w, h))
            elif glitch_type == "shift":
                w = random.randint(40, 120)
                h = random.randint(10, 20)
                x = random.randint(0, screen_width - w)
                y = random.randint(0, screen_height - h)
                alpha = random.randint(20, 60)
                pygame.draw.rect(glitch_overlay, (10, 10, 10, alpha), (x, y, w, h))

        screen.blit(glitch_overlay, (0, 0))

    pygame.display.flip()


def hdmi_intro_animation():
    """Play a screen animation simulating a matrix bootup."""
    global screen, font, zero_color, last_flip

    n = 4  # frequency
    current_cycle = 0
    last_randoms = [[(0, 0) for _ in range(9)] for _ in range(15)]
    border_x = screen_width * 0.09
    border_y = screen_height * 0.11
    groups = [6, 7, 8], [3, 4, 5], [0, 1, 2]

    # Function to draw a matrix column with shake effects
    def draw_column(col, rows):
        for row in rows:
            text = font.render('0', True, zero_color)
            spacing_x = (screen_width - 2 * border_x) / 14.3
            spacing_y = (screen_height - 2 * border_y) / 7.8

            if current_cycle % n == 0:
                random_x = random.randint(-5, 5)
                random_y = random.randint(-5, 5)
                pos_x = border_x + col * spacing_x + random_x
                pos_y = screen_height - border_y - row * spacing_y + random_y
                last_randoms[col][row] = (random_x, random_y)
            else:
                pos_x = border_x + col * spacing_x
                pos_y = screen_height - border_y - row * spacing_y
                pos_x += last_randoms[col][row][0]
                pos_y += last_randoms[col][row][1]
            screen.blit(text, (pos_x, pos_y))

    for idx, group in enumerate(groups):
        for column in range(15):
            screen.fill((0, 0, 0))

            for prev_group in groups[:idx]:
                for prev_column in range(15):
                    draw_column(prev_column, prev_group)

            for prev_column in range(column):
                draw_column(prev_column, group)

            draw_column(column, group)

            pygame.display.flip()
            last_flip = time.time()
            pygame.time.wait(int(60 / n))

            current_cycle += 1


def hdmi_outro_animation():
    """Play a screen animation simulating a matrix shutdown."""
    global screen, font, zero_color, last_flip

    n = 4  # frequency
    current_cycle = 0

    last_positions = [[(0, 0) for _ in range(9)] for _ in range(15)]

    border_x = screen_width * 0.09
    border_y = screen_height * 0.11

    groups = [0, 1, 2], [3, 4, 5], [6, 7, 8]

    # Function to draw a matrix column with shake effects
    def draw_column(col, rows):
        for row in rows:
            if current_cycle % n == 0:
                spacing_x = (screen_width - 2 * border_x) / 14.3
                spacing_y = (screen_height - 2 * border_y) / 7.8
                pos_x = border_x + col * spacing_x + random.randint(-5, 5)
                pos_y = screen_height - border_y - row * spacing_y + random.randint(-5, 5)
                last_positions[col][row] = (pos_x, pos_y)
            else:
                pos_x, pos_y = last_positions[col][row]

            text = font.render('0', True, zero_color)
            screen.blit(text, (pos_x, pos_y))

    for idx, group in enumerate(groups):
        for column in reversed(range(15)):
            screen.fill((0, 0, 0))

            for visible_group in groups[idx + 1:]:
                for visible_column in range(15):
                    draw_column(visible_column, visible_group)

            for visible_column in range(column):
                draw_column(visible_column, group)

            pygame.display.flip()
            last_flip = time.time()
            pygame.time.wait(int(100 / n))

            current_cycle += 1


def hdmi_video_stop(now=False):
    global video_playing, video_start_time, wait_for_new_video, video_stopped
    wait_for_new_video = False
    current_time = time.time()

    # Überprüfe, ob das Video bereits für die Mindestdauer gespielt hat
    if video_start_time and (current_time - video_start_time) < minimum_video_duration and not now:
        # Wenn die Mindestdauer noch nicht erreicht ist, wird der Stop-Vorgang verzögert
        # print("Video wird fortgesetzt, Mindestdauer noch nicht erreicht.")
        return False  # Gib False zurück, um anzuzeigen, dass das Video nicht gestoppt wurde
    else:
        # Die Mindestdauer ist erreicht, das Video kann gestoppt werden
        video_playing = False
        video_stopped = time.time()
        if time.time() - last_flip < 0.05:
            # print("wait a bit for hmdi")
            time.sleep(0.05)
            video_playing = False
        if time.time() - last_flip < 0.05:
            # print("kill hmdi")
            kill_current_hdmi()
            time.sleep(0.05)

        return True  # Gib True zurück, um anzuzeigen, dass das Video gestoppt wurde


def hdmi_video_start():
    global video_playing
    video_playing = True


@hdmi_in_thread
def hdmi_play_video(video_directory):
    global screen, video_playing, video_list, video_position, video_start_time, current_video_file, last_switch_time, \
        autoplay, wait_for_new_video, last_flip, video_stopped

    if wait_for_new_video or time.time() - video_stopped < 1.0:
        return

    video_playing = True
    video_start_time = time.time()  # Speichere die Startzeit

    if not video_list:
        video_list = get_video_list(video_directory)

    if not current_video_file:
        last_switch_time = time.time()
        current_video_file = video_list[0]

    video_path = os.path.join(video_directory, video_list[0])
    cap = cv2.VideoCapture(video_path)

    if video_position is not None:
        # Setze die Position des Videos auf die gespeicherte Position
        cap.set(cv2.CAP_PROP_POS_FRAMES, video_position)

    ret, frame = cap.read()
    if not ret:
        print("Fehler beim Lesen des Videos.")
        cap.release()
        return

    clock = pygame.time.Clock()
    frame_rate = 25  # Ziel-Frame-Rate

    while cap.isOpened() and video_playing:
        ret, frame = cap.read()

        if not ret:  # Wenn das Video zu Ende ist
            video_position = 0
            video_list.append(video_list.pop(0))  # Verschiebe das aktuelle Video ans Ende der Liste
            current_video_file = video_list[0]  # Aktualisiere den aktuellen Dateinamen
            last_switch_time = time.time()  # Setze den Timer für den nächsten Wechsel zurück
            cap.release()  # Schließe die aktuelle Video-Capture-Instanz

            if autoplay:
                # Starte das nächste Video direkt
                return hdmi_play_video(video_directory)
            else:
                hdmi_draw_black()
                wait_for_new_video = True
                video_playing = False
                return

        # Aktualisiere die aktuelle Position des Videos
        video_position = cap.get(cv2.CAP_PROP_POS_FRAMES)

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_surface = pygame.surfarray.make_surface(frame.swapaxes(0, 1))

        screen.blit(frame_surface, (0, 0))
        pygame.display.flip()
        last_flip = time.time()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                cap.release()
                pygame.quit()
                return

        clock.tick(frame_rate)

    cap.release()
    video_playing = False
    current_hdmi_thread = None

    current_time = time.time()
    # Überprüfen, ob das Video gewechselt werden sollte
    if last_switch_time and (current_time - last_switch_time) >= switch_interval:
        # Verschiebe das aktuelle Video ans Ende der Liste und setze die Zeit zurück
        # print("Neues Video ausgewählt, das alte lief schon lang genug.")
        video_list.append(video_list.pop(0))
        last_switch_time = time.time()  # Setze den Timer für den nächsten Wechsel zurück
        current_video_file = video_list[0]  # Aktualisiere den aktuellen Dateinamen
        video_position = None


def is_video_playing():
    global video_playing
    return video_playing

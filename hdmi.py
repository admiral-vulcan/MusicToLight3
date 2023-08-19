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

# Required Libraries
import pygame
import random
from helpers import *

# Screen resolution settings
screen_width = 1920
screen_height = 1080
screen = pygame.display.set_mode((screen_width, screen_height))

colors = [(255, 0, 0), (255, 0, 255), (0, 0, 255)]
font = 0
zero_color = (16, 16, 16)


def init_hdmi():
    """Initialize pygame and the display settings."""
    global font
    pygame.init()

    # Schriftart und Größe festlegen
    font = pygame.font.SysFont(None, 72)


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
    screen.fill((0, 0, 0))
    pygame.display.flip()


def hdmi_draw_centered_text(text):
    """Draw text centered on the screen."""
    global screen, font

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


@hdmi_in_thread
def hdmi_draw_matrix(matrix):
    """Draw a matrix of numbers with various colors and effects."""
    global screen, font, colors, zero_color
    if len(matrix) != 15 or any(len(row) != 9 for row in matrix):
        raise ValueError("Matrix must be 15x9.")

    border_x = screen_width * 0.09
    border_y = screen_height * 0.11
    spacing_x = (screen_width - 2 * border_x) / 14.3
    spacing_y = (screen_height - 2 * border_y) / 7.8

    screen.fill((0, 0, 0))

    for x, column in enumerate(matrix):
        for y, number in enumerate(column):
            base_color = colors[y // 3]
            color = adjust_color(base_color)
            if number == 0:
                color = zero_color
            text = font.render(str(number), True, color)
            pos_x = border_x + x * spacing_x + random.randint(-5, 5)
            pos_y = screen_height - border_y - y * spacing_y + random.randint(-5, 5)
            screen.blit(text, (pos_x, pos_y))
    pygame.display.flip()


@hdmi_in_thread
def hdmi_intro_animation():
    """Play a screen animation simulating a matrix bootup."""
    global screen, font, colors, zero_color

    n = 4 # frequency
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
            pygame.time.wait(int(60 / n))

            current_cycle += 1


@hdmi_in_thread
def hdmi_outro_animation():
    """Play a screen animation simulating a matrix shutdown."""
    global screen, font, colors, zero_color

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
            pygame.time.wait(int(100 / n))

            current_cycle += 1

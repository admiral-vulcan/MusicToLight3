import pygame
import random
from helpers import *

# Bildschirmauflösung setzen (angepasst an deine Bildschirmauflösung)
screen_width = 1920
screen_height = 1080
screen = pygame.display.set_mode((screen_width, screen_height))

colors = [(0, 0, 255), (255, 0, 0), (255, 255, 0)]
font = 0
zero_color = (64, 64, 64)


# Pygame Initialisierung
def init_hdmi():
    global font
    pygame.init()

    # Schriftart und Größe festlegen
    font = pygame.font.SysFont(None, 72)


def adjust_color(color):
    """Justiert den Farbwert mit zufälligen Werten."""
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
    screen.fill((0, 0, 0))


@hdmi_in_thread
def hdmi_draw_matrix(matrix):
    global screen, font, colors, zero_color
    # Sicherstellen, dass das Matrix-Format korrekt ist
    if len(matrix) != 15 or any(len(row) != 9 for row in matrix):
        print(f"Debug: Matrix hat Größe {len(matrix)}x{len(matrix[0])}")  # Debug-Print
        raise ValueError("Matrix muss 15x9 sein.")

    # Ränder, um den abgeschnittenen Bereich des TVs zu berücksichtigen
    border_x = screen_width * 0.09  # 5% des Bildschirms als Rand
    border_y = screen_height * 0.11

    # Abstand zwischen den Zahlen berechnen
    spacing_x = (screen_width - 2 * border_x) / 14.3
    spacing_y = (screen_height - 2 * border_y) / 7.8

    # Bildschirm mit Schwarz füllen
    screen.fill((0, 0, 0))

    for x, column in enumerate(matrix):
        for y, number in enumerate(column):
            # Farbe basierend auf der y-Position festlegen und dann anpassen
            base_color = colors[y // 3]
            color = adjust_color(base_color)

            # Wenn die Zahl 0 ist, setze die Farbe auf Weiß
            if number == 0:
                color = zero_color

            text = font.render(str(number), True, color)
            pos_x = border_x + x * spacing_x + random.randint(-5, 5)  # Abweichung in X-Richtung
            pos_y = screen_height - border_y - y * spacing_y + random.randint(-5, 5)  # Abweichung in Y-Richtung
            screen.blit(text, (pos_x, pos_y))

    pygame.display.flip()


@hdmi_in_thread
def hdmi_intro_animation():
    global screen, font, colors, zero_color

    n = 4  # Die Frequenz
    current_cycle = 0  # Zählt die Zyklen

    # Eine Liste, die die letzten wackelnden Positionen speichert
    last_randoms = [[(0, 0) for _ in range(9)] for _ in range(15)]

    # Ränder für die abgeschnittenen Bereiche des Bildschirms
    border_x = screen_width * 0.09
    border_y = screen_height * 0.11

    # Gruppen von Reihen, die nacheinander animiert werden (Reihenfolge umgedreht)
    groups = [6, 7, 8], [3, 4, 5], [0, 1, 2]

    def draw_column(col, rows):
        for row in rows:
            text = font.render('0', True, zero_color)  # Weiß für die Ziffer 0
            spacing_x = (screen_width - 2 * border_x) / 14.3
            spacing_y = (screen_height - 2 * border_y) / 7.8

            # Wenn der aktuelle Zyklus durch n teilbar ist, berechne eine neue Wackelposition
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
            # Bildschirm mit Schwarz füllen
            screen.fill((0, 0, 0))

            # Zeichne bereits gezeichnete Gruppen
            for prev_group in groups[:idx]:
                for prev_column in range(15):
                    draw_column(prev_column, prev_group)

            # Zeichne bereits gezeichnete Spalten für die aktuelle Gruppe
            for prev_column in range(column):
                draw_column(prev_column, group)

            # Zeichne die aktuelle Spalte für die aktuelle Gruppe
            draw_column(column, group)

            pygame.display.flip()
            pygame.time.wait(int(60 / n))

            # Aktualisiere den Zykluszähler
            current_cycle += 1


@hdmi_in_thread
def hdmi_outro_animation():
    global screen, font, colors, zero_color

    n = 4  # Die Frequenz
    current_cycle = 0  # Zählt die Zyklen

    # Eine Liste, die die letzten wackelnden Positionen speichert
    last_positions = [[(0, 0) for _ in range(9)] for _ in range(15)]

    # Ränder für die abgeschnittenen Bereiche des Bildschirms
    border_x = screen_width * 0.09
    border_y = screen_height * 0.11

    # Gruppen von Reihen, die nacheinander animiert werden (von unten nach oben)
    groups = [0, 1, 2], [3, 4, 5], [6, 7, 8]

    def draw_column(col, rows):
        for row in rows:
            # Wenn der aktuelle Zyklus durch n teilbar ist, berechne eine neue Wackelposition
            if current_cycle % n == 0:
                spacing_x = (screen_width - 2 * border_x) / 14.3
                spacing_y = (screen_height - 2 * border_y) / 7.8
                pos_x = border_x + col * spacing_x + random.randint(-5, 5)
                pos_y = screen_height - border_y - row * spacing_y + random.randint(-5, 5)
                last_positions[col][row] = (pos_x, pos_y)
            else:
                pos_x, pos_y = last_positions[col][row]

            text = font.render('0', True, zero_color)  # Weiß für die Ziffer 0
            screen.blit(text, (pos_x, pos_y))

    for idx, group in enumerate(groups):
        for column in reversed(range(15)):
            # Bildschirm mit Schwarz füllen
            screen.fill((0, 0, 0))

            # Zeichne Gruppen, die noch nicht verschwunden sind
            for visible_group in groups[idx+1:]:
                for visible_column in range(15):
                    draw_column(visible_column, visible_group)

            # Zeichne Spalten für die aktuelle Gruppe, die noch nicht verschwunden sind
            for visible_column in range(column):
                draw_column(visible_column, group)

            pygame.display.flip()
            pygame.time.wait(int(100 / n))

            # Aktualisiere den Zykluszähler
            current_cycle += 1


"""
# Testmatrix zum Zeichnen
matrix = [
    [1, 2, 3, 4, 5, 6, 7, 8, 9] for _ in range(15)
]


while True:
    draw_matrix(matrix)
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            sys.exit()
"""

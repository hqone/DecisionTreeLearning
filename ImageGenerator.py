import numpy as np
import cv2
import math


class ColorRGB:
    GREEN = (0, 255, 0)
    BLACK = (0, 0, 0)
    RED = (0, 0, 255)
    BLUE = (255, 0, 0)


class ImageGenerator:

    def blank_canvas(self, a_width, b_width=0):
        if b_width == 0:
            b_width = a_width
        img = np.zeros((a_width, b_width, 3), np.uint8)
        img.fill(255)
        return img

    def draw_square(self, a_width, padding, color, thickness):
        return cv2.rectangle(self.blank_canvas(a_width), (padding, padding), (a_width - padding, a_width - padding),
                             color, thickness)

    def draw_rectangle(self, a_width, b_width, padding, color, thickness):
        return cv2.rectangle(self.blank_canvas(a_width, b_width), (padding, padding),
                             (b_width - padding, a_width - padding), color, thickness)

    def draw_circle(self, a_width, radius, color, thickness):
        return cv2.circle(self.blank_canvas(a_width), (round(a_width / 2), round(a_width / 2)), radius, color,
                          thickness)

    def draw_ellipse(self, a_width, axes, angle, color, thickness):
        return cv2.ellipse(self.blank_canvas(a_width), (round(a_width / 2), round(a_width / 2)), axes, angle, 0,
                           360,
                           color, thickness)

    def draw_right_triangle(self, a_width, padding, color, thickness):
        vertices = np.array([
            [padding, padding],
            [padding, a_width - padding],
            [a_width - padding, a_width - padding]
        ], np.int32)
        pts = vertices.reshape((-1, 1, 2))

        return cv2.polylines(self.blank_canvas(a_width), [pts], True, color, thickness)

    def draw_equilateral_triangle(self, a_width, padding, color, thickness):
        y_bottom = a_width * math.sqrt(3) / 2
        vertices = np.array([
            [round(a_width / 2), padding],
            [padding, y_bottom - padding],
            [a_width - padding, y_bottom - padding]
        ], np.int32)
        pts = vertices.reshape((-1, 1, 2))

        return cv2.polylines(self.blank_canvas(a_width), [pts], True, color, thickness)


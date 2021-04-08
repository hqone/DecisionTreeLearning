import numpy as np
import cv2
import math


class ColorRGB:
    '''
    Stworzona w celu operowania na kolorach.
    '''
    GREEN = (0, 255, 0)
    BLACK = (0, 0, 0)
    RED = (0, 0, 255)
    BLUE = (255, 0, 0)


class ImageGenerator:
    '''
    Klasa prezentuje prostyintefrejs który na podstawie minimalnej liczby parametrów zwraca gotowe obrazy z odpowiednimi figurami.
    '''

    def blank_canvas(self, a_width, b_width=0):
        '''
        Puste płótno.
        :param a_width:
        :param b_width:
        :return:
        '''
        if b_width == 0:
            b_width = a_width
        img = np.zeros((a_width, b_width, 3), np.uint8)
        img.fill(255)
        return img

    def draw_square(self, a_width, padding, color, thickness):
        '''
        Rysyuje kwadrat o podanych parametrach.
        :param a_width:
        :param padding:
        :param color:
        :param thickness:
        :return:
        '''
        return cv2.rectangle(self.blank_canvas(a_width), (padding, padding), (a_width - padding, a_width - padding),
                             color, thickness)

    def draw_rectangle(self, a_width, b_width, padding, color, thickness):
        '''
        Rysyuje prostokąt o podanych parametrach.
        :param a_width:
        :param b_width:
        :param padding:
        :param color:
        :param thickness:
        :return:
        '''
        return cv2.rectangle(self.blank_canvas(a_width, b_width), (padding, padding),
                             (b_width - padding, a_width - padding), color, thickness)

    def draw_circle(self, a_width, radius, color, thickness):
        '''
        Rysyuje koło o podanych parametrach.
        :param a_width:
        :param radius:
        :param color:
        :param thickness:
        :return:
        '''
        return cv2.circle(self.blank_canvas(a_width), (round(a_width / 2), round(a_width / 2)), radius, color,
                          thickness)

    def draw_ellipse(self, a_width, axes, angle, color, thickness):
        '''
        Rysyuje elipsę o podanych parametrach.
        :param a_width:
        :param axes:
        :param angle:
        :param color:
        :param thickness:
        :return:
        '''
        return cv2.ellipse(self.blank_canvas(a_width), (round(a_width / 2), round(a_width / 2)), axes, angle, 0,
                           360,
                           color, thickness)

    def draw_right_triangle(self, a_width, padding, color, thickness):
        '''
        Rysyuje trójkąt prostokątny o podanych parametrach.
        :param a_width:
        :param padding:
        :param color:
        :param thickness:
        :return:
        '''
        vertices = np.array([
            [padding, padding],
            [padding, a_width - padding],
            [a_width - padding, a_width - padding]
        ], np.int32)
        pts = vertices.reshape((-1, 1, 2))

        return cv2.polylines(self.blank_canvas(a_width), [pts], True, color, thickness)

import glob
import logging

import cv2
import numpy as np
from ImageGenerator import ColorRGB, ImageGenerator
import copy

logger = logging.getLogger()

class DataSet:

    def generate_dataset_from_dir(self, search_path):

        file_paths = glob.glob("{}\\*.jpg".format(search_path))

        if not file_paths:
            logger.info("W podanej ścieżce nie znaleziono obrazów jpg.")

        dataset = []
        for file_path in file_paths:
            dataset.append(
                self.get_meta_data_from_image(
                    cv2.imread(file_path),
                    file_path.rsplit('\\', maxsplit=1)[1]
                ))
        return dataset

    def generate_training_dataset(self):
        ig = ImageGenerator()

        return [
            self.get_meta_data_from_image(ig.draw_square(456, 20, ColorRGB.GREEN, 3), 'Kwadrat'),
            self.get_meta_data_from_image(ig.draw_square(276, 20, ColorRGB.GREEN, 3), 'Kwadrat'),
            self.get_meta_data_from_image(ig.draw_square(226, 20, ColorRGB.GREEN, 3), 'Kwadrat'),
            self.get_meta_data_from_image(ig.draw_square(356, 20, ColorRGB.GREEN, 3), 'Kwadrat'),
            self.get_meta_data_from_image(ig.draw_rectangle(256, 210, 20, ColorRGB.BLACK, 4), 'Prostokat'),
            self.get_meta_data_from_image(ig.draw_rectangle(205, 220, 20, ColorRGB.BLACK, 4), 'Prostokat'),
            self.get_meta_data_from_image(ig.draw_rectangle(350, 300, 20, ColorRGB.BLACK, 4), 'Prostokat'),
            self.get_meta_data_from_image(ig.draw_circle(256, 60, ColorRGB.RED, 2), 'Kolo'),
            self.get_meta_data_from_image(ig.draw_circle(256, 110, ColorRGB.BLACK, 1), 'Kolo'),
            self.get_meta_data_from_image(ig.draw_circle(256, 110, ColorRGB.BLACK, 1), 'Kolo'),
            self.get_meta_data_from_image(ig.draw_ellipse(256, (60, 100), 20, ColorRGB.BLUE, 3), 'Elipsa'),
            self.get_meta_data_from_image(ig.draw_ellipse(256, (50, 100), 0, ColorRGB.BLUE, 3), 'Elipsa'),
            self.get_meta_data_from_image(ig.draw_ellipse(256, (120, 30), 0, ColorRGB.BLUE, 3), 'Elipsa'),
            self.get_meta_data_from_image(ig.draw_ellipse(256, (120, 30), 0, ColorRGB.BLUE, 3), 'Elipsa'),
            self.get_meta_data_from_image(ig.draw_ellipse(256, (100, 30), 0, ColorRGB.BLUE, 3), 'Elipsa'),
            self.get_meta_data_from_image(ig.draw_ellipse(256, (110, 30), 0, ColorRGB.BLUE, 3), 'Elipsa'),
            self.get_meta_data_from_image(ig.draw_right_triangle(256, 20, ColorRGB.GREEN, 1), 'Trojkat'),
            self.get_meta_data_from_image(ig.draw_right_triangle(512, 20, ColorRGB.BLACK, 1), 'Trojkat'),
            self.get_meta_data_from_image(ig.draw_right_triangle(200, 20, ColorRGB.BLACK, 1), 'Trojkat'),
        ]

    def generate_test_dataset(self):
        ig = ImageGenerator()

        return [
            self.get_meta_data_from_image(ig.draw_circle(512, 100, ColorRGB.BLACK, 1), 'Kolo'),
            self.get_meta_data_from_image(ig.draw_rectangle(300, 200, 20, ColorRGB.BLACK, 4), 'Prostokat'),
            self.get_meta_data_from_image(ig.draw_square(200, 20, ColorRGB.GREEN, 3), 'Kwadrat'),
            self.get_meta_data_from_image(ig.draw_ellipse(256, (130, 30), 0, ColorRGB.BLUE, 3), 'Elipsa'),
            self.get_meta_data_from_image(ig.draw_square(512, 20, ColorRGB.GREEN, 3), 'Kwadrat'),
            self.get_meta_data_from_image(ig.draw_right_triangle(550, 20, ColorRGB.BLACK, 1), 'Trojkat'),
            self.get_meta_data_from_image(ig.draw_square(256, 20, ColorRGB.GREEN, 3), 'Kwadrat'),
            self.get_meta_data_from_image(ig.draw_circle(256, 120, ColorRGB.BLACK, 1), 'Kolo'),
            self.get_meta_data_from_image(ig.draw_right_triangle(200, 20, ColorRGB.BLACK, 1), 'Trojkat'),
            self.get_meta_data_from_image(ig.draw_rectangle(220, 210, 20, ColorRGB.BLACK, 4), 'Prostokat'),
            self.get_meta_data_from_image(ig.draw_rectangle(250, 300, 20, ColorRGB.BLACK, 4), 'Prostokat'),
            self.get_meta_data_from_image(ig.draw_ellipse(256, (60, 120), 0, ColorRGB.BLUE, 3), 'Elipsa'),
            self.get_meta_data_from_image(ig.draw_circle(256, 70, ColorRGB.RED, 2), 'Kolo'),
            self.get_meta_data_from_image(ig.draw_ellipse(256, (30, 110), 20, ColorRGB.BLUE, 3), 'Elipsa'),
            self.get_meta_data_from_image(ig.draw_right_triangle(300, 20, ColorRGB.GREEN, 1), 'Trojkat'),
        ]

    def get_meta_data_from_image(self, img: np.ndarray, item_class: str = None):

        # cv2.imwrite('{}\\obrazy\\{}_{}.jpg'.format(os.getcwd(), item_class, randint(0, 1000)), img)

        lines_img_array, parallel_lines = self.find_lines(img)
        corners_pos = self.find_corners(img, lines_img_array)

        meta_data = {
            'count_lines': len(lines_img_array),
            'count_corners': len(corners_pos),
            'count_right_angles': self.find_right_angles(img, corners_pos),
            'parallel_lines': parallel_lines,
            'ratio_1_1': self.ratio_figure(img),
            'item_class': item_class
        }
        if item_class is not None:
            meta_data['item_class'] = item_class

        return meta_data

    def ratio_figure(self, img_in):
        img = copy.deepcopy(img_in)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = cv2.bitwise_not(img)

        ret, img = cv2.threshold(img, 21, 255, cv2.THRESH_BINARY)  # the same 21 as in `mask = (img > 21)`
        x, y, width, height = cv2.boundingRect(img)

        return 'yes' if width / height == 1.0 else 'no'

    def draw_clean_img(self, img):
        clean_img = copy.deepcopy(img)
        clean_img.fill(0)
        return clean_img

    def draw_contours(self, img):
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        contours_img = self.draw_clean_img(gray_img)

        _, binary = cv2.threshold(gray_img, 225, 255, cv2.THRESH_BINARY_INV)
        contours, hierarchy = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(contours_img, contours, 0, 255, 1)
        return contours_img

    def find_lines(self, img: np.ndarray):

        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        contours_img = self.draw_contours(img)

        lines = cv2.HoughLines(contours_img, 1, np.pi / 180, 100, np.array([]), 0, 0)
        # print(lines)
        lines_img_array = []
        line_eq_factor = []
        parallel_lines = 0

        # cv2.imshow('dst', contours_img)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

        if lines is not None:
            for i, line in enumerate(lines):
                rho, theta = line[0]
                a = np.cos(theta)
                b = np.sin(theta)
                x0 = a * rho
                y0 = b * rho
                x1 = int(x0 + 1000 * (-b))
                y1 = int(y0 + 1000 * a)
                x2 = int(x0 - 1000 * (-b))
                y2 = int(y0 - 1000 * a)

                points = [(x1, y1), (x2, y2)]
                x_coords, y_coords = zip(*points)
                if y_coords == (1000, -1000):
                    k = 'vertical'
                else:
                    A = np.vstack([x_coords, np.ones(len(x_coords))]).T
                    k, c = np.linalg.lstsq(A, y_coords, rcond=None)[0]
                    k = round(k)

                if k in line_eq_factor:
                    parallel_lines += 1

                line_eq_factor.append(k)
                lines_img_array.append(cv2.line(self.draw_clean_img(gray_img), (x1, y1), (x2, y2), 255, 1))

        return lines_img_array, parallel_lines

    def find_corners(self, img: np.ndarray, lines_img_array):

        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        circle_img = self.draw_clean_img(gray_img)
        corner_points_img = self.draw_clean_img(gray_img)

        for i, line1 in enumerate(lines_img_array):
            for j, line2 in enumerate(lines_img_array):
                if i <= j:
                    continue

                # cv2.imshow('dst', (line1 | line2))
                # cv2.waitKey(0)
                # cv2.destroyAllWindows()

                corner_points_img = corner_points_img | (line1 & line2)

        # cv2.imshow('dst', corner_points_img)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

        corners_pos = []
        for row, img_row in enumerate(corner_points_img):
            for col, color in enumerate(img_row):
                if color == 255:
                    corners_pos.append(np.array([col, row]))
                    cv2.circle(circle_img, (col, row), 19, 255, 1)

        return corners_pos

    def find_right_angles(self, img, corners_pos):
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        circle_img = self.draw_clean_img(gray_img)
        corner_points_img = self.draw_clean_img(gray_img)
        contours_img = self.draw_contours(img)
        right_angles = 0
        # print(corners_pos)
        for i in corners_pos:
            x, y = i.ravel()

            cv2.circle(corner_points_img, (x, y), 0, 255, -1)
            cv2.circle(circle_img, (x, y), 16, 255, 1)

            intersection_img = (circle_img & contours_img)
            #
            # cv2.imshow('dst', (circle_img & contours_img))
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()

            found_pixels = self.find_pixels(corner_points_img, x, y)
            found_pixels += self.find_pixels(intersection_img, x, y)

            ba = found_pixels[1] - found_pixels[0]
            bc = found_pixels[2] - found_pixels[0]

            cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
            angle = np.arccos(cosine_angle)

            if 85 <= np.degrees(angle) <= 95:
                right_angles += 1

        return right_angles

    def find_pixels(self, img, x, y, radius=20):

        height, width = img.shape[:2]

        crop_img = img[
                   y - radius if y - radius > 0 else 0:y + radius if height > y + radius else height,
                   x - radius if x - radius > 0 else 0:x + radius if width > x + radius else width
                   ]

        # found_pixels = [np.array([50, 50])]
        found_pixels = []
        # corner_points_img
        for row, img_row in enumerate(crop_img):
            for col, color in enumerate(img_row):
                if color == 255:
                    found_pixels.append(np.array([row, col]))

        return found_pixels

import numpy as np
import cv2
from pathlib import Path

path = ""

class image_tools(object):
    """
    'image_tools' contains useful tools for image manipulation based on the cv2 library, we wrap it so that we have easy access to it in our code
    """

    def show(self, image, window_name = 'Image', screen_res = (500.0, 500.0)):
        scale_width = screen_res[0] / image.shape[1]
        scale_height = screen_res[1] / image.shape[0]
        scale = min(scale_width, scale_height)
        window_width = int(image.shape[1] * scale)
        window_height = int(image.shape[0] * scale)

        if np.amax(image) > 1.:
            image = image / np.amax(image)

        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, window_width, window_height)

        cv2.imshow(window_name, image)
        cv2.waitKey()
        cv2.destroyAllWindows()
    
    def save_image(self, image, namefile):
        image = np.asarray(image)
        image = (image / np.amax(image) * 255).astype(np.uint8) 
        cv2.imwrite(str(path / namefile), image)

    def threshold(self, image):
        image = cv2.adaptiveThreshold(image.astype(np.uint8), 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 25, 5)
        return 255 - image

    def dilatate(self, image, kernel):
        cv2.dilate(image, kernel)
        return image

    def largest_contour(self, image):
        _, contours, h = cv2.findContours(image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        return max(contours, key=cv2.contourArea)

    def largest_4_side_contour(self, image):
        _, contours, h = cv2.findContours(image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        contours = sorted(contours, key=cv2.contourArea, reverse=True)
        for cnt in contours[:min(5,len(contours))]:
            if len(self.approx(cnt)) == 4:
                return cnt
        return None

    def make_it_square(self, image, side_length=306):
        return cv2.resize(image, (side_length, side_length))
    
    def area(self, image):
        return float(image.shape[0] * image.shape[1])

    def cut_out_sudoku_puzzle(self, image, contour):
        x, y, w, h = cv2.boundingRect(contour)
        image = image[y:y + h, x:x + w]
        return self.make_it_square(image, min(image.shape))

    def approx(self, cnt):
        perimeter = cv2.arcLength(cnt, True)
        app = cv2.approxPolyDP(cnt, 0.01 * perimeter, True)
        return app

    def get_rectangle_corners(self, cnt):
        pts = cnt.reshape(4, 2)
        rect = np.zeros((4, 2), dtype="float32")

        # Fun Fact: the top-left corner has the smallest coordinate sum and the bottom-rigth one the largest.
        Sum = pts.sum(axis=1)
        rect[0] = pts[np.argmin(Sum)]
        rect[2] = pts[np.argmax(Sum)]

        # Fun Fact 2: the top-right corner has the minimum coordinate difference and the bottom-left one the maximum difference.
        difference = np.diff(pts, axis=1)
        rect[1] = pts[np.argmin(difference)]
        rect[3] = pts[np.argmax(difference)]
        return rect

    def warp_perspective(self, rect, grid):
        (tl, tr, br, bl) = rect
        # Widths
        widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
        widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
        # Heights
        heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
        heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
        # Take Maximums
        maxWidth = max(int(widthA), int(widthB))
        maxHeight = max(int(heightA), int(heightB))

        # Construct corrispective destination points
        dst = np.array([
            [0, 0],
            [maxWidth - 1, 0],
            [maxWidth - 1, maxHeight - 1],
            [0, maxHeight - 1]], dtype="float32")
        
        # Construct perspective matrix
        M = cv2.getPerspectiveTransform(rect, dst)
        warp = cv2.warpPerspective(grid, M, (maxWidth, maxHeight))
        return self.make_it_square(warp, max(maxWidth, maxHeight))

    def get_top_line(self, image):
        for i, row in enumerate(image):
            if np.any(row):
                return i
        return None

    def get_bottom_line(self, image):
        for i in range(image.shape[0] - 1, -1, -1):
            if np.any(image[i]):
                return i
        return None

    def get_left_line(self, image):
        for i in range(image.shape[1]):
            if np.any(image[:, i]):
                return i
        return None

    def get_right_line(self, image):
        for i in range(image.shape[1] - 1, -1, -1):
            if np.any(image[:, i]):
                return i
        return None

    def row_shift(self, image, start, end, length):
        """
        Given the starting row and the ending row,
        it shifts them by a given number of steps
        """
        if start - length < 0 or end + length >= image.shape[0]:
            return image
        shifted = np.zeros(image.shape)
        
        for row in range(start, end + 1):
            shifted[row + length] = image[row]
        return shifted

    def col_shift(self, image, start, end, length):
        """
        Given the starting column and the ending column,
        it shifts them by a given number of steps
        """
        if start - length < 0 or end + length >= image.shape[1]:
            return image
        shifted = np.zeros(image.shape)
        
        for col in range(start, end + 1):
            shifted[:, col + length] = image[:, col]
        return shifted


        
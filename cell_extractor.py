import numpy as np
import cv2

from image_tools import image_tools
from digit_extractor import digit_extractor


class cell_extractor(object):
    """
    'cells' extracts each cell from a given sudoku grid.
    The sudoku grid is obtained from the general extractor.
    """
    def __init__(self, sudoku):
        print("Estrazione celle in corso...")
        self.tool = image_tools()
        self.cells = self.extract_cells(sudoku)
        print("Finita Estrazione!")

    def extract_cells(self, sudoku):
        cells = []
        W, H = sudoku.shape
        cell_size = W // 9
        i, j = 0, 0

        for r in range(0, W - W % 9, cell_size):
            row = []
            j = 0
            for c in range(0, W - W % 9, cell_size):
                print("Estrazione cella [" + str(i)+", " +str(j)+"]")

                cell = sudoku[r : r + cell_size, c : c + cell_size]
                
                self.tool.save_image(cell, str(i) + str(j) + "_1.png")

                cell = self.clean(cell)

                self.tool.save_image(cell, str(i) + str(j)+"_2.png")

                digit = digit_extractor(cell).digit
                
                self.tool.save_image(digit * 255, str(i) + str(j)+"_3.png")

                digit = self.refine(digit.astype(np.uint8))

                digit = self.center_digit(digit)
                
                self.tool.save_image(digit, str(i) + str(j)+"_4.png")
                row.append(digit)
                j += 1
            cells.append(row)
            i += 1

        # try thresholding
        cells = self.final_threshold(cells)

        return cells

    def final_threshold(self, cells):
        # Count points
        points = []
        for row in cells:
            for cell in row:
                unique, counts = np.unique(cell, return_counts=True)
                if len(unique) == 1:
                    points.append(0)
                else:
                    points.append(dict(zip(unique, counts))[1])
        points = np.asarray(points)
        
        # Try this basic threshold
        mean = np.sum(points) / 81

        for i in range(7):
            mean1 = np.sum(points[(points > mean)])/np.count_nonzero(points[(points > mean)])
            mean2 = np.sum(points[(points < mean)])/np.count_nonzero(points[(points < mean)])
            mean = (mean1 + mean2) / 2

        # Now obscure the unworthy ones!
        for i in range(9):
            for j in range(9):
                if points[i*9 + j] < mean:
                    cells[i][j] = np.zeros(cells[i][j].shape)
        return cells

    def clean(self, cell):
        cell = cv2.GaussianBlur(cell,(7,7),1)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        cell = cv2.morphologyEx(cell, cv2.MORPH_CLOSE, kernel)
        return cell

    def refine(self, image):
    	kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2,2))
    	image = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)
    	return image

    def center_digit(self, digit):
        digit = self.center_x(digit)
        digit = self.center_y(digit)
        return digit
    
    def center_x(self, digit):
        top_line = self.tool.get_top_line(digit)
        bottom_line = self.tool.get_bottom_line(digit)
        if top_line is None or bottom_line is None:
            return digit
        length_2 = abs(top_line - bottom_line) // 2
        corr_pos = digit.shape[0] // 2 - length_2
        length = corr_pos - top_line

        digit = self.tool.row_shift(digit, start = top_line, end = bottom_line, length = length)
        return digit

    def center_y(self, digit):
        left_line = self.tool.get_left_line(digit)
        right_line = self.tool.get_right_line(digit)
        if left_line is None or right_line is None:
            return digit
        length_2 = abs(left_line - right_line) // 2
        corr_pos = digit.shape[1] // 2 - length_2
        length = corr_pos - left_line

        digit = self.tool.col_shift(digit, start = left_line, end = right_line, length = length)
        return digit

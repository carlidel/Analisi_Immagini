import cv2
import numpy as np

from image_tools import image_tools
from cell_extractor import cell_extractor

class sudoku_extractor(object):
    """
    'sudoku_extractor' stores and manipulates the input image to extract the Sudoku puzzle all the way to the cells
    """
    def __init__(self, path):
        self.tool = image_tools()
        self.image = self.load_image(path)
        #self.tool.show(self.image, "begin!")
        
        self.preprocess()
        #self.tool.show(self.image, "Dopo Preprocessing")
        self.tool.save_image(self.image, "dopo_preprocessing.png")

        sudoku = self.crop_sudoku()
        #self.tool.show(sudoku, "Dopo Cropping out")
        self.tool.save_image(sudoku, "dopo_primo_cropping.png")

        sudoku = self.straighten(sudoku)
        #self.tool.show(sudoku, "Sudoku finale")
        self.tool.save_image(sudoku, "dopo_ultimo_ritaglio.png")

        sudoku = self.tool.make_it_square(sudoku, side_length=900)

        self.cells = cell_extractor(sudoku).cells
        
        row = []
        for i in range(9):
            row.append(np.concatenate((self.cells[i][:]), axis = 1))    
        sudoku = np.concatenate((row[:]), axis = 0)
        
        self.tool.save_image(sudoku, "binary.png")

    def load_image(self, path):
        color_img = cv2.imread(str(path))
        if color_img is None:
            raise IOError("Immagine non caricata!")
        print("Immagine caricata!")
        return color_img

    def preprocess(self):
        print("Preprocessing...")
        # Color Conversion and Thresholding
        self.image = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        self.image = cv2.GaussianBlur(self.image,(3,3),1) 
        self.image = self.tool.threshold(self.image)
        # Morph Close in order to close the external Square of the sudoku
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
        self.image = cv2.morphologyEx(self.image, cv2.MORPH_CLOSE, kernel)        
        print("Fine preprocessing!")

    def crop_sudoku(self):
        print("Ritagliando ora il Sudoku...")
        # Use image copy since findContour changes it   
        contour = self.tool.largest_contour(self.image.copy())
        sudoku = self.tool.cut_out_sudoku_puzzle(self.image.copy(), contour)
        print("Fine ritaglio!")
        return sudoku

    def straighten(self, sudoku):
        print("Raddrizzando ora l'immagine...")
        # Use image copy since findContour changes it   
        largest = self.tool.largest_4_side_contour(sudoku.copy())
        app = self.tool.approx(largest)
        corners = self.tool.get_rectangle_corners(app)
        sudoku = self.tool.warp_perspective(corners, sudoku)
        print("Finito!")
        return sudoku

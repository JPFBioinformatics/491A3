import matplotlib.pyplot as plt
import numpy as np

def find_nb(x_row, x_col, matrix):
    """
    Finds the i,j row,col idx values of the 4-neighbors of pixel x in the image matrix
    Params:
        x_row,x_col         row,col for pixel you want to analyze
        matrix              matrix the pixel resides in
    """
    # get matrix shape
    rows,cols = matrix.shape

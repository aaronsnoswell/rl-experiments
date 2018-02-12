
import numpy as np

# Lambda to convert a byte color (0-255) to a float color (0-1)
byte2float = lambda c: tuple((ci / 255 for ci in c))

# These are the "Tableau 20" colors as RGB.  
tableau20 = [byte2float(c) for c  in
		[(31, 119, 180), (174, 199, 232), (255, 127, 14), (255, 187, 120),
		 (44, 160, 44), (152, 223, 138), (214, 39, 40), (255, 152, 150),
		 (148, 103, 189), (197, 176, 213), (140, 86, 75), (196, 156, 148),
		 (227, 119, 194), (247, 182, 210), (127, 127, 127), (199, 199, 199),
		 (188, 189, 34), (219, 219, 141), (23, 190, 207), (158, 218, 229)]
	]

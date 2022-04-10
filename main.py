# Import libraries
import numpy as np
import matplotlib.pyplot as plt

# Define useful constants
width = 100 # Number of cells wide
height = 100 # Number of cells tall
density = 0.3 # Probability a cell should be yellow, which on average is proportion / density of yellow cells in grid

colours = np.random.rand(width, height)  # Initialise 2d array of random numbers between 0 and 1
colours = colours < density  # Sets each cell to 1 if it's less than density, else 0.

plt.title("p = " + str(density))  # Adds title showing probability of a point being filled yellow
plt.pcolor(colours, cmap="plasma")  # Makes the graph a rectangular grid plot with yellow and blue colour scheme
plt.plot()  # Plots the grid

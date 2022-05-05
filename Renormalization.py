import numpy as np
import matplotlib.pyplot as plt

# function that takes a numpy array of 0s and 1s, and interprets them as a binary number and adds 1
def increment_binary_array(array):
    for i, item in enumerate(array):
        item += 1
        if item == 1:
            array[i] = item
            return array
        if item == 2:
            array[i] = 0.0


def renormalization_approximate(gridsize):
    # creates a dictionary of keys and values that correspond to values of x^n in the polynomial and their coefficients
    coeff_x_n = dict()
    # values from 0 to gridsize-1 wont be able to percolate so we only need to consider ones higher than that.
    for i in range(gridsize, gridsize**2+1):
        coeff_x_n[i] = 0

    # pattern is the binary number that is unique to each permutation of cells, starting off at all off.
    pattern = np.zeros(gridsize**2)
    # up to what would be all cells on
    for i in range(pow(2,gridsize**2)):
        # the number of on squares is just the sum of each of the digits in the binary number
        total_squares = sum(pattern)
        # if it's less than the width then it can't percolate and we can skip
        if total_squares >= gridsize:
            # otherwise, the grid is the 2d interpretation of the pattern
            grid = np.reshape(pattern, (gridsize, gridsize))
            # if it percolates
            if find_path_across(grid):
                # then add 1 to the corresponding coefficient in the polynomial
                coeff_x_n[total_squares] += 1
        # then go to the next binary number
        increment_binary_array(pattern)

    # currently printing so I can see the coefficients are correct
    print(coeff_x_n)

    # gets an x axis for plotting it
    step = 0.0001
    x = np.arange(0,1+step,step)

    # function to output the result from plugging a value into the desired polynomial
    def poly_x(x, poly, gridsize):
        total = 0
        # for each of the powers in the dictionary
        for power in poly:
            # we add the coefficient * x^(power) * (1-x)^(width - power)
            total += poly[power]*pow(x, power) * pow(1-x, gridsize**2-power)
        # then subtract x from both sides of the final equation
        return total - x

    # gets the y axis, draws the line y = 0 and plots the x,y graph
    y = poly_x(x,coeff_x_n, gridsize)
    plt.hlines(0,0,1,"k")
    plt.plot(x,y)


# function from elsewhere that determines if a grid configuration percolates
def find_path_across(grid):
    '''
    For a numpy array of booleans (the yellow/blue grid), this function returns True or False
    depending on whether or not a yellow path exists from the left to the right edge.
    It will also display the path if you pass True
    This is not efficient
    '''

    # we know all the yellows on the left are reachable
    grid_size = np.shape(grid)
    x_length = grid_size[1]
    y_length = grid_size[0]
    # we make an array of the same size as the grid to store
    # which squares we can reach. Initially we assume we can't reach any
    # 0 represents a square is unreachable
    reachable = np.full(grid_size, 0)
    # yellows is a list of arrays which contain the coordinates of each yellow square
    yellows = np.argwhere(grid == True)

    # we store the x and y coordinates of the points in yellows in seperate arrays in a larger array yellows_xy
    yellows_xy = np.array(yellows).T

    # we find the indices in the x coordinate array of each yellow in the left hand column of the grid
    indices = np.where(yellows_xy[1] == 0)

    # using the indices we found we now find the y coordinate of these yellow squares in the first column
    # we iterate over each index to assign 1 to reachable[y,0] where [y,0] is yellow.
    # 1 signifies a yellow reachable square in reachable while all currently unreached squares are 0
    for i in (indices[0]):
        reachable[yellows_xy[0][i], 0] = 1

    # we create an array of all coordinates of sites already checked in the form of a list of lists
    reachable_array = np.asarray(np.where(reachable == 1)).T.tolist()

    # now we search for adjacent yellow squares to the ones we already have
    # we iterate over our array of coordinates of reachable squares
    # we assign y and x for each squares so that we can later compare to adjacent sites
    for square in reachable_array:
        y = square[0]
        x = square[1]
        # As soon as a site on the RHS is shown to be reachable we return True
        if 1 in reachable[:, x_length - 1]:
            return True
        # check adjacent
        # the first ifs in each part stops us getting index out of bounds errors
        # we check !=1 because otherwise we'd find the same squares again
        # and it would loop forever
        else:
            if x < x_length - 1:
                # checks right
                if grid[y, x + 1] == True and reachable[y, x + 1] != 1:
                    reachable[y, x + 1] = 1
                    reachable_array.append([y, x + 1])
            if y - 1 >= 0:
                # checks below
                if grid[y - 1, x] == True and reachable[y - 1, x] != 1:
                    reachable[y - 1, x] = 1
                    reachable_array.append([y - 1, x])
            if y < y_length - 1:
                # checks above
                if grid[y + 1, x] == True and reachable[y + 1, x] != 1:
                    reachable[y + 1, x] = 1
                    reachable_array.append([y + 1, x])
            if x - 1 >= 0:
                # checks left
                if grid[y, x - 1] == True and reachable[y, x - 1] != 1:
                    reachable[y, x - 1] = 1
                    reachable_array.append([y, x - 1])

    # Return False if after no further reachable path is found and we have not reached the right hand side of the grid
    return False

renormalization_approximate(2)
renormalization_approximate(3)
renormalization_approximate(4)
plt.show()

# Import libraries
import numpy as np
import matplotlib.pyplot as plt
import time

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


def display_rand_grid(grid_size, p_yellow):
    """
    A function to display a random square grid of blue and yellow squares.
    First argument is the side length and the second is the probability that a square is yellow
    """

    colours = np.random.rand(grid_size, grid_size)  # Initialise 2d array of random numbers between 0 and 1
    colours = colours < p_yellow  # Sets each cell to 1 if it's less than p_yellow, else 0.

    plt.figure(figsize=(6, 6))  # set appropriate figure size
    plt.title("probability = " + str(p_yellow))  # Adds title showing probability of a point being filled yellow
    plt.pcolor(colours, cmap="plasma")  # Makes the graph a rectangular grid plot with yellow and blue colour scheme
    plt.gca().set_aspect('equal')  # set equal aspect ratio
    plt.plot()  # Plots the grid
    plt.show()
    return colours


def display_grid(grid):
    plt.figure(figsize=(6, 6))  # set appropriate figure size
    plt.pcolor(grid, cmap="plasma")  # Makes the graph a rectangular grid plot with yellow and blue colour scheme
    plt.gca().set_aspect('equal')  # set equal aspect ratio
    plt.plot()  # Plots the grid
    plt.show()
    return None


def generate_grid(grid_size, p_yellow):
    # Generates a grid of random numbers - if p_yellow isn't 0, it will set them as booleans, otherwise it
    # will keep them as uniformly random numbers from 0 to 1
    colours = np.random.rand(grid_size, grid_size)  # Initialise 2d array of random numbers between 0 and 1
    if p_yellow != 0:
        colours = colours < p_yellow  # Sets each cell to 1 if it's less than p_yellow, else 0.
    return colours


def find_yellow_path(grid, show_path, testing):
    """
    For a numpy array of booleans (the yellow/blue grid), this function returns True or False
    depending on whether a yellow path exists from the left to the right edge.
    It will also display the path if you pass True
    This is not efficient
    """

    if testing:
        # gets the time when the function starts (for testing purposes)
        start_time = time.time()
    # we know all the yellows on the left are reachable
    grid_size = np.shape(grid)
    side_length = grid_size[0]
    # we make an array of the same size as the grid to store
    # which squares we can reach. Initially we assume we can't reach any
    # 0 represents a square is unreachable
    reachable = np.full(grid_size, 0)
    # yellows is a list of arrays which contain the coordinates of each yellow square
    yellows = np.argwhere(grid == True)
    # storing all the 0th column yellows as reachable
    # 1 represents a square is reachable
    for square in yellows:
        y, x = square
        if x == 0:
            reachable[y, x] = 1

    # Creates an array of all coordinates of sites already checked
    reachable_array = np.asarray(np.where(reachable == 1)).T.tolist()

    # now we search for adjacent yellow squares to the ones we already have
    # we keep searching through each element of reachable until we don't find any new squares
    end = False
    while not end:
        # the counter keeps track of how many new reachable squares are found
        # every time the loop below repeats
        counter = 0
        # iterates over our array of coordinates of reachable sites
        # assigns y and x for each site so that we can later compare to neighbouring sites
        for site in reachable_array:
            y = site[0]
            x = site[1]
            # ends the while loop as soon as a site on the RHS is shown to be reachable
            if x == side_length - 1:
                end = True
            # check adjacent
            # the first ifs in each part stops us getting index out of bounds errors
            # we check !=1 because otherwise we'd find the same squares again,
            # and it would loop forever
            else:
                if x < side_length - 1:
                    # checks right
                    if grid[y, x + 1] == True and [y, x + 1] not in reachable_array:
                        reachable[y, x + 1] = 1
                        reachable_array.append([y, x + 1])
                        counter += 1
                if y - 1 >= 0:
                    # checks below
                    if grid[y - 1, x] == True and [y - 1, x] not in reachable_array:
                        reachable[y - 1, x] = 1
                        reachable_array.append([y - 1, x])
                        counter += 1
                if y < side_length - 1:
                    # checks above
                    if grid[y + 1, x] == True and [y + 1, x] not in reachable_array:
                        reachable[y + 1, x] = 1
                        reachable_array.append([y + 1, x])
                        counter += 1
                if x - 1 >= 0:
                    # checks left
                    if grid[y, x - 1] == True and [y, x - 1] not in reachable_array:
                        reachable[y, x - 1] = 1
                        reachable_array.append([y, x - 1])
                        counter += 1
        # if no new squares are found, the while loop ends
        if counter == 0:
            end = True

    # displays the input grid if show_path is set to true
    if show_path:
        display_grid(reachable)

    # outputs how long it took to do the function if in testing mode
    if testing:
        print(time.time() - start_time)
    # this searches the last column of reachable for the value 1
    # if 1 is found then there must be a reachable square on the RHS.
    # if 1 is not found then there cannot be
    if 1 in reachable[:, side_length - 1]:
        return True
    else:
        return False


def fn(trials, n, p):
    count = 0

    for i in range(trials):
        grid = generate_grid(n, p)
        if find_yellow_path(grid, False):
            count += 1

    return 1.0 * count / trials


def plot_fn(trials, n):
    x_vals = np.arange(0.4, 0.8, 0.01)
    y_vals = [fn(trials, n, p) for p in x_vals]

    plt.plot(x_vals, y_vals)
    

def narrow_range(lower, upper, grid):
    mid = lower + (upper-lower)/2
    grid = grid < mid
    if find_path_across(grid) == True:
        upper = mid
    else:
        lower = mid
    return lower, upper


# this function only works on a grid of uniformly random numbers, not booleans!
def find_crit_point(grid, precision):
    # We know from testing that the critical point was somewhere around 0.59, definitely between 0.55 and 0.65
    # so our first upper and lower bounds will be those, and we will shrink it down.
    # accuracy represents how many times we should narrow our range
    lower_bound = 0.55
    upper_bound = 0.65
    for i in range(0, precision):
        lower_bound, upper_bound = narrow_range(lower_bound, upper_bound, grid)

    return lower_bound + (upper_bound-lower_bound)/2


def test_crit_points(samples, gridsize, precision):
    results = np.zeros(samples)
    for i in range(samples):
        grid = generate_grid(gridsize, 0)
        results[i] = find_crit_point(grid, precision)
    estimate = np.average(results)
    return results, estimate


def plot_test_results_scatter():
    results, estimate = test_crit_points(200,200,13)
    plt.scatter(np.arange(0, len(results), 1), np.sort(results), c=np.sort(results), cmap="plasma")
    plt.hlines(estimate, 0, len(results), colors="r", lw=0.7, label=str(estimate))
    plt.plot()
    plt.show()


def plot_test_results_boxes():
    results_10 = test_crit_points(100, 10, 12)[0]
    results_50 = test_crit_points(100, 50, 12)[0]
    results_100 = test_crit_points(100, 100, 12)[0]
    plt.boxplot([results_10, results_50, results_100])
    plt.plot()
    plt.show()


plot_test_results_scatter()
plot_test_results_boxes()

# function to create the breaks and connections lists using the rest of the data
def create_wires(height, num_wires, poisson_breaks, poisson_connections):
    # this section creates list of breaks in wires
    # initializes the list
    breaks = []
    # for each of the wires except the first and last (since breaks in those wouldn't matter)
    for i in range(0, num_wires - 2):
        # add an empty list to the list
        breaks.append([])
        # get a running total starting at 0 (representing the bottom of the wire)
        total = 0
        # until we pass the top of the wire
        while total < height:
            # get a random exponential as the gap between the previous break and the next one
            gap = np.random.exponential(poisson_breaks)
            # add it to the total
            total += gap
            # if you're still on the wire
            if total < height:
                # add the y coord of the break to the list
                breaks[i].append(total)

    # some notes:
    # We don't add breaks in the first or last wires as the whole first wire is powered anyway and therefore adding
    # breaks wouldn't change anything, and in the last wire, we only care if any point of it is powered.
    # Also, while it seems pointless to check if total < height again inside the while loop, if I didn't do that then
    # it would add an extra break to the list that would be greater than the height. I could fix this by then removing
    # the last one afterwards, but that would make the code less readable (although slightly more efficient).

    # this section creates list of connections between wires
    # initializes list
    connections = []
    # for each gap between the wires (so number of wires - 1)
    for i in range(0, num_wires - 1):
        # add empty list
        connections.append([])
        # make running total
        total = 0
        # very similar to the previous one
        while total < height:
            gap = np.random.exponential(poisson_connections)
            total += gap
            if total < height:
                connections[i].append(total)

    return breaks, connections


# function to draw the wires and breaks on a graph using the breaks and connections
def draw_wires(height, num_wires, breaks, connections):
    # makes the graph a nice size
    plt.figure(figsize=(7, 7))
    # gets x coords for vertical wires and puts them in a numpy array
    vertical_lines = np.arange(0, num_wires, 1)
    # plots the vertical wires
    plt.vlines(vertical_lines, 0, height, colors="k")

    # for each of the gaps between the wires
    for i in range(0, num_wires - 1):
        # draw horizontal lines on the y coords of each of the connections going between the wires
        plt.hlines(connections[i], i, i+1, colors="k")

    # break_width*2 is a nice width for the red lines that represent breaks in the wires
    break_width = 1/3
    # for each of the wires that aren't first or last
    for i in range(0, num_wires - 2):
        # draw horizontal lines on the wires where the breaks are
        plt.hlines(breaks[i], i+1 - break_width, i+1 + break_width, colors="r", lw=1)

    # plot and show the graph
    plt.plot()
    plt.show()


# similar to the previous function but instead uses vertical segments and horizontal segments, and also
# draws any of the wires that are assigned as reachable in cyan rather than black
def draw_wires_reachable(num_wires, verticals, horizontals):
    # makes figure a nice size
    plt.figure(figsize=(7, 7))
    # for each of the vertical wires
    for x in range(0, len(verticals)):
        # for each of the segments of vertical wire in the current wire
        for y in range(0, len(verticals[x])):
            # if it's powered
            if verticals[x][y][2] == 1:
                # draw it in cyan
                plt.vlines(x, verticals[x][y][0], verticals[x][y][1], colors="c")
            else:
                # if it's not powered draw it in black
                plt.vlines(x, verticals[x][y][0], verticals[x][y][1], colors="k")

    # for each of the gaps
    for x in range(0, len(horizontals)):
        # for each of the horizontal wires in the gap
        for y in range(0, len(horizontals[x])):
            # if it's powered
            if horizontals[x][y][1] == 1:
                # draw it in cyan
                plt.hlines(horizontals[x][y][0], x, x + 1, colors="c")
            else:
                # otherwise draw it in black
                plt.hlines(horizontals[x][y][0], x, x + 1, colors="k")

    # draws each of the breaks (same as previous function)
    break_width = 1/3
    for i in range(0, num_wires - 2):
        plt.hlines(breaks[i], i+1 - break_width, i + 1 + break_width, colors="r", lw=1)

    # plot and show graph
    plt.plot()
    plt.show()


# this function looks at all the horizontal wires, sees if they should be powered, and if so, powers them
def power_horizontals(verticals, horizontals):
    # changed sees if you end up changing any; if you don't turn any wires on then you are stuck forever and should stop
    changed = False
    # for each of the gaps (except the first, which all of them will be powered anyway since they're directly
    # connected to the first vertical wire which is powered)
    for x in range(1, len(horizontals)):
        # for all the horizontal wires in the gap
        for y in range(0, len(horizontals[x])):
            # if it's not powered
            if horizontals[x][y][1] == 0:
                # go through each of the vertical segments to the left
                for i in range(0, len(verticals[x])):
                    # loop until you find the one that the horizontal one is connected to
                    if verticals[x][i][1] > horizontals[x][y][0]:
                        # if that vertical segment is powered
                        if verticals[x][i][2] == 1:
                            # then power the horizontal wire
                            horizontals[x][y][1] = 1
                            # and since you've powered a wire, the state has changed, and we need to keep going
                            changed = True
                        # then either way stop looping, since any of the next vertical segments won't be connected
                        break

                # does the same as the previous section but checking the vertical wire segment on the right instead
                for i in range(0, len(verticals[x + 1])):
                    if verticals[x + 1][i][1] > horizontals[x][y][0]:
                        if verticals[x + 1][i][2] == 1:
                            horizontals[x][y][1] = 1
                            changed = True
                        break

    return horizontals, changed


# this function looks at each of the vertical segments, sees if they should get powered, and if so, powers them
def power_verticals(verticals, horizontals):
    # this is used to see if the final wire has been powered
    final_powered = False
    # for each of the long vertical wires (except the first which is always powered)
    for x in range(1, len(verticals)):
        # for each of the vertical wire segments
        for y in range(0, len(verticals[x])):
            # if the current segment is unpowered
            if verticals[x][y][2] == 0:
                # go through each of the horizontal wire segments to the left
                for i in range(0, len(horizontals[x - 1])):
                    # if the current horizontal segment is above the bottom part of the wire
                    if horizontals[x - 1][i][0] > verticals[x][y][0]:
                        # and it's below the top part of the wire
                        if horizontals[x - 1][i][0] < verticals[x][y][1]:
                            # and it's powered
                            if horizontals[x - 1][i][1] == 1:
                                # then power the vertical segment
                                verticals[x][y][2] = 1
                                # if it's in the final wire
                                if x == len(verticals) - 1:
                                    # then the final wire is powered
                                    final_powered = True
                                # since the wire is powered, we don't need to check any of the other horizontal wires
                                break
                        # if it's above the top part of the wire, we've gone past and don't need to check any others
                        else:
                            break

                # here we do the same thing but for the horizontal wires coming from the right - first however,
                # we need to check that this isn't the last wire, otherwise we'd go outside the index range
                if x != len(verticals) - 1:
                    # then we basically just do the same thing
                    for i in range(0, len(horizontals[x])):
                        if horizontals[x][i][0] > verticals[x][y][0]:
                            if horizontals[x][i][0] < verticals[x][y][1]:
                                if horizontals[x][i][1] == 1:
                                    verticals[x][y][2] = 1
                                    break
                            else:
                                break

    return verticals, final_powered


# this function determines whether a given set of breaks and connections has a path going from left to right
def path_of_current(height, breaks, connections):
    # verticals and horizontals are similar to breaks and connections, with a slight difference.
    # for horizontals, all it does is add a boolean to each connection to represent whether the wire is powered.
    # for verticals, rather than storing the breaks, it instead stores the segments of wire that are formed from
    # each of the breaks - for example, if there was one break right in the middle, then verticals would have two
    # wires with start and end points, and also a boolean to represent whether the wire is on or off.

    # first off, I convert the breaks and connections to verticals and horizontals
    # initialise lists
    verticals = []
    horizontals = []

    # the first wire will always be [0, height, 1], since it has no breaks, so it goes from 0 to the height, and
    # it's always powered.
    verticals.append([[0, height, 1]])

    # for each of breaks in a specific wire
    for breaks_in_wire in breaks:
        # if there are no breaks in our wire
        if len(breaks_in_wire) == 0:
            # then our segment is the whole wire
            verticals.append([[0, height, 0]])
        else:
            # otherwise, the first segment will go from 0 to our first break
            current_verticals = [[0, breaks_in_wire[0], 0]]
            # then each of the next segments will just be from the current break to the next
            for i in range(0, len(breaks_in_wire) - 1):
                current_verticals.append([breaks_in_wire[i], breaks_in_wire[i + 1], 0])
            # except the last, which goes from the last break to the height
            current_verticals.append([breaks_in_wire[-1], height, 0])
            # then we add all the segments of wire to our verticals
            verticals.append(current_verticals)
    # the last wire, similar to the first, has no breaks
    verticals.append([[0, height, 0]])

    # this is identical to the next part, except each horizontal wire connected to the first vertical one
    # can automatically start off powered, so this saves a bit of time.
    current_horizontals = []
    for wire in connections[0]:
        current_horizontals.append([wire, 1])
    horizontals.append(current_horizontals)

    # for each of the gaps between the wires (except for the first for the reasons above)
    for gap in connections[1:]:
        current_horizontals = []
        # for each of the horizontal wires in the gap
        for wire in gap:
            # add the y coord of this wire, and 0 (because it's off), to a temporary list
            current_horizontals.append([wire, 0])
        # then add this temporary list to the list of lists of horizontal wires
        horizontals.append(current_horizontals)

    # now we can start checking if we can reach the end!
    # we keep repeating until one of two conditions are met - if the final wire is powered, then we are done and don't
    # need to check and more wires. However, if no new horizontal wires are powered, then that means no new vertical
    # wires will be powered, so no new wires will ever be powered. Therefore, the end is not reachable. If neither
    # of these are true then we need to keep looking.
    while True:
        # powers any vertical wires that needs powering, and gets whether the final wire is powered.
        verticals, final_powered = power_verticals(verticals, horizontals)
        # if it is, then we are done, and can return the wires (we could also return "True" if that will be helpful)
        if final_powered:
            return True, verticals, horizontals

        # then powers any horizontal wires that need powering, and whether or not the board changed state
        horizontals, changed = power_horizontals(verticals, horizontals)
        # if it didn't change, then we are stuck, so we give up and return what wires we did end up powering.
        if not changed:
            return False, verticals, horizontals


"""# some example variables
# height is how tall you want each wire to be
height = 200
# num_wires is how many vertical wires across the board you want
num_wires = 100
# break_rate is the random poisson variable used to see how frequently there should be breaks in the wires
break_rate = 4.8
# similarly connect_rate determines how frequently there should be connections between the wires
connect_rate = 5
# (the smaller the number, the more frequently it will happen)

# this here just keeps trying random boards until it finds one that can reach the end, then draws it.
# I made the break rate slightly faster than the connect rate, so it will look more interesting.
while True:
    breaks, connections = create_wires(height, num_wires, break_rate, connect_rate)
    reachable, verts, horiz = path_of_current(height, breaks, connections)
    if reachable:
        draw_wires_reachable(num_wires, verts, horiz)
        break"""

def wire_percolation(height, num_wires, break_rate, connect_rate):
    breaks, connections = create_wires(height, num_wires, break_rate, connect_rate)
    return path_of_current(height, breaks, connections)[0]

def wires_Fn(trials, height, num_wires, break_rate, connect_rate):
    '''
    This function finds Fn (the probability of a path from left to right) for a given grid size and probability of yellows.
    This is an estimate based on the number of successful vs unsuccessful trials.
    '''
    # we initialise a count function to note the trials which succeed in finding a path
    count = 0
    
    # we iterate 'trials' number of times, each time creating a grid and checking whether there is a path from left to right
    for i in range(trials):
        # when there is a path we increase 'count' by 1
        if wire_percolation(height, num_wires, break_rate, connect_rate):
            count += 1
    # we return the probability of finding a path, the number of successful trials divided by the total number of trials
    return 1.0 * count/trials

def intercept_critical_point(trials, h1, h2, num_wires, connect_rate):
    
    # We find Fn at all 3 n's for 20 equally spaced probabilities between 0.5 and 0.7
    x_vals = np.linspace(2, 8, 20)
    y_vals1 = np.array([wires_Fn(trials, h1, num_wires, break_rate, connect_rate) for break_rate in x_vals])
    y_vals2 = np.array([wires_Fn(trials, h2, num_wires, break_rate, connect_rate) for break_rate in x_vals])
    
    # we create two interpolated splines to fit the values of Fn to smooth curves
    spline1 = inter.InterpolatedUnivariateSpline(x_vals, y_vals1)
    spline2 = inter.InterpolatedUnivariateSpline(x_vals, y_vals2)
    
    # This function find the absolute value of the difference between our two splines at a given point x
    def difference(x):
        return spline1(x) - spline2(x)
    
    # we use the 'fsolve' method to find the roots of our difference function starting from our estimate of 0.58
    # the roots of this function are when our two splines are equal - the intercept of the two.
    # This is our estimate for Pc
    intercept = opt.brentq(difference, 4.8, 5.2)
    
    # Plot of all three curves of Fn and our estimate of Pc
    plt.plot(x_vals, y_vals1, color='green', label=f'$height={h1}$')
    plt.plot(x_vals, y_vals2, color='cyan', label=f'$height={h2}$')
    plt.plot(intercept, spline1(intercept),'o', color = 'green')
    plt.text(intercept+0.018, spline1(intercept), f"$Critical Point = {intercept[0]:.2f}$", ha="center")
    plt.axvline(intercept, ls='--')
    plt.xlabel("Poisson Break Rate")
    plt.ylabel("Probability of Percolation")
    plt.title("Probability of Percolation for Wire Model")
    plt.legend()
   

# Continuous percolation Experimentation
# Function that generates a grid of "on" and "off" (or yellow and blue) dots that follow specific restraints:
# Choose a uniformly random x and y coordinate in the grid, and draw a dot with radius rad.
# If the dot is not overlapping any others, it chooses to be on with probability p, and off with 1-p
# If it's only overlapping dots of the same colour, it will be the same colour as them
# If it's overlapping dots of different colours, then it's an invalid position, and chooses a new spot
def create_dots_clumping(width, height, p, rad, num_dots):
    tries = 0
    # initialize the array of coordinates of dots, and their states (will be like [xcoord, ycoord, on/off])
    dot_coords = np.zeros((num_dots, 3))
    # we increase the number of dots drawn until we reach the desired amount, so start at 0 and stop when we're there
    current_dots = 0
    while current_dots < num_dots:
        # Start off with it being colourless
        colour = None
        # Clash is whether or not it's overlapping 2 different colours; it starts off not
        clash = False
        # generates a random point in the grid
        x, y = np.random.uniform(0,width), np.random.uniform(0,height)
        # Checking each of the previous dots
        for i in range(current_dots):
            # this is for optimisation - if the squares containing the dots overlap
            if abs(x - dot_coords[i][0]) + abs(y - dot_coords[i][1]) < 2*1.42*rad:
                # If the distance between the two dots is less than two radii (square both sides since sqrt is slow)
                if pow((x - dot_coords[i][0]), 2) + pow((y - dot_coords[i][1]), 2) < 4 * pow(rad, 2):
                    # If the dot is on
                    if dot_coords[i][2] == 1:
                        # and if the colour isnt already supposed to be off
                        if colour != 0:
                            # then the colour should now be on
                            colour = 1
                        # otherwise there's a clash, and we dont need to check anything else - just break and try again
                        else:
                            clash = True
                            break
                    # otherwise the dot is off
                    else:
                        # so same thing but in reverse
                        if colour != 1:
                            colour = 0
                        else:
                            clash = True
                            break
        # so long as it didnt clash, we can add it
        if not clash:
            # if we didnt assign it a colour already, then we choose it to be on with probability p
            if colour is None:
                colour = np.random.uniform() < p
            # then add the dot and it's colour to the dot coords
            dot_coords[current_dots] = [x, y, colour]
            # and since there's one more dot drawn, increase the counter by 1
            current_dots += 1
        tries += 1
    print(tries)
    return dot_coords
# this function is unfortunately extremely slow in its current state, due to the fact that each time it tries
# to add a dot it needs to check every other dot already drawn and do a distance function to see how close it is.
# to optimise it, I could do a couple of different things - one idea is instead of checking


def create_dots(width, height, num_dots)


# this takes graph dimensions, any number of arrays of dots, a list of colours and a radius size, and draws the dots
# on a graph, with the colours corresponding to list of lists of dots.
def draw_dots(width, height, dots_array, colours_array, rad):
    # gets graph stuff
    figure, axes = plt.subplots()
    # makes the graph background dark blue
    axes.set_facecolor('navy')

    # draws a circle in the correct colour at the correct coordinate
    for i, dots in enumerate(dots_array):
        for dot in dots:
            c = plt.Circle((dot[0], dot[1]), rad, color=colours_array[i])
            axes.add_artist(c)

    # makes the graph a good size
    axes.set_xlim(0, width)
    axes.set_ylim(0, height)
    axes.set_aspect(1.0)
    plt.show()


# For the purposes of finding a path from left to right, the off dots dont matter, so it would be helpful to trim
# them out of the list and only keep the useful data. (easy enough to understand)
def yellows_only(dots):
    yellows = []
    for dot in dots:
        if dot[2] == 1:
            yellows.append([dot[0], dot[1]])
    return yellows


# takes a list of currently unreachable dots, reachable dots, and the dots that were just added (and the radius),
# and sees which of the unreachable dots are reachable from the just added dots, and removes them from the unreachables
# and adds them to the reachables, and then changes it so these dots are the new "last added" dots.
def add_new_dots(unlinked_dots, linked_dots, last_added, rad):
    # I cant delete them from the array during the search since that would change how many items are in the array,
    # so I need to delete them after I've done all of the searching.
    to_delete = []
    new_dots = []
    # for each of the dots that arent yet reached
    for i in range(len(unlinked_dots)):
        # for each of the dots that were last added
        for j in range(len(last_added)):
            # if the current unlinked one is touching the current last powered
            if (unlinked_dots[i][0] - last_added[j][0]) ** 2 + (
                    unlinked_dots[i][1] - last_added[j][1]) ** 2 < 4 * rad ** 2:
                # add the unlinked dot to the linked dots
                linked_dots.append(unlinked_dots[i])
                # and also to the a temporary list that will become the last powered
                new_dots.append(unlinked_dots[i])
                # and add it's index to the array that will later bin it from the unlinked dots
                to_delete.append(i)
                # then since it's powered, we dont care if any other powered ones are touching it, so move on
                break
    # for each of the indexes that need deleting (in reverse order so it wont shift the index of other ones in the list)
    for i in to_delete[::-1]:
        # delete the corresponding dot in the unlinked dots
        del unlinked_dots[i]
    return unlinked_dots, linked_dots, new_dots


# this function determines if there's a path of dots on a grid of "on" dots with a radius rad
def yellow_dot_path(width, height, dots, rad):
    # all dots start off unlinked
    unlinked_dots = dots

    # we start off by adding all dots that are touching the line x=0 to our linked dots - these are ones where the
    # x coordinate is < radius - so do a very similar thing to add_new_dots but taking this into account.
    linked_dots = []
    last_added = []
    to_remove = []

    # for each of the unlinked dots
    for i in range(len(unlinked_dots)):
        # if the dot is touching the left side
        if dots[i][0] < rad:
            # add it to linked dots
            linked_dots.append(unlinked_dots[i])
            # add it to last added
            last_added.append(unlinked_dots[i])
            # delete it from unlinked dots later
            to_remove.append(i)

    # delete all of the newly linked dots from unlinked dots
    for i in to_remove[::-1]:
        del unlinked_dots[i]

    # you start off not reaching the end
    reached_end = False
    # while there are still unexplored dots and the end hasn't been reached
    while len(last_added) != 0 and not reached_end:
        # update the unlinked dots, linked dots, and last added dots
        unlinked_dots, linked_dots, last_added = add_new_dots(unlinked_dots, linked_dots, last_added, rad)
        # for each of the newly added dots
        for coords in last_added:
            # if any of them are touching the edge of the grid
            if coords[0] > width - rad:
                # then it's reachable!
                reached_end = True

    # draw the reached dots in gold, and the unreached ones in crimson, just to demonstrate the function works
    draw_dots(width, height, [unlinked_dots, linked_dots], ["crimson", "gold"], rad)


width, height = 100, 100
p_yellow = 0.55
dot_radius = 1.5
num_dots = 15000

dots = yellows_only(create_dots_clumping(width, height, p_yellow, dot_radius, num_dots))
yellow_dot_path(width, height, dots, dot_radius)

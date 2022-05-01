# Import libraries
import numpy as np
import matplotlib.pyplot as plt
import time


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
    """
    Given grid size and probability of a square being yellow, this outputs
    a numpy array of booleans, with true representing yellow
    """

    colours = np.random.rand(grid_size, grid_size)  # Initialise 2d array of random numbers between 0 and 1
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
    break_width = 1/6
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
    break_width = 1/6
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


# some example variables
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
        break

# simple implementation of Conway's Game of Life
# using numpy and matplotlib
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation


# setting the size of the grid
N = 100
# setting the number of generations
num_gen = 100

# filling the grid with random values
grid = np.random.choice([0, 1], N * N, p=[0.2, 0.8]).reshape(N, N)

# setting up the animation
fig, ax = plt.subplots()
mat = ax.matshow(grid)

def get_neighbours(grid, i, j):
    return grid[i-1:i+2, j-1:j+2]


def update(data):
    global grid
    # newGrid = grid.copy()
    print(grid.shape)

# def update(data):
#     global grid
#     # copy grid since we require 8 neighbors
#     # for calculation and we go line by line
#     newGrid = grid.copy()
#     for i in range(N):
#         for j in range(N):
#             # compute 8-neghbor sum
#             # using toroidal boundary conditions - x and y wrap around
#             # so that the simulaton takes place on a toroidal surface.
#             total = int((grid[i, (j - 1) % N] + grid[i, (j + 1) % N] +
#                             grid[(i - 1) % N, j] + grid[(i + 1) % N, j] +
#                             grid[(i - 1) % N, (j - 1) % N] + grid[(i - 1) % N, (j + 1) % N] +
#                             grid[(i + 1) % N, (j - 1) % N] + grid[(i + 1) % N, (j + 1) % N]) / 255)
#             # apply Conway's rules
#             if grid[i, j] == 1:
#                 if (total < 2) or (total > 3):
#                     newGrid[i, j] = 0
#             else:
#                 if total == 3:
#                     newGrid[i, j] = 1
#     # update data
#     mat.set_data(newGrid)
#     grid = newGrid
#     return [mat]

ani = animation.FuncAnimation(fig, update,frames=num_gen, interval=50)
plt.show()
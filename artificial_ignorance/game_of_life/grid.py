# simple implementation of Conway's Game of Life
# using numpy and matplotlib
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation




# setting the size of the grid
N = 20

# filling the grid with random values
grid = np.random.choice([0, 1], N * N, p=[0.2, 0.8]).reshape(N, N)
# N=3
# grid = np.array([[0, 0, 0], [0, 1, 0], [0, 0, 0]])

# setting up the animation
fig, ax = plt.subplots()
mat = ax.imshow(grid)

def get_neighbours(grid, x, y):
    return grid[x-1:x+1, y-1:y+1]

def update(data):
    global grid
    newGrid = grid.copy()
    # newGrid = grid.copy()
    x, y = grid.shape
    for x_coord in range(x):
        for y_coord in range(y):
            neighbours = get_neighbours(grid, x_coord, y_coord)
            # print(neighbours)
            # print(neighbours.sum())
            # print(neighbours.sum() - grid[x_coord, y_coord])
            if neighbours.sum() - grid[x_coord, y_coord] < 2:
                newGrid[x_coord, y_coord] = 0
            elif neighbours.sum() - grid[x_coord, y_coord] > 3:
                newGrid[x_coord, y_coord] = 0
            elif neighbours.sum() - grid[x_coord, y_coord] == 3:
                newGrid[x_coord, y_coord] = 1
    mat.set_data(newGrid)
    
    # return newGrid

ani = animation.FuncAnimation(fig, update, frames=1000, interval=2)
plt.show()
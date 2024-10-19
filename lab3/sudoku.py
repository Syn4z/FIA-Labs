from utils import *


def sudokuBacktracking(grid, row, col):
    if (row == N - 1 and col == N):
        return True
    if col == N:
        row += 1
        col = 0
    if grid[row][col] > 0:
        return sudokuBacktracking(grid, row, col + 1)
    for num in range(1, N + 1, 1):
        if isValid(grid, row, col, num):
            grid[row][col] = num
            if sudokuBacktracking(grid, row, col + 1):
                return True
        grid[row][col] = 0
    return False

def sudokuBacktrackingDomains(grid, row=0, col=0):
    domains = initializeDomains(grid)
    domains = propagateConstraints(domains)
    return backtrack(grid, domains, row, col)  

def sudokuForwardChecking(grid, row, col):
    domains = initializeDomains(grid)
    domains = propagateConstraints(domains)
    return forwardCheckSolver(grid, domains, row, col) 

def sudokuHeuristic(grid):
    domains = initializeDomains(grid)
    domains = propagateConstraints(grid, domains)
    return heuristicBacktrack(grid, domains)
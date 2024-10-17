from utils import *
import time
import glob


if __name__ == "__main__":
    grids = glob.glob('grid*.txt')
    backtrackTime = []
    domainsTime = []
    for grid in grids:
        gridName = grid.split('/')[-1]
        gridString = readGridFromFile(grid)
        grid = parseGrid(gridString)

        start_time = time.time()
        if solveSudokuBasic(grid, 0, 0):
            print(f"\nSolution for '{gridName}' using backtracking: ")
            printing(grid)
            end_time = time.time()
            backtrackTime.append(end_time - start_time)
        else:
            end_time = time.time()
            print(f"No solution exists for '{gridName}'")
            
        grid = parseGrid(gridString)
        start_time = time.time()
        if solveSudoku(grid):
            print(f"\nSolution for '{gridName}' using domains: ")
            printing(grid)
            end_time = time.time()
            domainsTime.append(end_time - start_time)
        else:
            end_time = time.time()
            print(f"No solution exists for '{gridName}'")

    print(f"Backtracking time: {backtrackTime} s\nDomains time: {domainsTime} s")    
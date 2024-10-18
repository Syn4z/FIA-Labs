from sudoku import *
from generateGrid import *
import time
import glob
import os


if __name__ == "__main__":
    grids = glob.glob('grids/grid*.txt')
    def resolveSudokuTeq(teq, grids):
        teqTime = []
        for grid in grids:
            gridName = grid.split('/')[-1]
            gridString = readGridFromFile(grid)
            grid = parseGrid(gridString)

            if grid is None:
                print(f"Skipping '{gridName}' due to parsing error")
                teqTime.append(None)
                continue

            try:
                start_time = time.time()
                if teq(grid, 0, 0):
                    end_time = time.time()
                    teqTime.append(f"{end_time - start_time:.4f}")
                    solution_filename = getSolutionFilename(gridName)
                    saveSolutionToFile(grid, solution_filename)
                else:
                    end_time = time.time()
                    print(f"'{gridName}' is not solvable using {teq.__name__}")
                    teqTime.append(None)
            except Exception as e:
                print(f"Error solving '{gridName}': {e}")
                teqTime.append(None)
        return teqTime

    backtrackTime = resolveSudokuTeq(sudokuBacktracking, grids)
    domainsTime = resolveSudokuTeq(sudokuDomains, grids)
    # forwardCheckingTime = resolveSudokuTeq(sudokuForwardChecking, grids)
    print("Solutions are saved in 'solutions' folder")
    print(f"Backtracking times: {backtrackTime} in seconds\nDomains times: {domainsTime} in seconds")  
    # print(f"Forward checking time: {forwardCheckingTime} in seconds")  

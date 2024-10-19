from sudoku import *
from generateGrid import *
import time
import glob


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

if __name__ == "__main__":
    grids = glob.glob('grids/grid*.txt')
    grid_names = [grid.split('\\')[-1] for grid in grids]
    grid_names = [name.replace('.txt', '') for name in grid_names]
    backtrackTime = resolveSudokuTeq(sudokuBacktracking, grids)
    domainsTime = resolveSudokuTeq(sudokuBacktrackingDomains, grids)
    forwardCheckingTime = resolveSudokuTeq(sudokuForwardChecking, grids)
    heuristicBacktrackTime = resolveSudokuTeq(sudokuHeuristic, grids)
    data = {
        'Backtracking': backtrackTime,
        'Backtracking wit Constraints': domainsTime,
        'Forward Checking': forwardCheckingTime,
        'Heuristic': heuristicBacktrackTime
    }
    table = [[algo] + [f"{time} s" for time in times] for algo, times in data.items()]
    table.insert(0, [""] + grid_names)
    print(tabulate(table, headers="firstrow", tablefmt="grid"))  

from sudoku import *
from generateGrid import *
import time
import glob


def resolveSudokuTeq(teq, grids):
    teqTime = []
    teqSteps = []
    for grid in grids:
        gridName = grid.split('/')[-1]
        gridString = readGridFromFile(grid)
        grid = parseGrid(gridString)

        if grid is None:
            print(f"Skipping '{gridName}' due to parsing error")
            teqTime.append(None)
            teqSteps.append(None)
            continue

        try:
            start_time = time.time()
            solved, steps = teq(grid, 0, 0)
            end_time = time.time()
            if solved:
                teqTime.append(f"{end_time - start_time:.4f}")
                teqSteps.append(steps)
                solution_filename = getSolutionFilename(gridName)
                saveSolutionToFile(grid, solution_filename)
            else:
                print(f"'{gridName}' is not solvable using {teq.__name__}")
                teqTime.append(None)
                teqSteps.append(None)
        except Exception as e:
            print(f"Error solving '{gridName}': {e}")
            teqTime.append(None)
            teqSteps.append(None)
    return teqTime, teqSteps

if __name__ == "__main__":
    grids = glob.glob('grids/grid*.txt')
    grid_names = [grid.split('\\')[-1] for grid in grids]
    grid_names = [name.replace('.txt', '') for name in grid_names]
    backtrackTime, backtrackSteps = resolveSudokuTeq(sudokuBacktracking, grids)
    domainsTime, domainsSteps = resolveSudokuTeq(sudokuBacktrackingDomains, grids)
    forwardCheckingTime, forwardCheckingSteps = resolveSudokuTeq(sudokuForwardChecking, grids)
    heuristicBacktrackTime, heuristicBacktrackSteps = resolveSudokuTeq(sudokuHeuristic, grids)
    ac3Time, ac3Steps = resolveSudokuTeq(sudokuAc3, grids)
    data = {
        'Backtracking': (backtrackTime, backtrackSteps),
        'Backtracking with Constraints': (domainsTime, domainsSteps),
        'Forward Checking': (forwardCheckingTime, forwardCheckingSteps),
        'Heuristic': (heuristicBacktrackTime, heuristicBacktrackSteps),
        'Arc Consistency-3': (ac3Time, ac3Steps)
    }

    table = [[algo] + [f"{time} s\n({steps} steps)" if time is not None else "N/A" for time, steps in zip(times, steps)] for algo, (times, steps) in data.items()]
    table.insert(0, ["Methods\\grids"] + grid_names)
    colalign = ["center"] * len(table[0])
    table_str = tabulate(table, headers="firstrow", tablefmt="grid", colalign=colalign)

    print('Results table in "results.txt"')
    with open('results.txt', 'w') as file:
        file.write(table_str)

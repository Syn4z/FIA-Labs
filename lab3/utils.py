import glob
import re
import copy
from tabulate import tabulate


N = 9

def create_table(data, algorithms, grid_names):
    headers = ['Algorithm'] + grid_names
    table_data = []
    for algorithm in algorithms:
        row = [algorithm.__name__]
        for grid_name in grid_names:
            row.append(data[algorithm.__name__].get(grid_name, 'N/A'))
        table_data.append(row)
    table = tabulate(table_data, headers, tablefmt='grid')
    return table

def getNextGridFilename():
    files = glob.glob('grids/grid*.txt')
    max_num = 0
    for file in files:
        match = re.search(r'grid(\d+)\.txt', file)
        if match:
            num = int(match.group(1))
            if num > max_num:
                max_num = num
    return f'grids/grid{max_num + 1}.txt'

def getSolutionFilename(grid_name):
    grid_num = re.search(r'grid(\d+)\.txt', grid_name).group(1)
    return f'solutions/solution{grid_num}.txt'

def saveSolutionToFile(grid, filename):
    with open(filename, 'w') as file:
        for row in grid:
            file.write(''.join(['*' if num == 0 else str(num) for num in row]) + '\n')

def readGridFromFile(filename):
    with open(filename, 'r') as file:
        return file.read()
    
def parseGrid(gridString):
    grid = []
    try:
        rows = gridString.strip().split('\n')
        for row in rows:
            grid.append([0 if char == '*' else int(char) for char in row])
        # Check if the grid is 9x9
        if len(grid) != 9 or any(len(row) != 9 for row in grid):
            raise ValueError("Grid is not 9x9")
    except ValueError as e:
        print(f"Error parsing grid: {e}")
        return None
    return grid

def isValid(grid, row, col, num):
    for x in range(9):
        if grid[row][x] == num:
            return False
    for y in range(9):
        if grid[y][col] == num:
            return False
    startRow = row - row % 3
    startCol = col - col % 3
    for i in range(3):
        for j in range(3):
            if grid[i + startRow][j + startCol] == num:
                return False
    return True

def backtrack(grid, domains, row, col):
    if row == N - 1 and col == N:
        return True
    if col == N:
        row += 1
        col = 0
    if grid[row][col] > 0:
        return backtrack(grid, domains, row, col + 1)
    for num in domains[row][col]:
        if isValid(grid, row, col, num):
            grid[row][col] = num
            if backtrack(grid, domains, row, col + 1):
                return True
            grid[row][col] = 0
    return False

def forwardCheckSolver(grid, domains, row, col):
    if row == N - 1 and col == N:
        return True
    if col == N:
        row += 1
        col = 0
    if grid[row][col] > 0:
        return forwardCheckSolver(grid, domains, row, col + 1)
    
    for num in list(domains[row][col]):
        if isValid(grid, row, col, num):
            grid[row][col] = num
            affected_domains = forwardCheck(domains, row, col, num)
            if affected_domains is not None:
                # Check if all domains are non-empty before making the recursive call
                if all(domains[i][j] for i in range(N) for j in range(N) if grid[i][j] == 0):
                    if forwardCheckSolver(grid, domains, row, col + 1):
                        return True
                revertDomains(domains, affected_domains)
            grid[row][col] = 0
    return False

def initializeDomains(grid):
    domains = [[set(range(1, 10)) for _ in range(N)] for _ in range(N)]
    for i in range(N):
        for j in range(N):
            if grid[i][j] != 0:
                domains[i][j] = {grid[i][j]}
    return domains

def propagateConstraints(domains):
    changed = True
    while changed:
        changed = False
        for i in range(N):
            for j in range(N):
                if len(domains[i][j]) == 1:
                    value = next(iter(domains[i][j]))
                    # Eliminate value from row
                    for col in range(N):
                        if col != j and value in domains[i][col]:
                            domains[i][col].remove(value)
                            changed = True
                    # Eliminate value from column
                    for row in range(N):
                        if row != i and value in domains[row][j]:
                            domains[row][j].remove(value)
                            changed = True
                    # Eliminate value from 3x3 subgrid
                    startRow, startCol = 3 * (i // 3), 3 * (j // 3)
                    for row in range(startRow, startRow + 3):
                        for col in range(startCol, startCol + 3):
                            if (row != i or col != j) and value in domains[row][col]:
                                domains[row][col].remove(value)
                                changed = True
    return domains

def forwardCheck(domains, row, col, value):
    affected_domains = []

    for i in range(N):
        if i != col and value in domains[row][i]:
            domains[row][i].remove(value)
            affected_domains.append((row, i, value))
            if not domains[row][i]:
                revertDomains(domains, affected_domains)
                return None
    for i in range(N):
        if i != row and value in domains[i][col]:
            domains[i][col].remove(value)
            affected_domains.append((i, col, value))
            if not domains[i][col]:
                revertDomains(domains, affected_domains)
                return None

    # Update the domains for the 3x3 subgrid
    startRow, startCol = 3 * (row // 3), 3 * (col // 3)
    for i in range(startRow, startRow + 3):
        for j in range(startCol, startCol + 3):
            if (i != row or j != col) and value in domains[i][j]:
                domains[i][j].remove(value)
                affected_domains.append((i, j, value))
                if not domains[i][j]:
                    revertDomains(domains, affected_domains)
                    return None

    return affected_domains

def revertDomains(domains, affected_domains):
    for row, col, value in affected_domains:
        domains[row][col].add(value)

def copyDomains(domains):
    return copy.deepcopy(domains)

def heuristicBacktrack(grid, domains):
    cell = selectUnassignedVariable(domains)
    if cell is None:
        return True
    row, col = cell
    for num in list(domains[row][col]):
        if isValid(grid, row, col, num):
            grid[row][col] = num
            original_domains = copyDomains(domains)
            if forwardCheck(domains, row, col, num):
                if heuristicBacktrack(grid, domains):
                    return True
            grid[row][col] = 0
            domains = original_domains
    return False

def selectUnassignedVariable(domains):
    min_domain_size = float('inf')
    selected_cell = None
    for i in range(N):
        for j in range(N):
            if len(domains[i][j]) > 1 and len(domains[i][j]) < min_domain_size:
                min_domain_size = len(domains[i][j])
                selected_cell = (i, j)
    return selected_cell
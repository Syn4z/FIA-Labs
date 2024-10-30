import glob
import re
from tabulate import tabulate
from collections import deque

# Define the grid size for a 9x9 Sudoku
N = 9

def create_table(data, algorithms, grid_names):
    """
    Generate a table that displays the performance of different algorithms
    on various Sudoku grids.

    Args:
        data (dict): A dictionary where keys are algorithm names and values are dictionaries
                     mapping grid names to performance data.
        algorithms (list): A list of algorithm functions.
        grid_names (list): A list of grid names.

    Returns:
        str: A formatted table displaying algorithm performance.
    """
    headers = ['Algorithm'] + grid_names
    table_data = []
    for algorithm in algorithms:
        row = [algorithm.__name__]
        for grid_name in grid_names:
            row.append(data[algorithm.__name__].get(grid_name, 'N/A'))
        table_data.append(row)
    table = tabulate(table_data, headers, tablefmt='grid')
    return table

def wrap_text(text, length):
    words = text.split()
    lines = []
    current_line = []
    current_length = 0

    for word in words:
        if current_length + len(word) + len(current_line) > length:
            lines.append(' '.join(current_line))
            current_line = [word]
            current_length = len(word)
        else:
            current_line.append(word)
            current_length += len(word)

    if current_line:
        lines.append(' '.join(current_line))

    return '\n'.join(lines)

def getNextGridFilename():
    """
    Generate the filename for the next Sudoku grid to be saved,
    based on the existing grid files in the directory.

    Returns:
        str: The filename for the next grid in the format 'grids/grid{number}.txt'.
    """
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
    """
    Generate the filename for saving a solution based on the grid's filename.

    Args:
        grid_name (str): The filename of the Sudoku grid.

    Returns:
        str: The filename for the solution in the format 'solutions/solution{number}.txt'.
    """
    grid_num = re.search(r'grid(\d+)\.txt', grid_name).group(1)
    return f'solutions/solution{grid_num}.txt'

def saveSolutionToFile(grid, filename):
    """
    Save a solved Sudoku grid to a file, replacing empty cells with '*' characters.

    Args:
        grid (list): The 9x9 solved Sudoku grid.
        filename (str): The filename where the grid will be saved.
    """
    with open(filename, 'w') as file:
        for row in grid:
            file.write(''.join(['*' if num == 0 else str(num) for num in row]) + '\n')

def readGridFromFile(filename):
    """
    Read a Sudoku grid from a file.

    Args:
        filename (str): The file containing the Sudoku grid.

    Returns:
        str: The grid as a string.
    """
    with open(filename, 'r') as file:
        return file.read()
    
def parseGrid(gridString):
    """
    Parse a grid string into a 9x9 list of lists, with empty cells represented as 0.

    Args:
        gridString (str): The string representation of the Sudoku grid.

    Returns:
        list: A 9x9 grid represented as a list of lists or None if an error occurs.
    """
    grid = []
    try:
        rows = gridString.strip().split('\n')
        for row in rows:
            grid.append([0 if char == '*' else int(char) for char in row])
        if len(grid) != 9 or any(len(row) != 9 for row in grid):
            raise ValueError("Grid is not 9x9")
    except ValueError as e:
        print(f"Error parsing grid: {e}")
        return None
    return grid

def isValid(grid, row, col, num):
    """
    Check if placing a number in the given row and column is valid
    according to Sudoku rules.

    Args:
        grid (list): The current Sudoku grid.
        row (int): The row index.
        col (int): The column index.
        num (int): The number to place.

    Returns:
        bool: True if the placement is valid, False otherwise.
    """
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

def initializeDomains(grid):
    """
    Initialize the domains for each cell in the grid. If a cell is filled,
    its domain will only contain the number already placed in the cell.

    Args:
        grid (list): The 9x9 Sudoku grid.

    Returns:
        list: A 9x9 list of sets representing the domain of each cell.
    """
    domains = [[set(range(1, 10)) for _ in range(N)] for _ in range(N)]
    for i in range(N):
        for j in range(N):
            if grid[i][j] != 0:
                domains[i][j] = {grid[i][j]}
    return domains

def constraintPropagation(domains):
    """
    Apply constraint propagation by removing values from the domains
    based on constraints from neighboring cells.

    Args:
        domains (list): The 9x9 list of sets representing domains of each cell.

    Returns:
        list: The updated domains after constraint propagation.
    """
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

def backtrack(grid, domains, row=0, col=0, steps=[0]):
    """
    Recursive backtracking algorithm to solve Sudoku.
    
    Args:
    grid (list of lists): The Sudoku board.
    domains (list of sets): The possible values for each cell.
    row (int): The current row to solve.
    col (int): The current column to solve.
    steps (list): A list to track the number of steps taken.

    Returns:
    tuple: A tuple containing a boolean indicating if the puzzle is solved, 
           and the total number of steps taken.
    """
    steps[0] += 1
    if row == N - 1 and col == N:
        return True, steps[0]  # Puzzle solved
    if col == N:
        row += 1
        col = 0
    if grid[row][col] > 0:  # Skip pre-filled cells
        return backtrack(grid, domains, row, col + 1, steps)
    
    # Try each number in the domain of the current cell
    for num in list(domains[row][col]):
        if isValid(grid, row, col, num):  # Check if the number is valid
            grid[row][col] = num
            if backtrack(grid, domains, row, col + 1, steps)[0]:
                return True, steps[0]  # Continue solving if valid
            grid[row][col] = 0  # Undo assignment (backtrack)
    return False, steps[0]


def forwardCheckSolver(grid, domains, row=0, col=0, steps=[0]):
    """
    Solves Sudoku using backtracking with forward checking.
    
    Args:
    grid (list of lists): The Sudoku board.
    domains (list of sets): The possible values for each cell.
    row (int): The current row to solve.
    col (int): The current column to solve.
    steps (list): A list to track the number of steps taken.

    Returns:
    tuple: A tuple containing a boolean indicating if the puzzle is solved,
           and the total number of steps taken.
    """
    steps[0] += 1
    if row == N - 1 and col == N:
        return True, steps[0]
    if col == N:
        row += 1
        col = 0
    if grid[row][col] > 0:  # Skip pre-filled cells
        return forwardCheckSolver(grid, domains, row, col + 1, steps)
    
    for num in list(domains[row][col]):
        if isValid(grid, row, col, num):  # Check if the number is valid
            grid[row][col] = num
            is_valid, affected_cells = forwardCheck(domains, row, col, num)
            if is_valid:
                if forwardCheckSolver(grid, domains, row, col + 1, steps)[0]:
                    return True, steps[0]
            grid[row][col] = 0  # Undo assignment (backtrack)
            restoreDomains(domains, affected_cells, num)
    return False, steps[0]

def forwardCheck(domains, row, col, num):
    """
    Update the domains after placing 'num' in grid[row][col].
    If any domain becomes empty, return False (invalid placement).
    
    Args:
    domains (list of sets): The possible values for each cell.
    row (int): The current row.
    col (int): The current column.
    num (int): The number to be placed.

    Returns:
    tuple: A boolean indicating if the placement is valid,
           and a list of affected cells for domain restoration.
    """
    affected_cells = []
    # Check row
    for c in range(N):
        if c != col and num in domains[row][c]:
            domains[row][c].remove(num)
            affected_cells.append((row, c))
            if len(domains[row][c]) == 0:
                return False, affected_cells

    # Check column
    for r in range(N):
        if r != row and num in domains[r][col]:
            domains[r][col].remove(num)
            affected_cells.append((r, col))
            if len(domains[r][col]) == 0:
                return False, affected_cells

    # Check 3x3 subgrid
    startRow, startCol = 3 * (row // 3), 3 * (col // 3)
    for r in range(startRow, startRow + 3):
        for c in range(startCol, startCol + 3):
            if (r != row or c != col) and num in domains[r][c]:
                domains[r][c].remove(num)
                affected_cells.append((r, c))
                if len(domains[r][c]) == 0:
                    return False, affected_cells
    return True, affected_cells

def restoreDomains(domains, affected_cells, num):
    """
    Restore the domains by adding 'num' back to the affected cells.
    
    Args:
    domains (list of sets): The possible values for each cell.
    affected_cells (list of tuples): The cells whose domains were affected.
    num (int): The number to restore.
    """
    for r, c in affected_cells:
        domains[r][c].add(num)

def propagateUniqueCandidates(domains):
    """
    Apply the unique candidates heuristic.
    If a number is a candidate in only one cell of a unit (row, column, or block), 
    assign it to that cell.
    
    Args:
    domains (list of sets): The possible values for each cell.
    """
    # For each row, column, and block, check for unique candidates
    for i in range(N):
        checkUniqueInUnit(domains, [(i, j) for j in range(N)])  # Row
        checkUniqueInUnit(domains, [(j, i) for j in range(N)])  # Column
    for row_block in range(0, N, 3):
        for col_block in range(0, N, 3):
            block_cells = [(row_block + i, col_block + j) for i in range(3) for j in range(3)]
            checkUniqueInUnit(domains, block_cells)

def checkUniqueInUnit(domains, cells):
    """
    Check for unique candidates in a unit (row, column, or block).
    
    Args:
    domains (list of sets): The possible values for each cell.
    cells (list of tuples): The cells in the unit.
    """
    candidate_count = {}
    for row, col in cells:
        for candidate in domains[row][col]:
            if candidate not in candidate_count:
                candidate_count[candidate] = []
            candidate_count[candidate].append((row, col))
    
    for candidate, positions in candidate_count.items():
        if len(positions) == 1:  # If the candidate can only be placed in one cell
            row, col = positions[0]
            domains[row][col] = {candidate}

def propagateNakedPairs(domains):
    """
    Apply the naked pairs heuristic.
    If two cells in a unit (row, column, or block) contain the same pair of candidates,
    remove those candidates from the other cells in the unit.
    
    Args:
    domains (list of sets): The possible values for each cell.
    """
    for i in range(N):
        findNakedPairs(domains, [(i, j) for j in range(N)])  # Row
        findNakedPairs(domains, [(j, i) for j in range(N)])  # Column
    for row_block in range(0, N, 3):
        for col_block in range(0, N, 3):
            block_cells = [(row_block + i, col_block + j) for i in range(3) for j in range(3)]
            findNakedPairs(domains, block_cells)


def findNakedPairs(domains, cells):
    """
    Find and propagate naked pairs in a unit.
    
    Args:
    domains (list of sets): The possible values for each cell.
    cells (list of tuples): The cells in the unit.
    """
    pairs = {}
    for row, col in cells:
        if len(domains[row][col]) == 2:  # If the cell has exactly two candidates
            pair = tuple(domains[row][col])
            if pair not in pairs:
                pairs[pair] = []
            pairs[pair].append((row, col))
    
    # Eliminate pairs from other cells in the unit
    for pair, positions in pairs.items():
        if len(positions) == 2:  # If exactly two cells have the same pair of candidates
            for row, col in cells:
                if (row, col) not in positions:
                    domains[row][col] -= set(pair)


def propagateConstraintsHeuristics(domains):
    """
    Apply constraint propagation heuristics such as unique candidates and naked pairs.
    
    Args:
    domains (list of sets): The possible values for each cell.
    
    Returns:
    list of sets: Updated domains after applying constraint propagation.
    """
    changed = True
    while changed:
        changed = False
        for i in range(N):
            for j in range(N):
                if len(domains[i][j]) == 1:  # If a cell has only one possible value
                    value = next(iter(domains[i][j]))
                    # Eliminate value from row, column, and block
                    for col in range(N):
                        if col != j and value in domains[i][col]:
                            domains[i][col].remove(value)
                            changed = True
                    for row in range(N):
                        if row != i and value in domains[row][j]:
                            domains[row][j].remove(value)
                            changed = True
                    startRow, startCol = 3 * (i // 3), 3 * (j // 3)
                    for row in range(startRow, startRow + 3):
                        for col in range(startCol, startCol + 3):
                            if (row != i or col != j) and value in domains[row][col]:
                                domains[row][col].remove(value)
                                changed = True

        # Apply advanced heuristics
        propagateUniqueCandidates(domains)
        propagateNakedPairs(domains)
    return domains

def ac3(domains):
    """
    Enforce arc consistency on the domains.
    Returns True if arc consistency is achieved, otherwise False.
    """
    queue = deque()
    
    # Initialize the queue with all arcs
    for row in range(N):
        for col in range(N):
            for value in domains[row][col]:
                for (n_row, n_col) in get_neighbors(row, col):
                    queue.append((row, col, n_row, n_col))

    while queue:
        (xi, xj, xk, xl) = queue.popleft()
        if revise(domains, xi, xj, xk, xl):
            if len(domains[xi][xj]) == 0:
                return False  # If any domain becomes empty, failure
            for (n_row, n_col) in get_neighbors(xi, xj):
                if (n_row, n_col) != (xk, xl):
                    queue.append((n_row, n_col, xi, xj))
    return True

def get_neighbors(row, col):
    """
    Get all neighboring cells that need to be checked for arc consistency.
    """
    neighbors = set()
    
    # Row and Column Neighbors
    for i in range(N):
        if i != col:
            neighbors.add((row, i))
        if i != row:
            neighbors.add((i, col))
    
    # 3x3 Box Neighbors
    start_row, start_col = 3 * (row // 3), 3 * (col // 3)
    for i in range(3):
        for j in range(3):
            r, c = start_row + i, start_col + j
            if (r, c) != (row, col):
                neighbors.add((r, c))
    
    return neighbors

def revise(domains, xi, xj, xk, xl):
    """
    Revise the domain of xi with respect to xj.
    
    Args:
        xi: The variable whose domain is being revised.
        xj: The variable with respect to which xi's domain is being revised.
        domains: A dictionary representing the domains of all variables.
    
    Returns:
        bool: True if a value was removed from the domain of xi, False otherwise.
    """
    revised = False
    for value in list(domains[xi][xj]):
        # Check if there is no value in xj that allows a consistent assignment
        if not any(is_valid_assignment(value, domains[xk][xl], xj) for (xk, xl) in get_neighbors(xi, xj)):
            domains[xi][xj].remove(value)
            revised = True
    return revised

def is_valid_assignment(value, neighbor_domain, neighbor):
    """
    Check if a value is a valid assignment for the neighbor's domain.
    
    Args:
        value: The value to be checked.
        neighbor_domain: The domain of the neighbor variable.
        neighbor: The neighbor variable.
    
    Returns:
        bool: True if the value is a valid assignment, False otherwise.
    """
    return any(value != neighbor_value for neighbor_value in neighbor_domain)

def constraintPropagationAc3(domains):
    """
    Perform constraint propagation using the AC-3 algorithm.
    
    Args:
        domains: A dictionary representing the domains of all variables.
    
    Returns:
        dict: The revised domains after applying the AC-3 algorithm.
    """
    changed = True
    while changed:
        changed = False
        if not ac3(domains):  # Ensure arc consistency
            return domains  # Early exit if arc consistency fails
        
        for i in range(N):
            for j in range(N):
                if len(domains[i][j]) == 1:
                    value = next(iter(domains[i][j]))
                    # Eliminate value from row, column, and block
                    for col in range(N):
                        if col != j and value in domains[i][col]:
                            domains[i][col].remove(value)
                            changed = True
                    for row in range(N):
                        if row != i and value in domains[row][j]:
                            domains[row][j].remove(value)
                            changed = True
                    startRow, startCol = 3 * (i // 3), 3 * (j // 3)
                    for row in range(startRow, startRow + 3):
                        for col in range(startCol, startCol + 3):
                            if (row != i or col != j) and value in domains[row][col]:
                                domains[row][col].remove(value)
                                changed = True
    return domains

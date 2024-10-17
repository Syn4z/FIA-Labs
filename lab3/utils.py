# N is the size of the 2D matrix   N*N
N = 9

# A utility function to print grid
def printing(arr):
    for i in range(N):
        for j in range(N):
            print(arr[i][j], end = " ")
        print()

# Checks whether it will be
# legal to assign num to the
# given row, col
def isSafe(grid, row, col, num):
  
    # Check if we find the same num
    # in the similar row , we
    # return false
    for x in range(9):
        if grid[row][x] == num:
            return False

    # Check if we find the same num in
    # the similar column , we
    # return false
    for y in range(9):
        if grid[y][col] == num:
            return False

    # Check if we find the same num in
    # the particular 3*3 matrix,
    # we return false
    startRow = row - row % 3
    startCol = col - col % 3
    for i in range(3):
        for j in range(3):
            if grid[i + startRow][j + startCol] == num:
                return False
    return True

def initializeDomains(grid):
    domains = [[set(range(1, 10)) for _ in range(N)] for _ in range(N)]
    for i in range(N):
        for j in range(N):
            if grid[i][j] != 0:
                domains[i][j] = {grid[i][j]}
    return domains

def propagateConstraints(grid, domains):
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

# Takes a partially filled-in grid and attempts
# to assign values to all unassigned locations in
# such a way to meet the requirements for
# Sudoku solution (non-duplication across rows,
# columns, and boxes) */
def solveSudoku(grid, row=0, col=0):
    domains = initializeDomains(grid)
    domains = propagateConstraints(grid, domains)
    return backtrackSolve(grid, domains, row, col)

def solveSudokuBasic(grid, row, col):
  
    # Check if we have reached the 8th
    # row and 9th column (0
    # indexed matrix) , we are
    # returning true to avoid
    # further backtracking
    if (row == N - 1 and col == N):
        return True
      
    # Check if column value  becomes 9 ,
    # we move to next row and
    # column start from 0
    if col == N:
        row += 1
        col = 0

    # Check if the current position of
    # the grid already contains
    # value >0, we iterate for next column
    if grid[row][col] > 0:
        return solveSudokuBasic(grid, row, col + 1)
    for num in range(1, N + 1, 1):
      
        # Check if it is safe to place
        # the num (1-9)  in the
        # given row ,col  ->we
        # move to next column
        if isSafe(grid, row, col, num):
          
            # Assigning the num in
            # the current (row,col)
            # position of the grid
            # and assuming our assigned
            # num in the position
            # is correct
            grid[row][col] = num

            # Checking for next possibility with next
            # column
            if solveSudokuBasic(grid, row, col + 1):
                return True

        # Removing the assigned num ,
        # since our assumption
        # was wrong , and we go for
        # next assumption with
        # diff num value
        grid[row][col] = 0
    return False

def backtrackSolve(grid, domains, row, col):
    if row == N - 1 and col == N:
        return True
    if col == N:
        row += 1
        col = 0
    if grid[row][col] > 0:
        return backtrackSolve(grid, domains, row, col + 1)
    for num in domains[row][col]:
        if isSafe(grid, row, col, num):
            grid[row][col] = num
            if backtrackSolve(grid, domains, row, col + 1):
                return True
            grid[row][col] = 0
    return False

def readGridFromFile(filename):
    with open(filename, 'r') as file:
        return file.read()
    
def parseGrid(gridString):
    grid = []
    rows = gridString.strip().split('\n')
    for row in rows:
        grid.append([0 if char == '*' else int(char) for char in row])
    return grid    
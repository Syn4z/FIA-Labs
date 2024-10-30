import random
import math
import argparse
from utils import getNextGridFilename

class Sudoku:
    def __init__(self, N, K):
        self.N = N
        self.K = K

        SRNd = math.sqrt(N)
        self.SRN = int(SRNd)
        self.mat = [[0 for _ in range(N)] for _ in range(N)]
    
    def fillValues(self):
        self.fillDiagonal()
        self.fillRemaining(0, self.SRN)
        # Remove Randomly K digits to make game
        self.removeKDigits()
    
    def fillDiagonal(self):
        for i in range(0, self.N, self.SRN):
            self.fillBox(i, i)
    
    def unUsedInBox(self, rowStart, colStart, num):
        for i in range(self.SRN):
            for j in range(self.SRN):
                if self.mat[rowStart + i][colStart + j] == num:
                    return False
        return True
    
    def fillBox(self, row, col):
        num = 0
        for i in range(self.SRN):
            for j in range(self.SRN):
                while True:
                    num = self.randomGenerator(self.N)
                    if self.unUsedInBox(row, col, num):
                        break
                self.mat[row + i][col + j] = num
    
    def randomGenerator(self, num):
        return math.floor(random.random() * num + 1)
    
    def checkIfSafe(self, i, j, num):
        return (self.unUsedInRow(i, num) and self.unUsedInCol(j, num) and self.unUsedInBox(i - i % self.SRN, j - j % self.SRN, num))
    
    def unUsedInRow(self, i, num):
        for j in range(self.N):
            if self.mat[i][j] == num:
                return False
        return True
    
    def unUsedInCol(self, j, num):
        for i in range(self.N):
            if self.mat[i][j] == num:
                return False
        return True
    
    def fillRemaining(self, i, j):
        # Check if it reached the end of the matrix
        if i == self.N - 1 and j == self.N:
            return True
        # Move to the next row if it reached the end of the current row
        if j == self.N:
            i += 1
            j = 0
        # Skip cells that are already filled
        if self.mat[i][j] != 0:
            return self.fillRemaining(i, j + 1)
    
        # Try filling the current cell with a valid value
        for num in range(1, self.N + 1):
            if self.checkIfSafe(i, j, num):
                self.mat[i][j] = num
                if self.fillRemaining(i, j + 1):
                    return True
                self.mat[i][j] = 0
        # No valid value was found, so backtrack
        return False

    def removeKDigits(self):
        count = self.K

        while (count != 0):
            i = self.randomGenerator(self.N) - 1
            j = self.randomGenerator(self.N) - 1
            if (self.mat[i][j] != 0):
                count -= 1
                self.mat[i][j] = 0
        return

    def saveToFile(self, filename):
        with open(filename, 'w') as file:
            for i in range(self.N):
                for j in range(self.N):
                    if self.mat[i][j] == 0:
                        file.write('*')
                    else:
                        file.write(str(self.mat[i][j]))
                file.write('\n')        

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate a Sudoku grid.")
    parser.add_argument('-k', type=int, default=50, help='Number of cells to remove (default: 50)')
    args = parser.parse_args()
    N = 9
    K = args.k
    sudoku = Sudoku(N, K)
    sudoku.fillValues()
    filename = getNextGridFilename()
    sudoku.saveToFile(filename)
    print(f"Grid saved to '{filename}'")             
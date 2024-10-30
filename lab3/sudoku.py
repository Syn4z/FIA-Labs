from utils import *


def sudokuBacktracking(grid, row=0, col=0, steps=[0]):
    domains = initializeDomains(grid)
    return backtrack(grid, domains, row, col, steps)

def sudokuBacktrackingDomains(grid, row=0, col=0, steps=[0]):
    domains = initializeDomains(grid)
    domains = constraintPropagation(domains)
    return backtrack(grid, domains, row, col, steps)  

def sudokuForwardChecking(grid, row=0, col=0, steps=[0]):
    domains = initializeDomains(grid)
    domains = constraintPropagation(domains)
    return forwardCheckSolver(grid, domains, row, col, steps) 

def sudokuHeuristic(grid, row=0, col=0, steps=[0]):
    domains = initializeDomains(grid)
    domains = propagateConstraintsHeuristics(domains)
    return forwardCheckSolver(grid, domains, row, col, steps) 

def sudokuAc3(grid, row=0, col=0, steps=[0]):
    domains = initializeDomains(grid)
    domains = constraintPropagationAc3(domains)
    return forwardCheckSolver(grid, domains, row, col, steps)
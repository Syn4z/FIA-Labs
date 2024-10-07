# multiAgents.py
# --------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


from util import manhattanDistance
from game import Directions
import random, util, sys
from util import PriorityQueue
import time

from game import Agent


class ReflexAgent(Agent):
    """
      A reflex agent chooses an action at each choice point by examining
      its alternatives via a state evaluation function.

      The code below is provided as a guide.  You are welcome to change
      it in any way you see fit, so long as you don't touch our method
      headers.
    """

    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {North, South, West, East, Stop}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()      # Pacman position after moving
        newFood = successorGameState.getFood()               # Remaining food
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        "*** YOUR CODE HERE ***"
        listFood = newFood.asList()                        # All remaining food as list
        ghostPos = successorGameState.getGhostPositions()  # Get the ghost position
        # Initialize with list 
        mFoodDist = []
        mGhostDist = []

        # Find the distance of all the foods to the pacman 
        for food in listFood:
          mFoodDist.append(manhattanDistance(food, newPos))

        # Find the distance of all the ghost to the pacman
        for ghost in ghostPos:
          mGhostDist.append(manhattanDistance(ghost, newPos))

        if currentGameState.getPacmanPosition() == newPos:
          return (-(float("inf")))

        for ghostDistance in mGhostDist:
          if ghostDistance < 2:
            return (-(float("inf")))

        if len(mFoodDist) == 0:
          return float("inf")
        else:
          minFoodDist = min(mFoodDist)
          maxFoodDist = max(mFoodDist)

        return 1000/sum(mFoodDist) + 10000/len(mFoodDist)


def scoreEvaluationFunction(currentGameState):
    """
      This default evaluation function just returns the score of the state.
      The score is the same one displayed in the Pacman GUI.

      This evaluation function is meant for use with adversarial search agents
      (not reflex agents).
    """

    pacmanPos = currentGameState.getPacmanPosition()
    ghostList = currentGameState.getGhostStates() 
    foods = currentGameState.getFood()
    capsules = currentGameState.getCapsules()
    # Return based on game state
    if currentGameState.isWin():
        return float("inf")
    if currentGameState.isLose():
        return float("-inf")
    # Populate foodDistList and find minFoodDist
    foodDistList = []
    for each in foods.asList():
        foodDistList = foodDistList + [util.manhattanDistance(each, pacmanPos)]
    minFoodDist = min(foodDistList)
    # Populate ghostDistList and scaredGhostDistList, find minGhostDist and minScaredGhostDist
    ghostDistList = []
    scaredGhostDistList = []
    for each in ghostList:
        if each.scaredTimer == 0:
            ghostDistList = ghostDistList + [util.manhattanDistance(pacmanPos, each.getPosition())]
        elif each.scaredTimer > 0:
            scaredGhostDistList = scaredGhostDistList + [util.manhattanDistance(pacmanPos, each.getPosition())]
    minGhostDist = -1
    if len(ghostDistList) > 0:
        minGhostDist = min(ghostDistList)
    minScaredGhostDist = -1
    if len(scaredGhostDistList) > 0:
        minScaredGhostDist = min(scaredGhostDistList)
    score = minFoodDist - minGhostDist

    return score

def improvedEvaluationFunction(currentGameState):
    """
      Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
      evaluation function (question 5).

      DESCRIPTION: This evaluation function considers the following factors:
      - Distance to the nearest food
      - Distance to the nearest ghost (penalizes being close to active ghosts)
      - Distance to the nearest scared ghost (prioritizes eating scared ghosts)
      - Number of remaining food pellets
    """
    pacmanPos = currentGameState.getPacmanPosition()
    ghostList = currentGameState.getGhostStates()
    foods = currentGameState.getFood()
    capsules = currentGameState.getCapsules()

    if currentGameState.isWin():
        return float("inf")
    if currentGameState.isLose():
        return float("-inf")

    # Distance to the nearest food
    foodDistList = [util.manhattanDistance(pacmanPos, food) for food in foods.asList()]
    minFoodDist = min(foodDistList) if foodDistList else 0

    # Distance to the nearest ghost and scared ghost
    ghostDistList = []
    scaredGhostDistList = []
    for ghost in ghostList:
        if ghost.scaredTimer == 0:
            ghostDistList.append(util.manhattanDistance(pacmanPos, ghost.getPosition()))
        else:
            scaredGhostDistList.append(util.manhattanDistance(pacmanPos, ghost.getPosition()))

    minGhostDist = min(ghostDistList) if ghostDistList else float("inf")
    minScaredGhostDist = min(scaredGhostDistList) if scaredGhostDistList else 0

    # Number of remaining food pellets
    remainingFood = len(foods.asList())

    # Calculate the score
    score = currentGameState.getScore()
    score += -1.5 * minFoodDist
    score += -2 * (1.0 / minGhostDist) if minGhostDist != float("inf") else 0
    score += 2 * (1.0 / minScaredGhostDist) if minScaredGhostDist != 0 else 0
    score += -2.5 * remainingFood

    return score

class MultiAgentSearchAgent(Agent):
    """
      This class provides some common elements to all of your
      multi-agent searchers.  Any methods defined here will be available
      to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

      You *do not* need to make any changes here, but you can if you want to
      add functionality to all your adversarial search agents.  Please do not
      remove anything, however.

      Note: this is an abstract class: one that should not be instantiated.  It's
      only partially specified, and designed to be extended.  Agent (game.py)
      is another abstract class.
    """

    def __init__(self, evalFn='improvedEvaluationFunction', depth='2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)


class MinimaxAgent(MultiAgentSearchAgent):
    def getAction(self, gameState):
        def minimax(agentIndex, depth, gameState):
            if depth == self.depth or gameState.isWin() or gameState.isLose():
                return self.evaluationFunction(gameState)

            # Get legal actions and sort them based on heuristic evaluation
            legalActions = gameState.getLegalActions(agentIndex)
            sortedActions = sorted(legalActions, key=lambda action: self.evaluationFunction(gameState.generateSuccessor(agentIndex, action)), reverse=(agentIndex == 0))

            if agentIndex == 0:  # Pacman's turn (maximizing player)
                bestScore = float('-inf')
                for action in sortedActions:
                    successorState = gameState.generateSuccessor(agentIndex, action)
                    score = minimax(1, depth, successorState)
                    bestScore = max(bestScore, score)
                return bestScore
            else:  # Ghosts' turn (minimizing player)
                bestScore = float('inf')
                nextAgent = (agentIndex + 1) % gameState.getNumAgents()
                nextDepth = depth + 1 if nextAgent == 0 else depth
                for action in sortedActions:
                    successorState = gameState.generateSuccessor(agentIndex, action)
                    score = minimax(nextAgent, nextDepth, successorState)
                    bestScore = min(bestScore, score)
                return bestScore

        # Get the best action for Pacman
        legalMoves = gameState.getLegalActions(0)
        scores = [minimax(1, 0, gameState.generateSuccessor(0, action)) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices)

        return legalMoves[chosenIndex]


class AlphaBetaAgent(MultiAgentSearchAgent):
    """
      Here is the place to define your Alpha-Beta Pruning Algorithm
    """

    def getAction(self, gameState):
        """
          Returns the minimax action using self.depth and self.evaluationFunction
        """
        def alphaBeta(agentIndex, depth, gameState, alpha, beta):
            if depth == self.depth or gameState.isWin() or gameState.isLose():
                return self.evaluationFunction(gameState)
            
            if agentIndex == 0:  # Pacman (maximizing player)
                value = float('-inf')
                for action in gameState.getLegalActions(agentIndex):
                    successorState = gameState.generateSuccessor(agentIndex, action)
                    value = max(value, alphaBeta(1, depth, successorState, alpha, beta))
                    if value > beta:
                        return value
                    alpha = max(alpha, value)
                return value
            else:  # Ghosts (minimizing player)
                value = float('inf')
                nextAgent = (agentIndex + 1) % gameState.getNumAgents()
                nextDepth = depth + 1 if nextAgent == 0 else depth
                for action in gameState.getLegalActions(agentIndex):
                    successorState = gameState.generateSuccessor(agentIndex, action)
                    value = min(value, alphaBeta(nextAgent, nextDepth, successorState, alpha, beta))
                    if value < alpha:
                        return value
                    beta = min(beta, value)
                return value

        # Get the best action for Pacman
        legalMoves = gameState.getLegalActions(0)
        alpha = float('-inf')
        beta = float('inf')
        bestAction = None
        bestScore = float('-inf')
        for action in legalMoves:
            successorState = gameState.generateSuccessor(0, action)
            value = alphaBeta(1, 0, successorState, alpha, beta)
            if value > bestScore:
                bestScore = value
                bestAction = action
            alpha = max(alpha, value)

        return bestAction


class AStarMinimaxAgent(MinimaxAgent):
    """
      An agent that uses A* search combined with Minimax algorithm to find the best action.
    """

    def getAction(self, gameState):
        """
          Returns the best action using A* search algorithm combined with Minimax algorithm.
        """
        def heuristic(state, food, capsules, ghostStates):
            pacmanPos = state.getPacmanPosition()
            foodDist = util.manhattanDistance(pacmanPos, food)
            capsuleDist = min([util.manhattanDistance(pacmanPos, cap) for cap in capsules], default=0)
            ghostDist = min([util.manhattanDistance(pacmanPos, ghost.getPosition()) for ghost in ghostStates if ghost.scaredTimer == 0], default=float('inf'))
            scaredGhostDist = min([util.manhattanDistance(pacmanPos, ghost.getPosition()) for ghost in ghostStates if ghost.scaredTimer > 0], default=0)

            if ghostDist < 2:
                ghostPenalty = float('inf')
            else:
                ghostPenalty = 1.0 / (ghostDist + 1)

            return foodDist + capsuleDist + ghostPenalty - scaredGhostDist

        def aStarSearch(gameState):
            startState = gameState
            foodList = startState.getFood().asList()
            capsules = startState.getCapsules()
            ghostStates = startState.getGhostStates()
            if not foodList:
                return []

            pq = PriorityQueue()
            pq.push((startState, []), 0)
            visited = set()
            startTime = time.time()

            while not pq.isEmpty():
                if time.time() - startTime > 1:  # Timeout after 1 second to avoid infinite loop
                    break
                currentState, actions = pq.pop()
                if currentState.getPacmanPosition() in foodList:
                    return actions
                if currentState not in visited:
                    visited.add(currentState)
                    for action in currentState.getLegalActions(0):
                        successor = currentState.generatePacmanSuccessor(action)
                        if successor is not None:
                            newActions = actions + [action]
                            cost = len(newActions) + heuristic(successor, foodList[0], capsules, ghostStates)
                            pq.push((successor, newActions), cost)
            return []

        # Perform A* search to find the best path to the food
        actions = aStarSearch(gameState)
        if actions:
            return actions[0]

        # If no path found by A*, fallback to Minimax
        return super().getAction(gameState)


class AStarAlphaBetaAgent(AlphaBetaAgent):
    """
      An agent that uses A* search combined with Alpha-Beta Pruning algorithm to find the best action.
    """
    def getAction(self, gameState):
        """
          Returns the best action using A* search algorithm combined with Alpha-Beta Pruning algorithm.
        """
        def heuristic(state, food, capsules, ghostStates):
            pacmanPos = state.getPacmanPosition()
            foodDist = util.manhattanDistance(pacmanPos, food)
            capsuleDist = min([util.manhattanDistance(pacmanPos, cap) for cap in capsules], default=0)
            ghostDist = min([util.manhattanDistance(pacmanPos, ghost.getPosition()) for ghost in ghostStates if ghost.scaredTimer == 0], default=float('inf'))
            scaredGhostDist = min([util.manhattanDistance(pacmanPos, ghost.getPosition()) for ghost in ghostStates if ghost.scaredTimer > 0], default=0)

            ghostPenalty = 0
            if ghostDist < 2:
                ghostPenalty = 10 / (ghostDist + 1)

            return foodDist + capsuleDist + ghostPenalty - scaredGhostDist

        def aStarSearch(gameState):
            startState = gameState
            foodList = startState.getFood().asList()
            capsules = startState.getCapsules()
            ghostStates = startState.getGhostStates()
            if not foodList:
                return []

            pq = PriorityQueue()
            pq.push((startState, []), 0)
            visited = set()
            startTime = time.time()

            while not pq.isEmpty():
                if time.time() - startTime > 1:  # Timeout after 1 second to avoid infinite loop
                    break
                currentState, actions = pq.pop()
                if currentState.getPacmanPosition() in foodList:
                    return actions
                if currentState not in visited:
                    visited.add(currentState)
                    for action in currentState.getLegalActions(0):
                        successor = currentState.generatePacmanSuccessor(action)
                        if successor is not None:
                            newActions = actions + [action]
                            cost = len(newActions) + heuristic(successor, foodList[0], capsules, ghostStates)
                            pq.push((successor, newActions), cost)
            return []

        # Perform A* search to find the best path to the food
        actions = aStarSearch(gameState)
        if actions:
            return actions[0]

        # If no path found by A*, fallback to Alpha-Beta Pruning
        return super().getAction(gameState)


def betterEvaluationFunction(currentGameState):
    """
      Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
      evaluation function (question 5).

      DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    pacmanPos = currentGameState.getPacmanPosition()
    ghostList = currentGameState.getGhostStates() 
    foods = currentGameState.getFood()
    capsules = currentGameState.getCapsules()
    # Return based on game state
    if currentGameState.isWin():
        return float("inf")
    if currentGameState.isLose():
        return float("-inf")
    # Populate foodDistList and find minFoodDist
    foodDistList = []
    for each in foods.asList():
        foodDistList = foodDistList + [util.manhattanDistance(each, pacmanPos)]
    minFoodDist = min(foodDistList)
    # Populate ghostDistList and scaredGhostDistList, find minGhostDist and minScaredGhostDist
    ghostDistList = []
    scaredGhostDistList = []
    for each in ghostList:
        if each.scaredTimer == 0:
            ghostDistList = ghostDistList + [util.manhattanDistance(pacmanPos, each.getPosition())]
        elif each.scaredTimer > 0:
            scaredGhostDistList = scaredGhostDistList + [util.manhattanDistance(pacmanPos, each.getPosition())]
    minGhostDist = -1
    if len(ghostDistList) > 0:
        minGhostDist = min(ghostDistList)
    minScaredGhostDist = -1
    if len(scaredGhostDistList) > 0:
        minScaredGhostDist = min(scaredGhostDistList)
    # Evaluate score
    score = scoreEvaluationFunction(currentGameState)
    """
        Your improved evaluation here
    """
    return score


# Abbreviation
better = betterEvaluationFunction


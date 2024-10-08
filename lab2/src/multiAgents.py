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
    This default evaluation function returns the score of the current game state.
    The score is the same as the one displayed in the Pacman GUI.

    The function evaluates the state's desirability for Pacman based on:
    - Distance to the nearest food pellet.
    - Distance to the nearest ghost (only considered if the ghost is not scared).
    
    Special Cases:
    - If Pacman wins, the score is set to positive infinity.
    - If Pacman loses, the score is set to negative infinity.

    Args:
        currentGameState (GameState): The current state of the game, which contains information 
        about Pacman's position, ghosts, food locations, and win/lose status.

    Returns:
        float: The evaluated score of the current state.
    """

    pacmanPos = currentGameState.getPacmanPosition()
    ghostList = currentGameState.getGhostStates() 
    foods = currentGameState.getFood()

    # If Pacman has won, return positive infinity
    if currentGameState.isWin():
        return float("inf")
    
    # If Pacman has lost, return negative infinity
    if currentGameState.isLose():
        return float("-inf")

    # Calculate the distance to each food pellet
    foodDistList = []
    for each in foods.asList():
        foodDistList.append(util.manhattanDistance(each, pacmanPos))

    # Get the minimum distance to the closest food pellet
    minFoodDist = min(foodDistList)

    # Calculate the distance to each ghost
    ghostDistList = []
    for each in ghostList:
        if each.scaredTimer == 0:  # Only consider active (non-scared) ghosts
            ghostDistList.append(util.manhattanDistance(pacmanPos, each.getPosition()))
    
    # If there are active ghosts, get the minimum distance to the closest ghost
    minGhostDist = -1
    if ghostDistList:
        minGhostDist = min(ghostDistList)

    # The score is based on minimizing the distance to food and avoiding ghosts
    score = minFoodDist - minGhostDist
    return score

def improvedEvaluationFunction(currentGameState):
    """
    This is an advanced evaluation function for the Pacman game. It improves Pacman's decision-making
    by considering various values beyond just the basic game score. The function calculates a score 
    based on the following aspects of the game state:

    Values:
    - Distance to the nearest food: Pacman should prioritize moving toward the closest food pellet.
    - Distance to the nearest ghost: Pacman should avoid getting too close to active ghosts.
    - Distance to the nearest scared ghost: If a ghost is scared, Pacman should prioritize eating it.
    - Number of remaining food pellets: Fewer remaining pellets increase the urgency of eating them.

    Args:
        currentGameState: The current state of the game, which includes the position of Pacman,
                          ghosts, food, score, and the game status (win/lose).

    Returns:
        score (float): A calculated evaluation score for the current game state, with higher
                       scores representing better positions for Pacman.
    """
    pacmanPos = currentGameState.getPacmanPosition()  
    ghostList = currentGameState.getGhostStates()     
    foods = currentGameState.getFood()                

    # If the game is won or lost, return a very high or low score accordingly
    if currentGameState.isWin():
        return float("inf")
    if currentGameState.isLose():
        return float("-inf")

    # Calculate the distance to the nearest food pellet
    foodDistList = [util.manhattanDistance(pacmanPos, food) for food in foods.asList()]
    minFoodDist = min(foodDistList) if foodDistList else 0  

    # Separate active ghosts from scared ghosts based on their scared timer
    ghostDistList = []  
    scaredGhostDistList = [] 
    for ghost in ghostList:
        if ghost.scaredTimer == 0:  # Active ghost
            ghostDistList.append(util.manhattanDistance(pacmanPos, ghost.getPosition()))
        else:  # Scared ghost
            scaredGhostDistList.append(util.manhattanDistance(pacmanPos, ghost.getPosition()))

    minGhostDist = min(ghostDistList) if ghostDistList else float("inf")
    minScaredGhostDist = min(scaredGhostDistList) if scaredGhostDistList else 0  
    remainingFood = len(foods.asList())

    # Calculate the evaluation score based on the various factors
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
        """
        Returns the best action for Pacman using the Minimax algorithm with a fixed search depth.

        The agent evaluates all possible future states up to a certain depth, alternating between
        Pacman's (maximizing) and ghosts' (minimizing) turns. The action that maximizes Pacman's 
        chances of winning or achieving the highest score is selected.

        Args:
            gameState (GameState): The current state of the game.

        Returns:
            str: The best action Pacman can take (e.g., 'North', 'South').
        """
        
        def minimax(agentIndex, depth, gameState):
            """
            A helper function that recursively performs Minimax search.

            Args:
                agentIndex (int): The index of the agent (0 for Pacman, >= 1 for ghosts).
                depth (int): The current depth of the search tree.
                gameState (GameState): The current state of the game.

            Returns:
                float: The evaluated score of the game state.
            """
            # Base case: If max depth is reached, or if the game is over, return the evaluation
            if depth == self.depth or gameState.isWin() or gameState.isLose():
                return self.evaluationFunction(gameState)

            legalActions = gameState.getLegalActions(agentIndex)
            # Sort actions by heuristic evaluation to optimize search
            sortedActions = sorted(legalActions, 
                                   key=lambda action: self.evaluationFunction(gameState.generateSuccessor(agentIndex, action)), 
                                   reverse=(agentIndex == 0))  # Sort in descending order for Pacman

            if agentIndex == 0:  # Pacman's turn (maximizing player)
                bestScore = float('-inf')
                for action in sortedActions:
                    successorState = gameState.generateSuccessor(agentIndex, action)
                    score = minimax(1, depth, successorState)
                    bestScore = max(bestScore, score)
                return bestScore
            else:  # Ghosts' turn (minimizing player)
                bestScore = float('inf')
                nextAgent = (agentIndex + 1) % gameState.getNumAgents()  # Move to the next agent
                nextDepth = depth + 1 if nextAgent == 0 else depth  # Increase depth only after all agents move
                for action in sortedActions:
                    successorState = gameState.generateSuccessor(agentIndex, action)
                    score = minimax(nextAgent, nextDepth, successorState)
                    bestScore = min(bestScore, score)
                return bestScore
            
        legalMoves = gameState.getLegalActions(0)
        # Evaluate each move using the Minimax algorithm
        scores = [minimax(1, 0, gameState.generateSuccessor(0, action)) for action in legalMoves]

        # Choose the action with the best score
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices)

        return legalMoves[chosenIndex]


class AlphaBetaAgent(MultiAgentSearchAgent):

    def getAction(self, gameState):
        """
        Determines the best action for Pacman using the Alpha-Beta pruning algorithm.

        Args:
            gameState: The current state of the game.
        
        Returns:
            The action that maximizes Pacman's chance of winning while avoiding ghosts,
            evaluated up to the depth specified by self.depth.
        """

        def alphaBeta(agentIndex, depth, gameState, alpha, beta):
            """
            Performs the Alpha-Beta pruning algorithm recursively.

            Args:
                agentIndex: The index of the current agent (0 for Pacman, 1 and onwards for ghosts).
                depth: The current depth in the game tree.
                gameState: The current state of the game.
                alpha: The best value that the maximizing player (Pacman) can guarantee so far.
                beta: The best value that the minimizing player (ghosts) can guarantee so far.

            Returns:
                The score of the game state evaluated using the self.evaluationFunction.
            """
            
            # Terminal state check: stop if depth limit is reached or game is won/lost
            if depth == self.depth or gameState.isWin() or gameState.isLose():
                return self.evaluationFunction(gameState)
            
            if agentIndex == 0:  # Pacman's turn (maximizing player)
                value = float('-inf')
                # Iterate over all legal actions available to Pacman
                for action in gameState.getLegalActions(agentIndex):
                    successorState = gameState.generateSuccessor(agentIndex, action)
                    # Call alphaBeta for the next agent (ghosts)
                    value = max(value, alphaBeta(1, depth, successorState, alpha, beta))
                    # Pruning: If value exceeds beta, stop searching this branch
                    if value > beta:
                        return value
                    # Update alpha, the best value found for Pacman
                    alpha = max(alpha, value)
                return value
            else:  # Ghosts' turn (minimizing player)
                value = float('inf')
                # Determine the next agent and update depth if it's Pacman's turn
                nextAgent = (agentIndex + 1) % gameState.getNumAgents()
                nextDepth = depth + 1 if nextAgent == 0 else depth
                # Iterate over all legal actions available to the ghosts
                for action in gameState.getLegalActions(agentIndex):
                    successorState = gameState.generateSuccessor(agentIndex, action)
                    # Call alphaBeta for the next agent (Pacman or another ghost)
                    value = min(value, alphaBeta(nextAgent, nextDepth, successorState, alpha, beta))
                    # Pruning: If value is less than alpha, stop searching this branch
                    if value < alpha:
                        return value
                    # Update beta, the best value found for the ghosts
                    beta = min(beta, value)
                return value

        legalMoves = gameState.getLegalActions(0)
        alpha = float('-inf')  
        beta = float('inf')    
        bestAction = None     
        bestScore = float('-inf')  

        # Evaluate each legal move for Pacman
        for action in legalMoves:
            successorState = gameState.generateSuccessor(0, action)
            value = alphaBeta(1, 0, successorState, alpha, beta)
            # Update the best action if this value is higher than the current best score
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


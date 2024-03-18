# search.py
# ---------
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


"""
In search.py, you will implement generic search algorithms which are called by
Pacman agents (in searchAgents.py).
"""

import itertools
from itertools import product
import util
from game import Actions
from game import Directions
from game import Grid
from util import foodGridtoDic


class SearchProblem:
    """
    This class outlines the structure of a search problem, but doesn't implement
    any of the methods (in object-oriented terminology: an abstract class).

    You do not need to change anything in this class, ever.
    """

    def getStartState(self):
        """
        Returns the start state for the search problem.
        """
        util.raiseNotDefined()

    def isGoalState(self, state):
        """
          state: Search state

        Returns True if and only if the state is a valid goal state.
        """
        util.raiseNotDefined()

    def getSuccessors(self, state):
        """
          state: Search state

        For a given state, this should return a list of triples, (successor,
        action, stepCost), where 'successor' is a successor to the current
        state, 'action' is the action required to get there, and 'stepCost' is
        the incremental cost of expanding to that successor.
        """
        util.raiseNotDefined()

    def getCostOfActions(self, actions):
        """
         actions: A list of actions to take

        This method returns the total cost of a particular sequence of actions.
        The sequence must be composed of legal moves.
        """
        util.raiseNotDefined()


def depthFirstSearch(problem):
    """
    Search the deepest nodes in the search tree first.

    Your search algorithm needs to return a list of actions that reaches the
    goal. Make sure to implement a graph search algorithm.

    To get started, you might want to try some of these simple commands to
    understand the search problem that is being passed in:

    print("Start:", problem.getStartState())
    print("Is the start a goal?", problem.isGoalState(problem.getStartState()))
    print("Start's successors:", problem.getSuccessors(problem.getStartState()))
    """
    "*** YOUR CODE HERE ***"

    util.raiseNotDefined()


def uniformCostSearch(problem):
    """Search the node of least total cost first."""
    "*** YOUR CODE HERE ***"
    util.raiseNotDefined()


def nullHeuristic(state, problem=None):
    """
    This heuristic is trivial.
    """
    return 0


def foodHeuristic(state, problem):
    """
    Your heuristic for the FoodSearchProblem goes here.

    This heuristic must be consistent to ensure correctness.  First, try to come
    up with an admissible heuristic; almost all admissible heuristics will be
    consistent as well.

    If using A* ever finds a solution that is worse uniform cost search finds,
    your heuristic is *not* consistent, and probably not admissible!  On the
    other hand, inadmissible or inconsistent heuristics may find optimal
    solutions, so be careful.

    The state is a tuple ( pacmanPosition, foodGrid ) where foodGrid is a Grid
    (see game.py) of either True or False. You can call foodGrid.asList() to get
    a list of food coordinates instead.

    If you want access to info like walls, capsules, etc., you can query the
    problem.  For example, problem.walls gives you a Grid of where the walls
    are.

    If you want to *store* information to be reused in other calls to the
    heuristic, there is a dictionary called problem.heuristicInfo that you can
    use. For example, if you only want to count the walls once and store that
    value, try: problem.heuristicInfo['wallCount'] = problem.walls.count()
    Subsequent calls to this heuristic can access
    problem.heuristicInfo['wallCount']
    """
    "*** YOUR CODE HERE for task1 ***"
    position, foodGrid = state
    distance = []
    distances_food =[0]
    for food in foodGrid.asList(): #人和食物的距离
        distance.append(getMazeDistance(position,food, problem))
        for tofood in foodGrid.asList(): #每个食物之间的距离
            distances_food.append(getMazeDistance(food,tofood, problem))
    return min(distance) + max(distances_food) if len(distance) else max(distances_food)

    # comment the below line after you implement the algorithm
    util.raiseNotDefined()


def createFoodGrid(pos, width, height):
    foodGrid = Grid(width, height, initialValue=False) #greate new grid
    x, y = pos #tuple (x, y)
    foodGrid[x][y]= True

    return foodGrid

def getMazeDistance(start, end, problem): 
    try:
        return problem.heuristicInfo[(start, end)] #test is the current distance between start and enc have been calculated, if so just return
    except:
        foodGrid = createFoodGrid(end, problem.walls.width, problem.walls.height)
        prob = SingleFoodSearchProblem(pos=start, food=foodGrid, walls=problem.walls) #create a new probl after new create
        problem.heuristicInfo[(start,end)]=len(astar(prob)) #it is a dict

        return problem.heuristicInfo[(start,end)] #return solution path


class MAPFProblem(SearchProblem):
    """
    A search problem associated with finding a path that collects all
    food (dots) in a Pacman game.

    A search state in this problem is a tuple ( pacmanPositions, foodGrid ) where
      pacmanPositions:  a dictionary {pacman_name: (x,y)} specifying Pacmans' positions
      foodGrid:         a Grid (see game.py) of either pacman_name or False, specifying the target food of that pacman_name. For example, foodGrid[x][y] == 'A' means pacman A's target food is at (x, y). Each pacman have exactly one target food at start
    """

    def __init__(self, startingGameState):
        "Initial function"
        "*** WARNING: DO NOT CHANGE!!! ***"
        self.start = (startingGameState.getPacmanPosition(), startingGameState.getFood())
        self.walls = startingGameState.getWalls()

    def getStartState(self):
        "Get start state"
        "*** WARNING: DO NOT CHANGE!!! ***"
        return self.start

    def isGoalState(self, state):
        "Return if the state is the goal state"
        "*** YOUR CODE HERE for task2 ***"
        pacmanPositions, foodGrid = state
        count = 0
        for name in list(pacmanPositions.keys()):
            count += foodGrid.count(item=name)
        if count == 0:
            return True 
        return False
        #return all(foodGrid[x][y] == False for x in range(foodGrid.width) for y in range(foodGrid.height))


        # comment the below line after you implement the function
        util.raiseNotDefined()

    def getSuccessors(self, state):
        """
            Returns successor states, the actions they require, and a cost of 1.
            Input: search_state
            Output: a list of tuples (next_search_state, action_dict, 1)

            A search_state in this problem is a tuple consists of two dictionaries ( pacmanPositions, foodGrid ) where
              pacmanPositions:  a dictionary {pacman_name: (x,y)} specifying Pacmans' positions
              foodGrid:    a Grid (see game.py) of either pacman_name or False, specifying the target food of each pacman.

            An action_dict is {pacman_name: direction} specifying each pacman's move direction, where direction could be one of 5 possible directions in Directions (i.e. Direction.SOUTH, Direction.STOP etc)


        """
        "*** YOUR CODE HERE for task2 ***"
        pacmanPositions, foodGrid = state
        successors = []
        #self._expanded += 1  # DO NOT CHANGE

        # Generate all possible moves for each pacman
        all_moves = {}
        for pacman_name, (x, y) in pacmanPositions.items():
            moves = []
            for direction in [Directions.NORTH, Directions.SOUTH, Directions.EAST, Directions.WEST, Directions.STOP]:
                dx, dy = Actions.directionToVector(direction)
                next_x, next_y = int(x + dx), int(y + dy)
                if not self.walls[next_x][next_y]:
                    moves.append(direction)
            all_moves[pacman_name] = moves

        # Generate all combinations of moves for all pacmans
        for move_combination in itertools.product(*all_moves.values()):
            nextPositions = pacmanPositions.copy()
            nextFoodGrid = foodGrid.copy()
            move_dict = {}
            idx = 0
            for pacman_name in pacmanPositions.keys():
                direction = move_combination[idx]
                x, y = nextPositions[pacman_name]
                dx, dy = Actions.directionToVector(direction)
                next_x, next_y = int(x + dx), int(y + dy)
                nextPositions[pacman_name] = (next_x, next_y)
                if nextFoodGrid[next_x][next_y] == pacman_name:
                    nextFoodGrid[next_x][next_y] = False  # Consume the target food
                move_dict[pacman_name] = direction
                idx += 1

            successors.append(((nextPositions, nextFoodGrid), move_dict, 1))

        return successors

        # comment the below line after you implement the function
        util.raiseNotDefined()


class CBSNode:
    def __init__(self, constraints, path, solution, cost):  # Use double underscores for __init__
        self.constraints = constraints
        self.path = path
        self.solution = solution
        self.cost = cost

    def __lt__(self, other):
        return self.cost < other.cost


def conflictBasedSearch(problem: MAPFProblem):
    """
        Conflict-based search algorithm.
        Input: MAPFProblem
        Output(IMPORTANT!!!): A dictionary stores the path for each pacman as a list {pacman_name: [action1, action2, ...]}.

    """
    "*** YOUR CODE HERE for task3 ***"
    root = CBSNode(constraints={}, path={}, solution={}, cost=0)
    for agent in problem.start[0].keys():  # Iterate over the keys of the dictionary
        
        path, solution = aStarAdaptSearch(problem, agent, root.constraints, problem.getStartState())
        root.path[agent] = path
        root.solution[agent] = solution
    root.cost = sum([len(path) for path in root.solution.values()])

    OPEN = util.PriorityQueue()
    OPEN.push(root, root.cost)

    while not OPEN.isEmpty():
        node = OPEN.pop()
        conflict = findConflict(node.path)
        if not conflict:
            return node.solution
        
        for agent in conflict['agents']:
            new_constraints = node.constraints.copy()
            new_constraints[(agent, conflict['x'], conflict['y'], conflict['t'])] = True
            new_node = CBSNode(constraints=new_constraints, path=node.path.copy(), solution=node.solution.copy(), cost=0)
            path, solution = aStarAdaptSearch(problem, agent, new_constraints, problem.getStartState())
            new_node.path[agent] = path
            new_node.solution[agent] = solution
            new_node.cost = sum([len(path) for path in new_node.solution.values()])
            OPEN.push(new_node, new_node.cost)

    return None

    util.raiseNotDefined()

def findConflict(paths):
    conflict ={}
    foundConflict = False
    max_length = max(len(path)for path in paths.values())
    for index in range(max_length):
        if foundConflict:
            return conflict
        positions_at_index = {agent: paths[agent][index] for agent in paths if index < len(paths[agent])}
        seen_positions ={}
        for agent,position in positions_at_index.items():
            if position in seen_positions:
                foundConflict = True
                conflict['x']= position[0]
                conflict['y']= position[1]
                conflict['t']= index
                seen_positions[position] += agent
                conflict['agents']= seen_positions[position]# 只记录参与了 conflict 的 agent
            else:
                seen_positions[position] = agent
    return conflict

def manhattanHeuristic(position, goal):
    xy1 = position
    xy2 = goal
    return abs(xy1[0] - xy2[0]) + abs(xy1[1] - xy2[1])

def getGoalState(problem, agent): 
    for x in range(problem.getFoodGrid().width):
        for y in range(problem.getFoodGrid().height):
            if problem.getFoodGrid():
                return [x][y] == agent # Assuming the foodGrid is marked with agent namesreturn(x，y)
    return None

def aStarAdaptSearch(problem, agent, constraints, state):
    myPQ = util.PriorityQueue()
    agentPositions, foodGrid = state
    startState = problem.getStartState()[0][agent]
    goalState =getGoalState(problem, agent)
    startNode =(startState,0,[startState],[],0)#(state, cost, path, actions, time)
    myPQ.push(startNode,0)# Priority is heuristic cost from start to goal
    visited = set()
    while not myPQ.isEmpty():
        currentNode = myPQ.pop()
        currentState, currentCost, currentPath, currentSolution, currentTime = currentNode
        if currentState == goalState:
            return(currentPath, currentSolution)
        if(currentState,currentTime)in visited:
            continue
        visited.add((currentState, currentTime))
        for nextState, action, stepCost in problem.getSuccessors(currentState, agent):
            if((agent, nextState[0],nextState[1],currentTime+1) not in constraints.keys()):# Skip the successor violates constraints
                newCost =currentCost + stepCost
                newPath = currentPath + [nextState]
                newSolution = currentSolution +[action]
                newTime = currentTime +1
                heuristicCost = manhattanHeuristic(nextState, goalState)# abs(xy1[0]- xy2[0])+
                totalCost = newCost + heuristicCost
                newNode =(nextState,newCost,newPath,newSolution,newTime)
                myPQ.push(newNode, totalCost)
# If no path found
    return []




"###WARNING: Altering the following functions is STRICTLY PROHIBITED. Failure to comply may result in a grade of 0 for Assignment 1.###"
"###WARNING: Altering the following functions is STRICTLY PROHIBITED. Failure to comply may result in a grade of 0 for Assignment 1.###"
"###WARNING: Altering the following functions is STRICTLY PROHIBITED. Failure to comply may result in a grade of 0 for Assignment 1.###"


class FoodSearchProblem(SearchProblem):
    """
    A search problem associated with finding a path that collects all
    food (dots) in a Pacman game.

    A search state in this problem is a tuple ( pacmanPosition, foodGrid ) where
      pacmanPosition: a tuple (x,y) of integers specifying Pacman's position
      foodGrid:       a Grid (see game.py) of either True or False, specifying remaining food
    """

    def __init__(self, startingGameState):
        self.start = (startingGameState.getPacmanPosition(), startingGameState.getFood())
        self.walls = startingGameState.getWalls()
        self._expanded = 0  # DO NOT CHANGE
        self.heuristicInfo = {}  # A optional dictionary for the heuristic to store information

    def getStartState(self):
        return self.start

    def isGoalState(self, state):
        return state[1].count() == 0

    def getSuccessors(self, state):
        "Returns successor states, the actions they require, and a cost of 1."
        successors = []
        self._expanded += 1  # DO NOT CHANGE
        for direction in [Directions.NORTH, Directions.SOUTH, Directions.EAST, Directions.WEST, Directions.STOP]:
            x, y = state[0]
            dx, dy = Actions.directionToVector(direction)
            next_x, next_y = int(x + dx), int(y + dy)
            if not self.walls[next_x][next_y]:
                nextFood = state[1].copy()
                nextFood[next_x][next_y] = False
                successors.append((((next_x, next_y), nextFood), direction, 1))
        return successors

    def getCostOfActions(self, actions):
        """Returns the cost of a particular sequence of actions.  If those actions
        include an illegal move, return 999999"""
        x, y = self.getStartState()[0]
        cost = 0
        for action in actions:
            # figure out the next state and see whether it's legal
            dx, dy = Actions.directionToVector(action)
            x, y = int(x + dx), int(y + dy)
            if self.walls[x][y]:
                return 999999
            cost += 1
        return cost


class SingleFoodSearchProblem(FoodSearchProblem):
    """
    A special food search problem with only one food and can be generated by passing pacman position, food grid (only one True value in the grid) and wall grid
    """

    def __init__(self, pos, food, walls):
        self.start = (pos, food)
        self.walls = walls
        self._expanded = 0  # DO NOT CHANGE
        self.heuristicInfo = {}  # A optional dictionary for the heuristic to store information


def breadthFirstSearch(problem):
    """Search the shallowest nodes in the search tree first."""
    Q = util.Queue()
    startState = problem.getStartState()
    startNode = (startState, 0, [])
    Q.push(startNode)
    while not Q.isEmpty():
        node = Q.pop()
        state, cost, path = node
        if problem.isGoalState(state):
            return path
        for succ in problem.getSuccessors(state):
            succState, succAction, succCost = succ
            new_cost = cost + succCost
            newNode = (succState, new_cost, path + [succAction])
            Q.push(newNode)

    return None  # Goal not found


def aStarSearch(problem, heuristic=nullHeuristic):
    """Search the node that has the lowest combined cost and heuristic first."""
    myPQ = util.PriorityQueue()
    startState = problem.getStartState()
    startNode = (startState, 0, [])
    myPQ.push(startNode, heuristic(startState, problem))
    best_g = dict()
    while not myPQ.isEmpty():
        node = myPQ.pop()
        state, cost, path = node
        if (not state in best_g) or (cost < best_g[state]):
            best_g[state] = cost
            if problem.isGoalState(state):
                return path
            for succ in problem.getSuccessors(state):
                succState, succAction, succCost = succ
                new_cost = cost + succCost
                newNode = (succState, new_cost, path + [succAction])
                myPQ.push(newNode, heuristic(succState, problem) + new_cost)

    return None  # Goal not found


# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
cbs = conflictBasedSearch

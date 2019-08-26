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

import util
from util import *


class action():
    def __init__(self, current, previous, direction, traversed: bool, cost=0):
        self.current = current
        self.previous = previous
        self.direction = direction
        self.traversed = traversed
        self.cost = cost



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


def tinyMazeSearch(problem):
    """
    Returns a sequence of moves that solves tinyMaze.  For any other maze, the
    sequence of moves will be incorrect, so only use this for tinyMaze.
    """
    from game import Directions
    s = Directions.SOUTH
    w = Directions.WEST
    return [s, s, w, s, w, w, s, w]


def depthFirstSearch(problem):
    """
    Search the deepest nodes in the search tree first.
    """

    current_state = problem.getStartState()
    previous_state = None

    explored = []
    explored.append(current_state)

    nodes = []
    nodes.append(action(current=current_state, previous=None, direction=None, traversed=False))

    # Pacman_actions hold all the actions of the agent
    pacman_actions = []
    frontier = Stack()
    frontier.push((current_state, pacman_actions))

    while not frontier.isEmpty():

        current_state, state_actions = frontier.pop()

        explored.append(current_state)
        # Get all the nodes that their state is tagged as current
        currentNodes = [node for node in nodes if node.current == current_state]

        # Correct the non current states
        if len(currentNodes) > 1:
            for node in currentNodes:
                if node.previous == previous_state:
                    node.traversed = True
        else:
            currentNodes[0].traversed = True

        previous_state = current_state
        pacman_actions = state_actions

        if problem.isGoalState(current_state):
            return pacman_actions

        # Update the node and frontier states
        for successor in problem.getSuccessors(current_state):
            n_state, n_direction = (successor[0], successor[1])

            if n_state not in explored:
                nodes.append(action(current=n_state, previous=current_state, direction=n_direction, traversed=False))
                frontier.push((n_state, state_actions + [n_direction]))


def breadthFirstSearch(problem):
    """Search the shallowest nodes in the search tree first."""

    current_state = problem.getStartState()
    explored = []

    nodes = []
    nodes.append(action(current=current_state, previous=None, direction=None, traversed=False))

    pacman_actions = []
    frontier = Queue()
    frontier.push((current_state, pacman_actions))

    while not frontier.isEmpty():
        current_state, state_actions = frontier.pop()
        if current_state in explored:
            continue

        explored.append((current_state))

        # Find the current node
        current_node = next((item for item in nodes if item.current == current_state), None)
        current_node.traversed = True

        pacman_actions = state_actions
        if problem.isGoalState(current_state):
            return pacman_actions

        # Update the node and frontier states
        for successor in problem.getSuccessors(current_state):
            n_state, n_direction = (successor[0], successor[1])
            if n_state not in explored:
                nodes.append(action(current=n_state, previous=current_state, direction=n_direction, traversed=False))
                frontier.push((n_state, state_actions + [n_direction]))


def uniformCostSearch(problem):
    current_state = problem.getStartState()
    explored = []

    nodes = []
    nodes.append(action(current=current_state, previous=None, direction=None, traversed=False, cost=0))
    frontier = PriorityQueue()
    pacman_actions = []
    frontier.push((current_state, pacman_actions), 0)

    while not frontier.isEmpty():
        current_state, state_actions = frontier.pop()
        # Skip if already explored
        if current_state in explored:
            continue

        explored.append((current_state))
        # Update candidates for exploration
        potentialNodes = []
        for node in nodes:
            if node.current == current_state:
                potentialNodes.append(node)
        # Select the candidate with the lowest score
        if len(potentialNodes) > 1:
            smallNode = potentialNodes[0]
            for node in potentialNodes:
                if smallNode.cost > node.cost:
                    smallNode = node
            current_node = smallNode
        else:
            current_node = potentialNodes[0]

        current_node.traversed = True

        pacman_actions = state_actions
        if problem.isGoalState(current_state):
            return pacman_actions

        # Update the node and frontier states
        for successor in problem.getSuccessors(current_state):
            n_state, n_direction, n_cost = (successor[0], successor[1], successor[2])
            if n_state not in explored:
                costSoFar = n_cost + current_node.cost
                nodes.append(action(current=n_state, previous=current_state, direction=n_direction, traversed=False,
                                    cost=costSoFar))
                frontier.push((n_state, state_actions + [n_direction]), costSoFar)


def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """

    return 0


def aStarSearch(problem, heuristic=nullHeuristic):
    current_state = problem.getStartState()
    explored = []

    nodes = []
    nodes.append(action(current=current_state, previous=None, direction=None, traversed=False, cost=0))
    frontier = PriorityQueue()
    pacman_actions = []
    frontier.push((current_state, pacman_actions), 0)

    while not frontier.isEmpty():
        current_state, state_actions = frontier.pop()
        # Skip if already explored
        if current_state in explored:
            continue

        explored.append((current_state))
        # Update candidates for exploration
        potentialNodes = []
        for node in nodes:
            if node.current == current_state:
                potentialNodes.append(node)
        # Select the candidate with the lowest score
        if len(potentialNodes) > 1:
            smallNode = potentialNodes[0]
            for node in potentialNodes:
                if smallNode.cost > node.cost:
                    smallNode = node
            current_node = smallNode
        else:
            current_node = potentialNodes[0]

        current_node.traversed = True

        pacman_actions = state_actions
        if problem.isGoalState(current_state):
            return pacman_actions

        # Update the node and frontier states
        for successor in problem.getSuccessors(current_state):
            n_state, n_direction, n_cost = (successor[0], successor[1], successor[2])
            if n_state not in explored:
                costSoFar = n_cost + current_node.cost
                nodes.append(action(current=n_state, previous=current_state, direction=n_direction, traversed=False,
                                    cost=costSoFar))
                frontier.push((n_state, state_actions + [n_direction]), costSoFar + heuristic(n_state, problem))


# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch

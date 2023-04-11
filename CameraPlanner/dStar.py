#!/usr/bin/env python
'''
Package providing helper classes and functions for performing graph search operations for planning.
'''
import sys
import numpy as np
import heapq
import matplotlib.pyplot as plotter
from math import hypot, sqrt

_DEBUG = False
_DEBUG_END = True
_ACTIONS = ['u','d','l','r']
_ACTIONS_2 = ['u','d','l','r','ne','nw','sw','se']
_T = 2
_X = 1
_Y = 0
_GOAL_COLOR = 0.75
_INIT_COLOR = 0.25
_PATH_COLOR_RANGE = _GOAL_COLOR-_INIT_COLOR
_VISITED_COLOR = 0.9


class GridMap:
    '''
    Class to hold a grid map for navigation. Reads in a map.txt file of the format
    0 - free cell, x - occupied cell, g - goal location, i - initial location.
    Additionally provides a simple transition model for grid maps and a convience function
    for displaying maps.
    '''
    def __init__(self, map_path=None):
        '''
        Constructor. Makes the necessary class variables. Optionally reads in a provided map
        file given by map_path.

        map_path (optional) - a string of the path to the file on disk
        '''
        self.rows = None
        self.cols = None
        self.goal = None
        self.init_pos = None
        self.occupancy_grid = None
        if map_path is not None:
            self.read_map(map_path)

    def read_map(self, map_path):
        '''
        Read in a specified map file of the format described in the class doc string.

        map_path - a string of the path to the file on disk
        '''
        map_file = open(map_path,'r')
        lines = [l.rstrip().lower() for l in map_file.readlines()]
        map_file.close()
        self.rows = len(lines)
        self.cols = max([len(l) for l in lines])
        if _DEBUG:
            print('rows', self.rows)
            print('cols', self.cols)
            print(lines)
        self.occupancy_grid = np.zeros((self.rows, self.cols), dtype=np.bool)
        for r in range(self.rows):
            for c in range(self.cols):
                if lines[r][c] == 'x':
                    self.occupancy_grid[r][c] = True
                if lines[r][c] == 'g':
                    self.goal = (r,c)
                elif lines[r][c] == 'i':
                    self.init_pos = (r,c,0)

    def is_goal(self,s):
        '''
        Test if a specifid state is the goal state

        s - tuple describing the state as (row, col) position on the grid.

        Returns - True if s is the goal. False otherwise.
        '''
        return (s[_X] == self.goal[_X] and
                s[_Y] == self.goal[_Y])

    def transition(self, s, a):
        '''
        Transition function for the current grid map.

        s - tuple describing the state as (row, col) position on the grid.
        a - the action to be performed from state s

        returns - s_prime, the state transitioned to by taking action a in state s.
        If the action is not valid (e.g. moves off the grid or into an obstacle)
        returns the current state.
        '''
        new_pos = list(s[:])
        # Ensure action stays on the board
        if a == 'u':
            if s[_Y] > 0:
                new_pos[_Y] -= 1
        elif a == 'd':
            if s[_Y] < self.rows - 1:
                new_pos[_Y] += 1
        elif a == 'l':
            if s[_X] > 0:
                new_pos[_X] -= 1
        elif a == 'r':
            if s[_X] < self.cols - 1:
                new_pos[_X] += 1
        elif a == 'ne':
            if s[_Y] > 0 and s[_X] < self.cols - 1:
                new_pos[_X] += 1 # moves one to the right
                new_pos[_Y] -= 1 # moves one up
        elif a == 'nw':
            if s[_Y] > 0 and s[_X] > 0:
                new_pos[_X] -= 1 # moves one to the left
                new_pos[_Y] -= 1 # moves one up
        elif a == 'se':
            if s[_Y] < self.rows - 1 and s[_X] < self.cols - 1:
                new_pos[_X] += 1 # moves one to the right
                new_pos[_Y] += 1 # moves one down
        elif a == 'sw':
            if s[_Y] < self.rows - 1 and s[_X] > 0:
                new_pos[_X] -= 1 # moves one to the left
                new_pos[_Y] += 1 # moves one down
        else:
            print('Unknown action:', str(a))

        # Test if new position is clear
        if self.occupancy_grid[new_pos[0], new_pos[1]]:
            s_prime = tuple(s)
        else:
            s_prime = tuple(new_pos)
        return s_prime

    def pose(self, s, a):
        '''
        Transition function for the current grid map.

        s - tuple describing the state as (row, col) position on the grid.
        a - the action to be performed from state s

        returns - s_prime, the state transitioned to by taking action a in state s.
        If the action is not valid (e.g. moves off the grid or into an obstacle)
        returns the current state.
        '''
        new_pos = list(s[:])
        # Ensure action stays on the boar
        if a == 'f':
            if s[_T] == 0:
                if s[_Y] > 0:
                    new_pos[_Y] -= 1
            elif s[_T] == 4:
                if s[_Y] < self.rows - 1:
                    new_pos[_Y] += 1
            elif s[_T] == 2:
                if s[_X] > 0:
                    new_pos[_X] -= 1
            elif s[_T] == 6:
                if s[_X] < self.cols - 1:
                    new_pos[_X] += 1
            elif s[_T] == 7:
                if s[_Y] > 0 and s[_X] < self.cols - 1:
                    new_pos[_X] += 1 # moves one to the right
                    new_pos[_Y] -= 1 # moves one up
            elif s[_T] == 1:
                if s[_Y] > 0 and s[_X] > 0:
                    new_pos[_X] -= 1 # moves one to the left
                    new_pos[_Y] -= 1 # moves one up
            elif s[_T] == 5:
                if s[_Y] < self.rows - 1 and s[_X] < self.cols - 1:
                    new_pos[_X] += 1 # moves one to the right
                    new_pos[_Y] += 1 # moves one down
            elif s[_T] == 3:
                if s[_Y] < self.rows - 1 and s[_X] > 0:
                    new_pos[_X] -= 1 # moves one to the left
                    new_pos[_Y] += 1 # moves one down
        elif a == 'tl':
            new_pos[_T] += 1
        elif a == 'tr':
            new_pos[_T] -= 1
        else:
            print('Unknown action:', str(a))
        if abs(new_pos[_T]) > 7:
            new_pos[_T] = new_pos[_T] % 8
        # Test if new position is clear
        if self.occupancy_grid[new_pos[0], new_pos[1]]:
            s_prime = tuple(s)
        else:
            s_prime = tuple(new_pos)
        return s_prime

    def display_map(self, path=[], visited={}, filename=None):
        '''
        Visualize the map read in. Optionally display the resulting plan and visisted nodes

        path - a list of tuples describing the path take from init to goal
        visited - a set of tuples describing the states visited during a search
        filename - relative path to file where image will be saved
        '''
        display_grid = np.array(self.occupancy_grid, dtype=np.float32)

        # Color all visited nodes if requested
        for v in visited:
            display_grid[v[0],v[1]] = _VISITED_COLOR
        # Color path in increasing color from init to goal
        for i, p in enumerate(path):
            disp_col = _INIT_COLOR + _PATH_COLOR_RANGE*(i+1)/len(path)
            display_grid[p[0],p[1]] = disp_col

        display_grid[self.init_pos[0],self.init_pos[1]] = _INIT_COLOR
        display_grid[self.goal] = _GOAL_COLOR

        # Plot display grid for visualization
        imgplot = plotter.imshow(display_grid)
        # Set interpolation to nearest to create sharp boundaries
        imgplot.set_interpolation('nearest')
        # Set color map to diverging style for contrast
        imgplot.set_cmap('Spectral')
        if filename is not None:
            plotter.savefig(filename)
        plotter.show()
        
    def euclidean_heuristic(self, s):
        '''
        Heuristic estimating cost based on euclidean distance from given state to goal state

        s - tuple describing the state as (row, col) position on the grid.

        returns - floating point estimate of the cost to the goal from state s
        '''
        euclidan_distance = sqrt(abs(self.goal[_X]-s[_X])**2 + abs(self.goal[_Y]-s[_Y])**2)
        return euclidan_distance

    def manhattan_heuristic(self, s):
        '''
        Heuristic estimating cost based on euclidean distance from given state to goal state

        s - tuple describing the state as (row, col) position on the grid.

        returns - floating point estimate of the cost to the goal from state s
        '''
        manhattan_distance = abs(self.goal[_X]-s[_X]) + abs(self.goal[_Y]-s[_Y])
        return manhattan_distance

class SearchNode:
    def __init__(self, s, A, parent=None, parent_action=None, cost=0):
        '''
        s - the state defining the search node
        A - list of actions
        parent - the parent search node
        parent_action - the action taken from parent to get to s
        '''
        self.parent = parent
        self.cost = cost
        self.parent_action = parent_action
        self.state = s[:]
        self.actions = A[:]

    def __str__(self):
        '''
        Return a human readable description of the node
        '''
        return str(self.state) + ' ' + str(self.actions)+' '+str(self.parent)+' '+str(self.parent_action)
    def __lt__(self, other):
        return self.cost < other.cost

class PriorityQ:
    '''
    Priority queue implementation with quick access for membership testing
    Setup currently to only with the SearchNode class
    '''
    def __init__(self):
        '''
        Initialize an empty priority queue
        '''
        self.l = [] # list storing the priority q
        self.s = set() # set for fast membership testing

    def __contains__(self, x):
        '''
        Test if x is in the queue
        '''
        return x in self.s

    def push(self, x, cost):
        '''
        Adds an element to the priority queue.
        If the state already exists, we update the cost
        '''
        if x.state in self.s:
            return self.replace(x, cost)
        heapq.heappush(self.l, (cost, x))
        self.s.add(x.state)

    def pop(self):
        '''
        Get the value and remove the lowest cost element from the queue
        '''
        x = heapq.heappop(self.l)
        self.s.remove(x[1].state)
        return x[1]

    def peak(self):
        '''
        Get the value of the lowest cost element in the priority queue
        '''
        x = self.l[0]
        return x[1]

    def __len__(self):
        '''
        Return the number of elements in the queue
        '''
        return len(self.l)

    def replace(self, x, new_cost):
        '''
        Removes element x from the q and replaces it with x with the new_cost
        '''
        for y in self.l:
            if x.state == y[1].state:
                self.l.remove(y)
                self.s.remove(y[1].state)
                break
        heapq.heapify(self.l)
        self.push(x, new_cost)

    def get_cost(self, x):
        '''
        Return the cost for the search node with state x.state
        '''
        for y in self.l:
            if x.state == y[1].state:
                return y[0]

    def __str__(self):
        '''
        Return a string of the contents of the list
        '''
        return str(self.l)

def dfs(init_state, f, is_goal, actions):
    '''
    Perform depth first search on a grid map.

    init_state - the intial state on the map
    f - transition function of the form s_prime = f(s,a)
    is_goal - function taking as input a state s and returning True if its a goal state
    actions - set of actions which can be taken by the agent

    returns - ((path, action_path), visited) or None if no path can be found
    path - a list of tuples. The first element is the initial state followed by all states
        traversed until the final goal state
    action_path - the actions taken to transition from the initial state to goal state
    '''
    # Fill me in!
    frontier = [] # Search stack
    visitedSet = [] # Visited set
    n0 = SearchNode(init_state,actions)
    frontier.append(n0)
    while len(frontier) > 0:
        # Take of last most recent element
        ni = frontier.pop()
        if ni.state not in visitedSet:
            visitedSet.append(ni.state)
            if is_goal(ni.state):
                return (backpath(ni),visitedSet)
            else:
                for a in actions:
                    sPrime = f(ni.state,a)
                    nPrime = SearchNode(sPrime,actions,ni,a)
                    frontier.append(nPrime)
    return (backpath(ni),visitedSet)

def ids(init_state, f, is_goal, actions, m=1):
    '''
    Perform depth first search on a grid map.

    init_state - the intial state on the map
    f - transition function of the form s_prime = f(s,a)
    is_goal - function taking as input a state s and returning True if its a goal state
    actions - set of actions which can be taken by the agent

    returns - ((path, action_path), visited) or None if no path can be found
    path - a list of tuples. The first element is the initial state followed by all states
        traversed until the final goal state
    action_path - the actions taken to transition from the initial state to goal state
    '''
    frontier = [] # Search stack
    for depth in range(m,100):
        visitedSet = {} # Visited set
        n0 = SearchNode(init_state,actions)
        frontier.append(n0)
        while len(frontier) > 0:
            # Take off most recent element
            ni = frontier.pop()
            currPath = backpath(ni)
            if len(currPath[0]) > depth:
                continue
            if ni.state not in visitedSet or visitedSet[ni.state] > len(currPath[0]):
                visitedSet[ni.state]=len(currPath[0])
                if is_goal(ni.state):
                    return (backpath(ni),visitedSet)
                else:
                    for a in actions:
                        sPrime = f(ni.state,a)
                        nPrime = SearchNode(sPrime,actions,ni,a)
                        frontier.append(nPrime)
    return (backpath(ni),visitedSet)

def bfs(init_state, f, is_goal, actions):
    '''
    Perform breadth first search on a grid map.

    init_state - the intial state on the map
    f - transition function of the form s_prime = f(s,a)
    is_goal - function taking as input a state s and returning True if its a goal state
    actions - set of actions which can be taken by the agent

    returns - ((path, action_path), visited) or None if no path can be found
    path - a list of tuples. The first element is the initial state followed by all states
        traversed until the final goal state
    action_path - the actions taken to transition from the initial state to goal state
    '''
    frontier = [] # Search stack
    visitedSet = [] # Visited set
    n0 = SearchNode(init_state,actions)
    frontier.append(n0)
    while len(frontier) > 0:
        # Take off least recent element
        ni = frontier.pop(0)
        if ni.state not in visitedSet:
            visitedSet.append(ni.state)
            if is_goal(ni.state):
                return (backpath(ni),visitedSet)
            else:
                for a in actions:
                    sPrime = f(ni.state,a)
                    nPrime = SearchNode(sPrime,actions,ni,a)
                    frontier.append(nPrime)
    return (backpath(ni),visitedSet)

def uniform_cost_search(init_state, f, is_goal, actions):
    frontier = PriorityQ() # Search stack
    visitedSet = [] # Visited set
    n0 = SearchNode(init_state,actions)
    frontier.push(n0,cost=0)
    while len(frontier) > 0:
        # Take off least costly element
        ni = frontier.pop()
        if ni.state not in visitedSet:
            visitedSet.append(ni.state)
            if is_goal(ni.state):
                return (backpath(ni),visitedSet)
            else:
                for a in actions:
                    sPrime = f(ni.state,a)
                    if a in _ACTIONS:
                        cost = ni.cost+1
                    elif a in _ACTIONS_2:
                        cost = ni.cost + 1.5
                    else:
                        if a == 'f':
                            if sPrime[2] == 0 or sPrime[2] == 2 or sPrime[2] == 4 or sPrime[2] == 6:
                                cost = ni.cost + 1
                            else:
                                cost = ni.cost + 1.5
                        else:
                            cost = ni.cost + 0.25
                    if sPrime == ni.state:
                        cost = ni.cost
                    nPrime = SearchNode(sPrime,actions,ni,a,cost)
                    if nPrime.state in frontier:
                        if frontier.get_cost(nPrime) > cost:
                            frontier.push(nPrime,cost)
                    else:
                        frontier.push(nPrime,cost)
    return (backpath(ni),visitedSet)

def a_star_search(init_state, f, is_goal, actions, h):
    '''
    init_state - value of the initial state
    f - transition function takes input state (s), action (a), returns s_prime = f(s, a)
        returns s if action is not valid
    is_goal - takes state as input returns true if it is a goal state
        actions - list of actions available
    h - heuristic function, takes input s and returns estimated cost to goal
        (note h will also need access to the map, so should be a member function of GridMap)
    '''
    frontier = PriorityQ() # Search stack
    visitedSet = [] # Visited set
    n0 = SearchNode(init_state,actions)
    frontier.push(n0,cost=0)
    while len(frontier) > 0:
        # Take off least costly element
        ni = frontier.pop()
        if ni.state not in visitedSet:
            visitedSet.append(ni.state)
            if is_goal(ni.state):
                return (backpath(ni),visitedSet)
            else:
                for a in actions:
                    sPrime = f(ni.state,a)
                    if a in _ACTIONS:
                        cost = h(ni.state) + ni.cost+1
                    elif a in _ACTIONS_2:
                        cost = h(ni.state) + ni.cost + 1.5
                    else:
                        if a == 'f':
                            if sPrime[2] == 0 or sPrime[2] == 2 or sPrime[2] == 4 or sPrime[2] == 6:
                                cost = h(ni.state) + ni.cost + 1
                            else:
                                cost = h(ni.state) + ni.cost + 1.5
                        else:
                            cost = h(ni.state) + ni.cost + 0.25
                    if sPrime == ni.state:
                        cost = h(ni.state) + ni.cost
                    nPrime = SearchNode(sPrime,actions,ni,a,cost)
                    if nPrime.state in frontier:
                        if frontier.get_cost(nPrime) > cost:
                            frontier.push(nPrime,cost)
                    else:
                        frontier.push(nPrime,cost)
    return (backpath(ni),visitedSet)

def backpath(node):
    '''
    Function to determine the path that lead to the specified search node

    node - the SearchNode that is the end of the path

    returns - a tuple containing (path, action_path) which are lists respectively of the states
    visited from init to goal (inclusive) and the actions taken to make those transitions.
    '''
    path = []
    action_path = []
    tempPath = []
    tempAction = []
    tempNode = node
    while tempNode != None:
        tempPath.append(tempNode.state)
        tempAction.append(tempNode.parent_action)
        tempNode = tempNode.parent
    for i in range(len(tempPath)-1,-1,-1):
        path.append(tempPath[i])
        action_path.append(tempAction[i])
    return (path, action_path)

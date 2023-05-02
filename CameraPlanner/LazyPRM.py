#Library for Lazy PRM
from ssl import ALERT_DESCRIPTION_BAD_CERTIFICATE_STATUS_RESPONSE
import numpy as np
import matplotlib.pyplot as plotter
from math import sqrt
from collisions import PolygonEnvironment
import time
import heapq

_DEBUG = False
_DEBUG_END = True

def fake_in_collision(q):
	'''
	We never collide with this function!
	'''
	return False

def euclidean_heuristic(s, goal):
	'''
	Euclidean heuristic function

	s - configuration vector
	goal - goal vector

	returns - floating point estimate of the cost to the goal from state s
	'''
	return np.linalg.norm(s - goal)

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
		if tuple(x.state.tolist()) in self.s:
			return self.replace(x, cost)
		heapq.heappush(self.l, (cost, x))
		self.s.add(tuple(x.state.tolist()))

	def pop(self):
		'''
		Get the value and remove the lowest cost element from the queue
		'''
		x = heapq.heappop(self.l)
		self.s.remove(tuple(x[1].state.tolist()))
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
			if tuple(x.state.tolist()) == tuple(y[1].state.tolist()):
				self.l.remove(y)
				self.s.remove(tuple(y[1].state.tolist()))
				break
		heapq.heapify(self.l)
		self.push(x, new_cost)

	def get_cost(self, x):
		for y in self.l:
			if tuple(x.state.tolist()) == tuple(y[1].state.tolist()):
				return y[0]

	def __str__(self):
		return str(self.l)

def backpath(node):
	'''
	Function to determine the path that lead to the specified search node

	node - the SearchNode that is the end of the path

	returns - a tuple containing (path, action_path) which are lists respectively of the states
	visited from init to goal (inclusive) and the actions taken to make those transitions.
	'''
	path = []
	while node.parent is not None:
		path.append(node.state)
		node = node.parent
	path.append(node.state)
	path.reverse()
	return path

class StraightLinePlanner:
	def __init__(self, step_size, collision_func = None):
		self.in_collision = collision_func
		self.epsilon = step_size
		if collision_func is None:
			self.in_collision = fake_in_collision

	def plan(self, start, goal):
		'''
		Check if edge is collision free, taking epsilon steps towards the goal
		Returns: None / False if edge in collsion
				 Plan / True if edge if free
		'''

		q = start;
		if self.in_collision is not None and self.in_collision(q):
			return None;
		while True:
			dists = goal - q;
			dist = sqrt(np.sum(dists ** 2));
			if (dist < self.epsilon):
				q = goal;
				if self.in_collision is not None:
					return not self.in_collision(q);
			else:
				q = dists * (self.epsilon / dist) + q;

			if self.in_collision is not None and self.in_collision(q):
				return None;

class RoadMapNode:
	'''
	Nodes to be used in a built RoadMap class
	'''
	def __init__(self, state, cost=0, parent=None):
		self.state = np.array(state)
		self.neighbors = {}
		self.cost = cost
		self.parent = parent

	def add_neighbor(self, n_new):
		'''
		n_new - new neighbor
		'''
		self.neighbors[n_new] = 0;

	def is_neighbor(self, n_test):
		'''
		Test if n_test is already our neighbor
		'''
		for n in self.neighbors:
			if np.linalg.norm(n.state - n_test.state) == 0.0:
				return True
		return False

	def distance_to(self, q):
		'''
		q - state of other node
		returns distance to node
		'''
		return np.linalg.norm(self.state - q);

	def __eq__(self, other):
		return np.linalg.norm(self.state - other.state) == 0.0
	
	def __hash__(self):
		return hash(tuple(self.state))
	
	def __lt__(self, other):
		return self.cost < other.cost

class RoadMap:
	'''
	Class to store a built roadmap for searching in our multi-query PRM
	'''
	def __init__(self):
		self.nodes = []

	def add_node(self, q, neighbors):
		'''
		Add a node to the roadmap. Connect it to its neighbors
		'''
		# Avoid adding duplicates
		node = RoadMapNode(q);
		self.nodes.append(node)
		for n in neighbors:
			node.add_neighbor(n)
			if not n.is_neighbor(node):
				n.add_neighbor(node)

		return node;
	def remove_node(self, q):
		'''
		Add a node to the roadmap. Connect it to its neighbors
		'''
		# Avoid adding duplicates
		for n in self.nodes:
			if(np.array_equal(n.state, q)):
				self.nodes.remove(n);
				break;
		
	def get_states_and_edges(self):
		states = np.array([n.state for n in self.nodes])

		edges = []
		for n in self.nodes:
			for n_n in n.neighbors:
				edges.append((n.state, n_n.state, n.neighbors[n_n]));
		return (states, edges)

class LZ_PRM:
	def __init__(self, 
				 local_planner: StraightLinePlanner, 
				 num_dimensions, 
				 lims = None,
				 collision_func:PolygonEnvironment.test_collisions = None, 
				 n_nodes = 500, #the desired number of nodes in the final roadmap (at least)
				 radius=2.0, 
				 epsilon=0.1):
		self.local_planner = local_planner
		self.r = radius
		self.N = n_nodes
		self.n = num_dimensions
		self.epsilon = epsilon

		self.in_collision = collision_func
		if collision_func is None:
			self.in_collision = fake_in_collision

		# Setup range limits
		self.limits = lims
		if self.limits is None:
			self.limits = []
			for n in range(num_dimensions):
				self.limits.append([0,100])
			self.limits = np.array(self.limits)

		self.ranges = self.limits[:,1] - self.limits[:,0]

		# Build the roadmap instance
		self.T = RoadMap()

	def build_lazy_prm(self, reset=False):
		'''
		reset - empty the current roadmap if requested
		'''
		if reset:
			self.T = RoadMap()

		for i in range(self.N):
			q = self.sample();
			print(i)
			self.T.add_node(q, self.find_valid_neighbors(q, self.r));
		print('done')

	def find_valid_neighbors(self, q, r):
		'''
		Find the nodes that are close to n_query and can be attached by the local planner
		returns - list of neighbors reached by the local planner
		'''
		valid_neighbors = [];
		
		for n in self.T.nodes:
			if (n.distance_to(q) < r):
				valid_neighbors.append(n);

		return valid_neighbors
	
	def query(self, start, goal):
		'''
		Generate a path from start to goal using the built roadmap
		returns - Path of configurations if in roadmap, None otherwise
		'''
		start_node = self.T.add_node(start, self.find_valid_neighbors(start, self.r));
		goal_node = self.T.add_node(goal, self.find_valid_neighbors(goal, self.r));

		def is_goal(x):
			'''
			Test if a sample is at the goal
			'''
			return np.linalg.norm(x - goal) < self.epsilon        
	
		# Use uniform cost search to search for a path 
		path =None;
		time =0
		while(path == None and time <100):
			print(f"start_node neighbor:{len(start_node.neighbors)}")
			print(f"goal_node neighbor:{len(goal_node.neighbors)}")
			path, visited = self.astar(start_node, is_goal, goal_node.distance_to, self.local_planner);
			time +=1
			if(path == None):
       
				for i in range(self.N):

					q = self.sample();
					for n in self.T.nodes:
						if(np.array_equal(n.state, q)):
							continue;
					self.T.add_node(q, self.find_valid_neighbors(q, self.r));
				print(len(self.T.nodes))
				self.T.remove_node(start);
				self.T.remove_node(goal);
				start_node = self.T.add_node(start, self.find_valid_neighbors(start, self.r))
				goal_node = self.T.add_node(goal, self.find_valid_neighbors(goal, self.r))
			
			print(f"time:{time}")
   
		if path is not None and len(path) > 1:
			return path, visited
		
		return None, visited

	def astar(self, init_node, is_goal, goal_dist, planner):
		'''
		Perform graph search on the roadmap
		'''
		frontier = PriorityQ();
		visited = set();
		frontier.push(init_node,0);

		while len(frontier) > 0:
			n_i = frontier.pop();
			visited.add(tuple(n_i.state.tolist()));
			if is_goal(n_i.state):
				return (backpath(n_i), visited);
			else:
				for n in n_i.neighbors:
					if tuple(n.state.tolist()) in visited:
						continue;
					cost = n_i.distance_to(n.state);
					heuristic = goal_dist(n.state);
					fcost = frontier.get_cost(n);
					ncost = n_i.cost + cost;
					cph = ncost + heuristic;
					if ((fcost is None) or (fcost >= cph)):
						flag = n_i.neighbors[n];
						if (flag == 0):
							if (planner.plan(n_i.state, n.state) is not None):
								n_i.neighbors[n] = 1;
								n.neighbors[n_i] = 1;
								flag = 1;
							else:
								n_i.neighbors[n] = -1;
								n.neighbors[n_i] = -1;
								flag = -1;

						if (flag == 1):
							frontier.push(n, cph);
							n.cost = ncost;
							n.parent = n_i;
		return (None, visited);
	
	def sample(self):
		'''
		Sample a new configuration
		Returns a configuration of size self.n bounded in self.limits
		'''
		return (np.random.rand(self.n) * self.ranges) + self.limits[:,0];



def saveFig(name,close = True):
	plotter.savefig(name + ".png")
	if(close):
		plotter.close()

  
def test_prm_env(num_samples=10, step_length=1, env='./env0.txt'):
	pe = PolygonEnvironment()
	pe.read_env(env)

	dims = len(pe.start)
	start_time = time.time()

	local_planner = StraightLinePlanner(step_length, pe.test_collisions)
	lzprm = LZ_PRM( local_planner,
				  dims, 
				  lims = pe.lims,
				  n_nodes = num_samples,
				  radius=15,
				  epsilon=step_length,
				  collision_func=pe.test_collisions)
	print('Building PRM')
	lzprm.build_lazy_prm()
	build_time = time.time() - start_time
	print('Build time', build_time)
	
	start_time = time.time()
	print('Finding Plan')
	plan, visited = lzprm.query(pe.start, pe.goal)
	print('Plan Found')
	build_time = time.time() - start_time
	print('Build time', build_time)

	pe.draw_env(show=False)
	pe.draw_plan(plan, lzprm,False,False,True)

	start_time = time.time()
	print('Finding Plan')
	plan, visited = lzprm.query([50,0], [35,130])
	print('Plan Found')
	build_time = time.time() - start_time
	print('Build time', build_time)

	pe.draw_env(show=False)
	pe.draw_plan(plan, lzprm,False,False,True)

	start_time = time.time()
	print('Finding Plan')
	plan, visited = lzprm.query(pe.start, pe.goal)
	print('Plan Found')
	build_time = time.time() - start_time
	print('Build time', build_time)

	pe.draw_env(show=False)
	pe.draw_plan(plan, lzprm,False,False,True)

	start_time = time.time()
	print('Finding Plan')
	plan, visited = lzprm.query([50,0], [35,130])
	print('Plan Found')
	build_time = time.time() - start_time
	print('Build time', build_time)

	pe.draw_env(show=False)
	pe.draw_plan(plan, lzprm,False,False,True)

	start_time = time.time()
	print('Finding Plan')
	plan, visited = lzprm.query([-60,60], [85,100])
	print('Plan Found')
	build_time = time.time() - start_time
	print('Build time', build_time)

	pe.draw_env(show=False)
	pe.draw_plan(plan, lzprm,False,False,True)

	return plan, lzprm, visited

if __name__== "__main__":
  test_prm_env()

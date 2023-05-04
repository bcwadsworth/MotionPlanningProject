import sys
import os
import time
import numpy as np
from casadi import SX, MX, Function, nlpsol, vertcat, norm_2, mmin, power, sum2, mtimes, transpose, repmat
from math import sqrt

from LazyPRM import LZ_PRM, StraightLinePlanner
from collisions import PolygonEnvironment;

class Optimizer:
	def __init__(self, planner, obsts = [], alpha = 1000, beta = 50000, gamma = 1):

		self.lb_theta = []         # Lower bounds on joint positions, length N
		self.ub_theta = []         # Upper bounds on joint positions, length N
		self.g = []                # Constraints to be satisfied
		self.lbg = []              # Lower bounds on constraints 
		self.ubg = []              # Upper bounds on constraints
		self.running_cost = 0      # Running cost for optimization
		self.planner = planner
		self.dof = self.planner.n
		self.obsts = obsts;
		self.alpha = alpha;
		self.beta = beta;
		self.gamma = gamma;
	
	def query(self, start, goal):
		"""
		Setup the optimization problem for 2 and run the solver.
		Input:
			start: Configuration Position of Start
			goal: Configuration Position of Goal
			obsts: Configuration space collection of obstacle points. Rows are different points.
			alpha: Parameter governing influence of smoothing term
			beta: Parameter governing influence of obstacle avoidance term
			gamma: Parameter governing norm of change of states per timestep
		"""

		initplan, _ = self.planner.query(start, goal);
		if initplan is None:
			return None;

		lb_theta = []         # Lower bounds on joint positions, length N
		ub_theta = []         # Upper bounds on joint positions, length N
		g = []                # Constraints to be satisfied
		lbg = []              # Lower bounds on constraints 
		ubg = []              # Upper bounds on constraints
		theta_traj = []
		theta_guess = []
		running_cost = 0      # Running cost for optimization

		def cost_function(goal, theta, theta_prev):
			t = repmat(transpose(theta), self.obsts.shape[0], 1)
			s = t - self.obsts;
			p = power(s, 2);
			s = sum2(p);
			m = mmin(s);
			return norm_2(theta - goal)**2 + self.alpha * norm_2(theta-theta_prev)**2 + self.beta * 1/m;

		# Setup the CasADi function for computing cost given defined cost function
		theta = SX.sym('theta', self.dof)  # Config Space Position
		theta_pre = SX.sym('theta_prev', self.dof);

		cost = cost_function(goal, theta, theta_pre)
		F = Function('F', [theta, theta_pre], [cost], ['theta', 'theta_prev'], ['cost'])

		# Add initial conditions
		theta_0 = MX.sym('theta_0', self.dof)
		theta_prev = theta_0
		theta_traj += [theta_0]
		theta_guess += initplan[0].tolist();
		# Setting upper/lower bounds the same => equality constraint
		lb_theta += initplan[0].tolist();
		ub_theta += initplan[0].tolist();

		# Add symbolic variables to build the optimization problem
		k = 1
		for i, step in enumerate(initplan):
			if (i == 0):
				continue;

			q = initplan[i-1];
			while True:
				dists = step - q;
				dist = sqrt(np.sum(dists ** 2));
				flag = False;
				if (dist < self.gamma):
					q = step;
					flag = True;
				else:
					q = dists * (self.gamma / dist) + q;
					
				# Create a symbolic variable for joint state at this timestep
				theta_k = MX.sym('theta_' + str(k), self.dof)
				k += 1;

				# Build symbolic trajectory, these are the variables being optimized
				theta_traj += [theta_k];
				
				# Add upper/lower limits on the state (joint limit constraints)
				lb_theta += self.planner.limits[:,0].tolist();
				ub_theta += self.planner.limits[:,1].tolist();

				# Provide initial guess for solution
				theta_guess += q.tolist();

				# Compute the cost for this timestep and add to running cost
				running_cost += F(theta=theta_k, theta_prev=theta_prev)['cost']
			
				# Limit Instantaneous Velocity
				g += [norm_2(theta_k - theta_prev)];
				lbg += [0];
				ubg += [self.gamma];

				theta_prev = theta_k

				if flag:
					break;

		# Handle final timestep
		theta_N = MX.sym('theta_N', self.dof)
		theta_traj += [theta_N]
		lb_theta += goal.tolist()
		ub_theta += goal.tolist()
		theta_guess += goal.tolist()

		# Velocity = 0 at end
		g += [theta_N - theta_prev];
		lbg += [0 for _ in range(self.dof)];
		ubg += [0 for _ in range(self.dof)];

		# Let's solve the optimization problem!
		problem = {
			'x': vertcat(*theta_traj),
			'g': vertcat(*g),
			'f': running_cost
		}
		solver = nlpsol("nlp", "ipopt", problem)
		solved = solver(
			x0=theta_guess,
			lbx=lb_theta,
			ubx=ub_theta,
			lbg=lbg,
			ubg=ubg)

		soln = solved['x'].full().ravel();
		trajectory = np.split(soln, np.shape(soln)[0]/self.dof)
		return initplan, trajectory;

def test_opt(num_samples=1000, step_length=1, env='./env0.txt'):
	pe = PolygonEnvironment()
	pe.read_env(env)

	dims = len(pe.start)
	
	obsts = np.vstack(pe.polygons);
	start_time = time.time()

	local_planner = StraightLinePlanner(step_length, pe.test_collisions)
	lzprm = LZ_PRM( local_planner,
				  dims, 
				  lims = pe.lims,
				  n_nodes = num_samples,
				  radius=10,
				  epsilon=step_length,
				  collision_func=pe.test_collisions)
	print('Building PRM')
	lzprm.build_lazy_prm()
	build_time = time.time() - start_time
	print('Build time', build_time)

	opt = Optimizer(lzprm, obsts= obsts);
	
	start_time = time.time()
	print('Finding Plan')
	initplan, optplan = opt.query(pe.start, pe.goal)
	print('Plan Found')
	build_time = time.time() - start_time
	print('Build time', build_time)

	#pe.draw_env(show=False);
	#pe.draw_plan(initplan, lzprm, False,False,True);
	#pe.draw_plan(optplan, None, False,True,True);

	start_time = time.time()
	print('Finding Plan')
	initplan, optplan = opt.query(np.array([50,0]), np.array([35,130]))
	print('Plan Found')
	build_time = time.time() - start_time
	print('Build time', build_time)

	pe.draw_env(show=False);
	#pe.draw_plan(initplan, lzprm, False,True,True);
	pe.draw_plan(optplan, None, False,False,True);

	start_time = time.time()
	print('Finding Plan')
	initplan, optplan = opt.query(pe.start, pe.goal)
	print('Plan Found')
	build_time = time.time() - start_time
	print('Build time', build_time)

	pe.draw_env(show=False);
	#pe.draw_plan(initplan, lzprm, False,True,True);
	pe.draw_plan(optplan, None, False,False,True);

	start_time = time.time()
	print('Finding Plan')
	initplan, optplan = opt.query(np.array([50,0]), np.array([35,130]))
	print('Plan Found')
	build_time = time.time() - start_time
	print('Build time', build_time)

	pe.draw_env(show=False);
	#pe.draw_plan(initplan, lzprm, False,True,True);
	pe.draw_plan(optplan, None, False,False,True);
	
	start_time = time.time()
	print('Finding Plan')
	initplan, optplan = opt.query(np.array([-60,60]), np.array([85,100]))
	print('Plan Found')
	build_time = time.time() - start_time
	print('Build time', build_time)

	pe.draw_env(show=False);
	#pe.draw_plan(initplan, lzprm, False,True,True);
	pe.draw_plan(optplan, None, False,False,True);

if __name__== "__main__":
  test_opt()
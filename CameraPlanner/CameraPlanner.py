import numpy as np
from math import pi
import time
import vrepWrapper
import LazyPRM

num_samples=5000
step_length=0.25
radius = 10
epsilon = 0.1

local_planner = LazyPRM.StraightLinePlanner(step_length)

environment = vrepWrapper.vrepWrapper()

dims = len(environment.start)
start_time = time.time()

LZ_prm = LazyPRM.LZ_PRM(local_planner,
          dims,
          lims = environment.lims,
          collision_func=environment.test_collisions,
          n_nodes = num_samples,
          radius=radius,
          epsilon=epsilon)
LZ_prm.build_lazy_prm()
plan, visited = LZ_prm.query(environment.start, environment.goal)

environment.vrepReset()

run_time = time.time() - start_time
print('plan:', plan)
print('run_time =', run_time)

debugThing = environment.draw_plan(plan, LZ_prm,False,True,True)
print('all done')

time.sleep(10)
environment.vrepStop()
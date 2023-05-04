This program includes a LazyPRM, a collisions.py, some Vrep files, and the CameraPlanner.py.

The main code is run from CameraPlanner.py. This uses CoppeliaSim and simulates a 7-DOF robotic arm in an environment. 

The code runs a simulation for a camera on a robotic arm. The code first uses Lazy PRM to create a roadmap of possible
nodes and their neighbors. Normally PRM checks for collisions when it is creating this roadmap but we wanted the code
to work for moving obstacles. Obstacle checking does not happen until the second step. After a roadmap is randomly 
generated, a path is found using A*. A* uses the cost and obstacle detection to find the best path. That path is 
optimized using Interior Point Methods.
import numpy as np
import torch
import math

import matplotlib.pyplot as plt
import utils
from linear_constraint_walker import LinearConstraintWalker


A=np.array([[-1,0],
			 [0, -1],
			 [0, 1],
			 [0.2425,    0.9701]]);

b=np.array([[0],
			[0],
			[1],
			[1.2127]])


# B, x0 = utils.largestBallInPolytope(A,b)

# print(f"Largest ellipsoid as B={B} and x0={x0}")

num_steps=10;
my_layer=LinearConstraintWalker(A,b,num_steps, use_max_ellipsoid=False)

all_optimal_points=torch.tensor(np.array([[],[]],dtype=np.float32))

fig, ax = plt.subplots()

dim=A.shape[1]

#for j in range(10000):
for theta in np.arange(0,2*math.pi, 0.01): #[0.93]: #
	# x=torch.rand((dim*(dim+1)/2,1))
	tmp=torch.Tensor(np.array([[math.cos(theta)],[math.sin(theta)],[3000]]));
	# x=torch.Tensor(my_layer.input_numel, 1).uniform_(-1, 1)
	# x=torch.Tensor(np.array([[0.5],[4],[3]]));
	tmp= tmp.repeat(num_steps, 1)
	# x=torch.cat((x,tmp),0)
	x=tmp;
	# print(f"x={x}")
	# print(f"x.mT={x.mT}")
	optimal_point=my_layer.forward(x)
	#print(f"optimal_point.mT={optimal_point.mT}")
	all_optimal_points=torch.cat((all_optimal_points,optimal_point),1)
	print(f"Theta={theta}")
	# my_layer.plotAllSteps(ax)


print(f"all_optimal_points={all_optimal_points}");
all_optimal_points_np=all_optimal_points.numpy();


# plot

ax.scatter(all_optimal_points_np[0,:], all_optimal_points_np[1,:])

utils.plot2DPolyhedron(A,b,ax)

utils.plot2DEllipsoidB(my_layer.B.numpy(),my_layer.x0.numpy(),ax)

ax.set_aspect('equal')
plt.show()

# %%

# x0=torch.tensor(np.array([[0.5],[0.2]]));
# B=scaleEllipsoidB(torch.tensor(B),torch.tensor(A),torch.tensor(b),x0)

# print(f"Using x0={x0} and scaling gives B={B}")

# print(B)
# print(x0)

# a=np.array([[4], [8], [9], [7], [2], [8],[5], [1], [0.5],]);
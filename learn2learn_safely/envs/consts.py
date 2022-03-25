import numpy as np

PLACEMENT_EXTENTS = [-2, -2, 2, 2]

OBSTACLES = ['hazard', 'vase', 'gremlin', 'pillar']

GROUP_OBSTACLES = 0
GROUP_GOAL = 1

# Vases
VASES_COLOR = np.array([0, 1, 1, 1])
VASES_SINK = 4e-5
VASES_DENSITY = 0.001

# Hazards
HAZARDS_COLOR = np.array([0, 0, 1, 1])
HAZARDS_HEIGHT = 1e-2

# Gremlins
GREMLINS_COLOR = np.array([0.5, 0, 1, 1])
GREMLINS_DENSITY = 0.001

# Pillars
PILLARS_HEIGHT = 0.5
PILLARS_COLOR = np.array([.5, .5, 1, 1])

# Goal
GOAL_COLOR = np.array([0, 1, 0, 1])

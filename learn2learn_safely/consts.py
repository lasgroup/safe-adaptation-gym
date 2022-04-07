import os

import numpy as np

import learn2learn_safely

BASE_DIR = os.path.join(os.path.dirname(learn2learn_safely.__file__), 'assets')

PLACEMENT_EXTENTS = [-2, -2, 2, 2]

OBSTACLES = ['hazards', 'vases', 'gremlins', 'pillars']

GROUP_OBSTACLES = 1
GROUP_GOAL = 2
GROUP_OBJECTS = 3

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

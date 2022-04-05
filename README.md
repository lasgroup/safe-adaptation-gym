# learn2learn-safely
A Safety-Gym based benchmark suite for safe meta reinforcement learning

Separate into three categories: MDP variations due to reward and CMDP variations due to costs, MDP variations on the actions (e.g., disable doggo's leg)


# TODO 
1. Think of the different classes of task. For instance, where does the variability in tasks arises from?
2. Think what would be the best implementation. My current hunch is Wrappers of Sgym envs.
3. Think.

# Ideas
1. Generalize across different dynamics
2. Generalize across different tasks (but same robot.)
3. ~~Generalize across different cost values. (to make some agents more safe than others.)~~
4. Action noise scale is resampled (different scale for each dimension of actions).
5. Action scale is resampled (different scale for each dimension of actions).
6. ~~Take one of doggo legs off.~~ (Look for 4. and 5.)
7. Goal attributes (size + color) can change between goal resampling.
8. ~~Extreme transfer (for example, button -> goal)~~
9. ~~How to compose different variablities?~~
10. Radius of gremlins travel.
11. Use meta world and add objects that should not be touched. 
12. Mix the obstacles types but keep the task.
13. Mix tasks keep the types.
14. How parametric and non-parametric transfer can be tested together in a more intelligent way than metaworlds.
15. Think of more tasks
16. Cross validate tasks: in each experiment make a different mixture of training, testing splits.
17. Randomly moving obstacles
18. Egg shaped objects (easier to roll in a certain direction.)
19. All obstacles are red - new task with different shape of obstacle - see if it can generalize.

# Possible tasks
## Objective (keep dynamics and objects, change reward)
1. Follow a circle trajectory
2. Follow a square trajectory
3. Follow a triangle trajectory

## Safety (keep reward, change dynamics and collideable objects)
1. Make obstacles larger/dense
2. Change action sensitivity (sensitivity per joint)
3. Change action noise (noise per joint)

## Transfer (change objects and rewards, keep dynamics)
1. Push box to location
2. Press buttons
3. Go to goal
4. Billiard (hit a ball -> make it hit another specific ball, impose costs on the number of hits?)
5. Soccer.
6. Change sizes of goals/box/buttons

## All tasks
1. Push box to location (box, goal locations)
2. Press buttons (needed button, other buttons)
3. Go to goal (static goal location)
4. Billiard (hit/roll a ball -> make it hit another specific ball, impose costs on the number of hits? If touched ball on prev state but this state not give more reward? Better use gating as in original safety gym. Better not give reward if both robot and second ball are inside goal area. Or reward given the distance of the second ball traveled) (ball to hit, ball that the white ball should hit)
5. Follow a circle trajectory (-)
6. Follow a square trajectory (-)
7. Follow a triangle trajectory (-)
8. Drive box to a platform. (direction to the slope and then change to center.)
9. Hit/roll a ball to a goal. (same ball as billiard?, but just go to goal area)
10. Collect trash (e.g. five balls that get attached to the robot somehow), put in garbage (e.g. for each trash inside garbage plus point) (https://www.roboti.us/forum/index.php?threads/connect-two-object-when-contact.4094/, https://github.com/openai/mujoco-py/issues/570)
11. Follow the leader but don't hit it (Generate random circular splines: https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.CubicSpline.html, https://stackoverflow.com/questions/64796809/how-to-avoid-spline-overlap-with-random-blobby-circle)/Catch a moving goal (instead of follow the leader?)

# Train & Test tasks
## Train
1. Follow a circle trajectory (-)
2. Follow a square trajectory (-)
3. Billiard (hit/roll a ball -> make it hit another specific ball, impose costs on the number of hits? If touched ball on prev state but this state not give more reward?) (ball to hit, ball that the white ball should hit)
4. Push box to location (box, goal locations)
5. Press buttons (needed button, other buttons)
6. Go to goal (goal location)

## Test
1. Follow a triangle trajectory (-)
2. Hit/roll a ball to a goal. (same ball as billiard?, but just go to goal area)
3. Drive box to a platform. (direction to the slope and then change to center.)


# Benchmark
## Non Meta-RL algos
Make sure that (e.g.) PPO can solve each task independantly -- this would be the normalization factor as in Ray et al. 2019
Afterwards, make sure that the baseline algos from Ray et al. 2019 fail on the Meta-RL case.

## Meta-RL algos
Use a lagrangian based approach for the constrained optimization (perhaps even copy paste from safe-agents)

## Vision tasks
Three thoughts: 
1. Use image augmentation (as per with what Danijar proposed)
2. Just a simple CNN encoder + Meta-RL algos. (can we look at metaworld for similar reference?)
3. Can LAMBDA solve it? Probably not because posterior over is unimodal whereas there are multiple tasks?







PILLARS, GREMLINS, HAZARDS, VASES
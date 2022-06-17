# learn2learn-safely
A Safety-Gym based benchmark suite for safe meta reinforcement learning

Separate into three categories: MDP variations due to reward and CMDP variations due to costs, MDP variations on the actions (e.g., disable doggo's leg)

# Ideas
1. Generalize across different dynamics
2. Generalize across different tasks (but same robot.)
3. ~~Generalize across different cost values. (to make some agents more safe than others.)~~
4. Action noise scale is resampled (different scale for each dimension of actions).
5. Action scale is resampled (different scale for each dimension of actions).
6. ~~Take one of doggo legs off.~~ (Look for 4. and 5.)
7. ~~Goal attributes (size + color) can change between goal resampling.~~
8. ~~Extreme transfer (for example, button -> goal)~~
9. ~~How to compose different variablities?~~
10. ~~Radius of gremlins travel.~~
11. ~~Use meta world and add objects that should not be touched.~~ 
12. Mix the obstacles types but keep the task.
13. Mix tasks keep the types.
14. How parametric and non-parametric transfer can be tested together in a more intelligent way than metaworlds.
15. Cross validate tasks: in each experiment make a different mixture of training, testing splits.

# Possible tasks

## Transfer (change objects and rewards, keep dynamics)
1. Push box to location
2. Press buttons
3. Go to goal
4. ~~Billiard (hit a ball -> make it hit another specific ball, impose costs on the number of hits?)~~
5. Soccer.
6. Change sizes of goals/box/buttons

## All tasks
1. Push box to location (box, goal locations)
2. Press buttons (needed button, other buttons)
3. Go to goal (static goal location)
4. ~~Billiard (hit/roll a ball -> make it hit another specific ball, impose costs on the number of hits? If touched ball on prev state but this state not give more reward? Better use gating as in original safety gym. Better not give reward if both robot and second ball are inside goal area. Or reward given the distance of the second ball traveled) (ball to hit, ball that the white ball should hit)~~
5. ~~Follow a circle trajectory (-)~~
6. ~~Follow a square trajectory (-)~~
7. ~~Follow a triangle trajectory (-)~~
8. ~~Drive box to a platform. (direction to the slope and then change to center.)~~
9. Hit/roll a ball to a goal. (same ball as billiard?, but just go to goal area)
10. Collect apples, avoid bad objects (as in CPO paper)
11. ~~Follow the leader but don't hit it (Generate random circular splines: https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.CubicSpline.html, https://stackoverflow.com/questions/64796809/how-to-avoid-spline-overlap-with-random-blobby-circle)/Catch a moving goal (instead of follow the leader?)~~
12. Go to moving goal.

# Benchmark
## Non Meta-RL algos
Make sure that (e.g.) PPO can solve each task independantly -- this would be the normalization factor as in Ray et al. 2019
Afterwards, make sure that the baseline algos from Ray et al. 2019 fail on the Meta-RL case.

## Meta-RL algos
1. Use a lagrangian based approach for the constrained optimization (perhaps even copy paste from safe-agents)
2. Implement PEARL as a method for task inference.
3. Implement the model-based MAML CEM-MPC algorithm
4. Migrate LAMBDA.






PILLARS, GREMLINS, HAZARDS, VASES
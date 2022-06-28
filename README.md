# learn2learn-safely
A Safety-Gym based benchmark suite for safe meta reinforcement learning

Separate into three categories: MDP variations due to reward and CMDP variations due to costs, MDP variations on the actions (e.g., disable doggo's leg)

# Ideas
1. Generalize across different dynamics
2. Generalize across different tasks (but same robot.)
4. Action noise scale is resampled (different scale for each dimension of actions).
5. Action scale is resampled (different scale for each dimension of actions).
6. ~~Take one of doggo legs off.~~ (Look for 4. and 5.)
12. Mix the obstacles types but keep the task.
13. Mix tasks keep the types.
14. How parametric and non-parametric transfer can be tested together in a more intelligent way than metaworlds.
15. Cross validate tasks: in each experiment make a different mixture of training, testing splits.

# Possible tasks

offsamples = 0 (to speed rendering?)

Package drop collection.

Wall in between?

Coverage problems.

Box with hole. Balls falling from the air, collect as many balls as possible.

Press a button that removes half of the obstacles.

## Transfer (change objects and rewards, keep dynamics)
1. Push box to location.
2. Pull instead of push.
3. Press buttons
4. Go to goal
7. Change sizes of goals/box/buttons

## All tasks
11. Follow the leader but don't hit it (Generate random circular splines: https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.CubicSpline.html, https://stackoverflow.com/questions/64796809/how-to-avoid-spline-overlap-with-random-blobby-circle)/Catch a moving goal (instead of follow the leader?)
12. Go to moving goal.

# Benchmark
## Non Meta-RL algos
Make sure that (e.g.) PPO can solve each task independantly -- this would be the normalization factor as in Ray et al. 2019
Afterwards, make sure that the baseline algos from Ray et al. 2019 fail on the Meta-RL case.

## Meta-RL algos
1. Use a lagrangian based approach for the constrained optimization (perhaps even copy and paste from safe-agents)
2. Implement PEARL as a method for task inference.
3. Implement the model-based MAML CEM-MPC algorithm
4. Migrate LAMBDA.






PILLARS, GREMLINS, HAZARDS, VASES
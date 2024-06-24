# RL with PPO on the Cartpole environment

This github repo aims to document my progress learning proximal policy optimization. Thus far, I have attempted to solve PPO on the cartpole environment, using this page as a source of inspiration and guidance: https://medium.com/deeplearningmadeeasy/simple-ppo-implementation-e398ca0f2e7c

## Very initial attempt:

https://github.com/rsrinivasan1/cartpole/assets/52140136/d8290b21-8e20-4679-a09f-d336e929456f

This project is still a work in progress for me, and I will be gradually updating it with my results. In my initial attempts at solving the problem, I'm noticing a couple details:
- With a smaller network, the algorithm loves to tilt the pole slightly in one direction and move continuously in that direction. This results in a mediocre but good enough reward for the model, and it struggles to overcome this plateau
- I think the termination step of each episode is actually very important for the model to learn. Getting a singular reward with no predicted further rewards (because the episode has ended) influences the model to make changes and prevent the episode from ending

I'm planning on continuing this problem on this environment until the model is consistently able to score 195 over 100 consecutive episodes, then move on to bigger and better things!

# RL with PPO on the Cartpole environment

This github repo aims to document my progress learning proximal policy optimization. Thus far, I have attempted to solve PPO on the cartpole environment, using this page as a source of inspiration and guidance: https://medium.com/deeplearningmadeeasy/simple-ppo-implementation-e398ca0f2e7c

## Very initial attempt with smaller network:

https://github.com/rsrinivasan1/cartpole/assets/52140136/d8290b21-8e20-4679-a09f-d336e929456f

## Second attempt with larger network:

https://github.com/rsrinivasan1/cartpole/assets/52140136/8223d410-99c7-4d61-98c8-b7a4951599b5

This project is still a work in progress for me, and I will be gradually updating it with my results. In my initial attempts at solving the problem, I'm noticing a couple details:
- With a smaller network (1st attempt), the algorithm loves to tilt the pole slightly in one direction and move continuously in that direction. This results in a mediocre but good enough reward for the model, and it struggles to overcome this plateau
- With a larger network (2nd attempt) and the LeakyReLU activation function, the model actually learns to balance, yet the cart tends to still move gradually in one direction. I think modifying the reward function to incentivize the cart to stay near the center might be important - I'll try this next!
- I think the termination step of each episode is actually very important for the model to learn. Getting a singular reward with no predicted further rewards (because the episode has ended) influences the model to make changes and prevent the episode from ending

I'm planning on continuing this problem on this environment until the model is consistently able to score 195 over 100 consecutive episodes, then move on to bigger and better things!

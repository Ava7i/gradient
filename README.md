
# Gradient descent using JAX 
| Version Info | [![Python](https://img.shields.io/badge/python-v3.9.0-green)](https://www.python.org/downloads/release/python-3900/) [![Platform](https://img.shields.io/badge/Platforms-Ubuntu%2022.04.4%20LTS%2C%20win--64-orange)](https://releases.ubuntu.com/22.04/) [![anaconda](https://img.shields.io/badge/anaconda-v22.9.0-blue)](https://anaconda.org/anaconda/plotly/files?version=22.9.0) |
| ------------ | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |



###  Gradient Descent 


Consider you're a hiker and you climed top of the mountain but suddenly you attacked by birds or you facing some difficult situation. Then what should you do? First thing come to your mind how can you go bottom in a optimal way.So you would  start at the top of the mountain and takes steps in the direction of the steepest decline. This is similar to gradient descent, where the cost function is the height of the mountain and the parameters are the position of the hiker. By repeatedly taking steps in the direction of the steepest decline, the hiker eventually reaches the bottom of the mountain. Similarly, gradient descent adjusts the parameters of a model until the cost function reaches its minimum value. 

### In a nutshell
Gradient descent is an optimization algorithm used to minimize a cost function in machine learning and deep learning. It works by iteratively adjusting the parameters of a model in the direction of the steepest decrease in the cost function.In this repository, I build a gradient descent using Google Jax that accelerate my code.Without jax code also available in grad.py file. The main purpose of this repo is give a berief description about gradient descent.

 ### Some interesting fact about Gradient Descent
 

### 1.
Gradient descent was originally developed for solving linear regression problems, but it is now widely used in deep learning and artificial neural networks.
### 2. 
Gradient descent is not the only optimization algorithm: Although gradient descent is widely used, there are other optimization algorithms that may be better suited to specific problem types, such as conjugate gradient, BFGS, and L-BFGS.

### 3. 
There are multiple variations of gradient descent, including batch gradient descent, mini-batch gradient descent, and stochastic gradient descent. Each of these variations has its own advantages and disadvantages and is used in different types of problems.
### 4.
Gradient descent has different variants: There are several variants of gradient descent, including batch gradient descent, stochastic gradient descent, and mini-batch gradient descent, each of which have their own strengths and weaknesses.

### 5.
One of the main advantages of gradient descent is that it is relatively easy to implement and is computationally efficient.

### 6.
Gradient descent is sensitive to the choice of learning rate, which determines the step size of each iteration. A too large learning rate can cause the algorithm to converge slowly or not at all, while a too small learning rate can result in a slow convergence.Too high of a learning rate may result in oscillation or divergence, while too low of a learning rate may result in slow convergence.

### 7.

Gradient descent can be used for non-differentiable functions: Although gradient descent is typically used with differentiable functions, it can also be used with non-differentiable functions by using subgradients, which allow for a generalization of gradient descent to non-differentiable functions.
### 8.
The convergence of gradient descent can be improved by using optimization techniques such as momentum, adaptive learning rate, and regularization.
### 9.
Gradient descent is not guaranteed to find the global minimum of the cost function and may get stuck in local minima. This can be addressed by using more advanced optimization algorithms, such as second-order methods or genetic algorithms.
### 10.
Gradient descent can be parallelized to run on multiple processors or GPUs, which can greatly speed up the optimization process.


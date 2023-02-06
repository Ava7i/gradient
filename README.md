
# Gradient descent using JAX 
| Version Info | [![Python](https://img.shields.io/badge/python-v3.9.0-green)](https://www.python.org/downloads/release/python-3900/) [![Platform](https://img.shields.io/badge/Platforms-Ubuntu%2022.04.4%20LTS%2C%20win--64-orange)](https://releases.ubuntu.com/22.04/) [![anaconda](https://img.shields.io/badge/anaconda-v22.9.0-blue)](https://anaconda.org/anaconda/plotly/files?version=22.9.0) |
| ------------ | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |



## What is Gradient descent?

Let's say there is a  chef  who is trying to make the tastiest pizza. Every time the chef takes a bite of the pizza, he assesses the flavor and adjusts the ingredients accordingly. This is similar to how gradient descent adjusts the parameters of a machine learning model to minimize the cost function.

So, the chef keeps repeating this process until he finally finds the perfect recipe, just as gradient descent repeats the process of adjusting the parameters until the cost function reaches its minimum.

In this analogy, the size of the ingredient adjustments can be thought of as the learning rate in gradient descent, as it determines how much the chef (or algorithm) changes the ingredients in each iteration. Just like a chef needs to find the right balance between making big changes for bold flavors and making smaller changes for more subtle adjustments, the learning rate in gradient descent needs to be set just right to find the minimum of the cost function in an efficient manner.

So, whether you're a chef whipping up a tasty dish or a machine learning algorithm trying to minimize a cost function, the key to success is finding the right ingredients at the right time!

![](https://github.com/Ava7i/gradient/blob/master/gradient_descent_line_graph.gif)


## In a nutshell
Gradient descent is an optimization algorithm used to minimize a cost function in machine learning and deep learning. It works by iteratively adjusting the parameters of a model in the direction of the steepest decrease in the cost function.In this repository, I build a gradient descent using Google JAX that accelerate my code.Without JAX, code also available in grad.py file. The main purpose of this repo is give a berief description about gradient descent using JAX.

## What is JAX?

JAX is [Autograd](https://github.com/hips/autograd) and [XLA](https://www.tensorflow.org/xla),
brought together for high-performance machine learning research. JAX is a library for machine learning research that provides functionality for automatic differentiation, a process that allows gradient computation with respect to inputs of a function. Here I used JAX because-
####  Speed:
JAX uses just-in-time (JIT) compilation to speed up computation, making it particularly well-suited for large-scale numerical computations. JAX can also transparently run computations on GPUs for even faster performance.
####  Flexibility: 
JAX allows you to mix and match different numerical backends (such as NumPy and TensorFlow), making it easy to switch between CPU and GPU computations, or to switch between different deep learning frameworks.
####  Performance: 
JAX NumPy can be faster than NumPy, particularly when working with GPU acceleration, as JAX can transparently compile and run computations on GPUs.



 ## Some interesting fact about Gradient Descent
 

 1. Gradient descent was originally developed for solving linear regression problems, but it is now widely used in deep learning and artificial neural networks.
 2. Gradient descent is not the only optimization algorithm: Although gradient descent is widely used, there are other optimization algorithms that may be better suited to specific problem types, such as conjugate gradient, BFGS, and L-BFGS.
 3. There are multiple variations of gradient descent, including batch gradient descent, mini-batch gradient descent, and stochastic gradient descent. Each of these variations has its own advantages and disadvantages and is used in different types of problems.
 4. Gradient descent has different variants: There are several variants of gradient descent, including batch gradient descent, stochastic gradient descent, and mini-batch gradient descent, each of which have their own strengths and weaknesses.
 5. One of the main advantages of gradient descent is that it is relatively easy to implement and is computationally efficient.
 6. Gradient descent is sensitive to the choice of learning rate, which determines the step size of each iteration. A too large learning rate can cause the algorithm to converge slowly or not at all, while a too small learning rate can result in a slow convergence.Too high of a learning rate may result in oscillation or divergence, while too low of a learning rate may result in slow convergence.
 7. Gradient descent can be used for non-differentiable functions: Although gradient descent is typically used with differentiable functions, it can also be used with non-differentiable functions by using subgradients, which allow for a generalization of gradient descent to non-differentiable functions.
 8. The convergence of gradient descent can be improved by using optimization techniques such as momentum, adaptive learning rate, and regularization.
 9. Gradient descent is not guaranteed to find the global minimum of the cost function and may get stuck in local minima. This can be addressed by using more advanced optimization algorithms, such as second-order methods or genetic algorithms.
 10. Gradient descent can be parallelized to run on multiple processors or GPUs, which can greatly speed up the optimization process.


## Quickstart:
Clone the repository and you have to first install the JAX.

## Installation

JAX is written in pure Python, but it depends on XLA, which needs to be
installed as the `jaxlib` package. Use the following instructions to install a
binary package with `pip` or `conda`, or to [build JAX from
source](https://jax.readthedocs.io/en/latest/developer.html#building-from-source).

JAX support installing or building `jaxlib` on Linux (Ubuntu 16.04 or later) and
macOS (10.12 or later) platforms.

### pip installation on CPU

To install a CPU-only version of JAX, which might be useful for doing local
development on a laptop, you can run

```bash
pip install --upgrade pip
pip install --upgrade "jax[cpu]"
```
**These `pip` installations do not work with Windows, and may fail silently;**

### pip installation: GPU (CUDA)

If you want to install JAX with both CPU and NVidia GPU support, you must first
install [CUDA](https://developer.nvidia.com/cuda-downloads) and
[CuDNN](https://developer.nvidia.com/CUDNN),
if they have not already been installed. Unlike some other popular deep
learning systems, JAX does not bundle CUDA or CuDNN as part of the `pip`
package.

JAX provides pre-built CUDA-compatible wheels for **Linux only**,
with CUDA 11.4 or newer, and CuDNN 8.2 or newer. Note these existing wheels are currently for `x86_64` architectures only. Other combinations of
operating system, CUDA, and CuDNN are possible, but require [building from
source](https://jax.readthedocs.io/en/latest/developer.html#building-from-source).

* CUDA 11.4 or newer is *required*.
  * Your CUDA installation must be new enough to support your GPU. If you have
    an Ada Lovelace (e.g., RTX 4080) or Hopper (e.g., H100) GPU,
    you must use CUDA 11.8 or newer.
* The supported cuDNN versions for the prebuilt wheels are:
  * cuDNN 8.6 or newer. We recommend using the cuDNN 8.6 wheel if your cuDNN
    installation is new enough, since it supports additional functionality.
  * cuDNN 8.2 or newer.
* You *must* use an NVidia driver version that is at least as new as your
  [CUDA toolkit's corresponding driver version](https://docs.nvidia.com/cuda/cuda-toolkit-release-notes/index.html#cuda-major-component-versions__table-cuda-toolkit-driver-versions).
  For example, if you have CUDA 11.4 update 4 installed, you must use NVidia
  driver 470.82.01 or newer if on Linux. This is a strict requirement that
  exists because JAX relies on JIT-compiling code; older drivers may lead to
  failures.
  * If you need to use an newer CUDA toolkit with an older driver, for example
    on a cluster where you cannot update the NVidia driver easily, you may be
    able to use the
    [CUDA forward compatibility packages](https://docs.nvidia.com/deploy/cuda-compatibility/)
    that NVidia provides for this purpose.


Next, run

```bash
pip install --upgrade pip
# Installs the wheel compatible with CUDA 11 and cuDNN 8.6 or newer.
# Note: wheels only available on linux.
pip install --upgrade "jax[cuda]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
```

### pip installation: Google Cloud TPU
JAX also provides pre-built wheels for
[Google Cloud TPU](https://cloud.google.com/tpu/docs/users-guide-tpu-vm).
To install JAX along with appropriate versions of `jaxlib` and `libtpu`, you can run
the following in your cloud TPU VM:
```bash
pip install --upgrade pip
pip install "jax[tpu]>=0.2.16" -f https://storage.googleapis.com/jax-releases/libtpu_releases.html
```

### pip installation: Colab TPU
Colab TPU runtimes come with JAX pre-installed, but before importing JAX you must run the following code to initialize the TPU:
```python
import jax.tools.colab_tpu
jax.tools.colab_tpu.setup_tpu()
```
Colab TPU runtimes use an older TPU architecture than Cloud TPU VMs, so installing `jax[tpu]` should be avoided on Colab.
If for any reason you would like to update the jax & jaxlib libraries on a Colab TPU runtime, follow the CPU instructions above (i.e. install `jax[cpu]`).

### Conda installation

There is a community-supported Conda build of `jax`. To install using `conda`,
simply run

```bash
conda install jax -c conda-forge
```

### Building JAX from source
See [Building JAX from
source](https://jax.readthedocs.io/en/latest/developer.html#building-from-source).

Then install numpy and run the py files. 
If you want to run the grad.py file without JAX simply run the py file. you don't need to install JAX.
For JAX you need to install the JAX then run the py file. You can easily store your data on text file.

```
python -u grad_jax.py > results.txt

```
### Usage

Import jax.numpy as jnp. JAX NumPy is a version of NumPy that is optimized for use with the JAX library. JAX is a library for machine learning research that provides functionality for automatic differentiation, a process that allows gradient computation with respect to inputs of a function. JAX NumPy provides an interface similar to NumPy but with JAX's differentiation capabilities integrated.
```
import jax.numpy as jnp
from jax import random
```
#### Initialise some parameters
```
key = random.PRNGKey(0)
x = random.normal(key, (10,))
#x = jnp.random.randn(10,1)
y = 5*x + random.normal(key, ())
# Parameters
w = 0.0 
b = 0.0 
# Hyperparameter 
learning_rate = 0.001

```

## Documentation 

For details about the JAX, see the
[reference documentation](https://jax.readthedocs.io/en/latest/notebooks/quickstart.html/).


For details about the JAX, see the
[reference documentation](https://github.com/google/jax).


For details about the Gradient Descent, see the
[paper](https://arxiv.org/abs/1609.04747).



More details read this [Link](https://kevinbinz.com/2019/05/26/intro-gradient-descent/)

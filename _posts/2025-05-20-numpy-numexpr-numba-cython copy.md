---
layout: post
title: Numpy or ... ? What can I choose?
date: 2025-05-20 15:40:16
description: Numpy optimizations
tags: Inference Optimizations Performance ML
categories: Optimizations
thumbnail: assets/post_images/numpy/numpy_go_brr.png
giscus_comments: true
toc: 
  beginning: true
---

### Speeding Up Complex Math in Python: NumPy, Numba, Cython, and NumExpr Compared

If you’ve ever worked with big arrays in Python, you know how important it is to make your code run fast—especially when you’re crunching numbers with complex math. Today, I’m going to walk you through a little experiment where I tried out several popular tools to see which one handles the expression the fastest. Let’s dive in!

As an example we will use this easy to play with expression:

$$
f(x) = e^{\sin(x) + \cos(x)}
$$

### 1. The Baseline: Pure Numpy

```python
import numpy as np

def complex_function(x: np.ndarray):
    return np.exp(np.sin(x) + np.cos(x))

```

How did it do?

Processing an array of 1 million elements took about 22 miliseconds. Not bad, but let’s see if we can do better!

### 2. Trying out np.vectorize

```python
import math
vec_f = np.vectorize(lambda x: math.exp(math.sin(x) + math.cos(x)))
```

The result?

Surprisingly, it was actually slower than pure NumPy! (223 ms)

Why? Because `np.vectorize` is really just a convenient wrapper for a Python loop—not a true speed booster. Lesson learned: Stick with NumPy’s built-in vectorized functions for performance.

### 3. Supercharging with Numba

```python
from numba import vectorize, float64
@vectorize([float64(float64)])
def numba_func(x: float):
    """
    A Numba-optimized function that computes the sine and cosine of an element.

    Parameters:
    x (float): Input value.

    Returns:
    float: Exponential of the sine and cosine of the input value.
    """
    result = np.exp(np.sin(x) + np.cos(x))  # Combine sine and cosine functions
    return result
```

How fast? 16ms. About 27% faster than pure NumPy! That’s a nice bump in speed for almost no extra effort.

### 4. Going deeper with Cython

```python
import numpy as np
cimport cython

from libc.math cimport sin, cos, exp
from cython.parallel import prange

cimport numpy as cnp

cnp.import_array()
DTYPE = np.float64

ctypedef cnp.float64_t DTYPE_t

@cython.boundscheck(False)
@cython.wraparound(False)
def cython_complex_function(cnp.ndarray[DTYPE_t, ndim=1] f):
    """
    A Cythonized function that computes the sine and cosine of an array.
    This version is parallelized using OpenMP.

    Parameters:
    f (np.ndarray): Input array of floats.

    Returns:
    np.ndarray: Resulting array after applying the operations.
    """
    cdef double s
    cdef double c
    cdef int i
    cdef int n = f.shape[0]
    # Allocate the array directly with cnp.ndarray
    cdef cnp.ndarray[DTYPE_t, ndim=1] arr = cnp.ndarray(shape=(n,), dtype=DTYPE, order='C')

    # Use prange for parallel execution with OpenMP
    for i in prange(n, nogil=True):
        s, c = sin(f[i]), cos(f[i])
        arr[i] = exp(s + c)

    return arr
```

Performance: About the same as Numba. 

But you do need to write some extra code and deal with compiling and its bit more cumbersome.

> *Note*
> 
> We could further optimize the Cython version by doing more low level tricks but we chose to not apply more effort for this experiment

### 5. The Surprise Winner: NumExpr

Finally, I tried out NumExpr, which is designed to optimize and parallelize mathematical expressions:

```python
import numexpr as ne

ne.evaluate("exp(sin(x) + cos(x))", local_dict={"x": x})
```
Its extremely surprising how easy was to code this one and the result is even more impressive. Blazing fast with only 2ms.

### So, What’s the Best Choice?
- NumExpr is the clear winner for this kind of math-heavy, element-wise calculation. It’s super easy to use and incredibly fast.

- Numba and Cython are great if you want more control or are already using them in your project.

And remember: `np.vectorize` might look handy, but it won’t speed up your code.

### Checkout the full notebook below

{::nomarkdown}
{% assign jupyter_path = 'assets/jupyter/numpy_go_brr.ipynb' | relative_url %}
{% capture notebook_exists %}{% file_exists assets/jupyter/blog.ipynb %}{% endcapture %}
{% if notebook_exists == 'true' %}
  {% jupyter_notebook jupyter_path %}
{% else %}
  <p>Sorry, the notebook you are looking for does not exist.</p>
{% endif %}
{:/nomarkdown}

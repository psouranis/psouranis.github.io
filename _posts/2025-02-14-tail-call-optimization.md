---
layout: post
title: Tail Call Optimization
date: 2025-02-14 12:40:16
description: Tail Call Optimization
tags: C Optimizations Performance
categories: Optimizations
thumbnail: assets/post_images/tco_complexity.png
giscus_comments: true
toc: 
  beginning: true
---

### Tail Recursive Methods

In this post we will talk about optimizations in C and specifically about Tail Call Optimization. But let’s start from defining what is a tail recursion.

A recursive method is referred to as tail-recursive *when the recursive call is the last thing executed by the that method*, otherwise, it’s known as *head recursive*

Here is a simple example of a *tail recursive method*:

```c
void printNumbers(int n) {
	if (n <= 0) {
		return;
	}	
	printf("%d ", n);
	printNumbers(n-1);
}
```

Notice in this function how the last thing executed by the method is a recrusive call, making it a *tail recursive method*. Let’s take another example now, the factorial function. The factorial function can be implemented as follows:

```c
int factorial(int n) {
	if (n==1) {
		return 1;
	}
	return n * factorial(n-1);
}
```

Is this a tail recursive method? On the first look someone would say yes but if we were to look more closely we would see that the function can be written as follows:

```c
int factorial(int n) {
	if (n==1) {
		return 1;
	}
	int x = factorial(n-1);
	return n * x;
}
```

We realize that the last executed statement by this method is the multiplication and not the recursion meaning that the method we previously implemented is not *tail recursive* but rather *head recursive*. So let’s try first to transform it to a tail recursive method and then discuss it’s benefits.

```c
int factorial(int n) {
	return factorialTail(n, 1);
}

int factorialTail(int n, int x) {
	if (n == 1) {
		return x;
	}
	return factorial(n - 1, n * x);
}
```

Notice that now we have 2 methods instead of 1 (`factorial` and `factorialTail`). The original method is turned to a helper method and is now called with an initial value. This initial value, in most cases, will be the return value of the stop condition we had in the original recursive call. 

The next thing you will notice, *is how we included the multiplication* which was the operation executed after the recursive call, inside the the recursion itself as an **accumulator** and it is the final result of this **accumulator** that is being returned to the user when the calculation is finished.

In a tail recusrive method, *the result of the stop condition is actually the result of the whole recursion* because that is what it will be returned by all child calls all the way to the parent and initial call. 

Let’s take another example as well, the Fibonnaci sequence:

$$
\text{Fibonacci}(n) = \text{Fibonnaci}(n-1) + \text{Fibonnaci}(n-2)
$$

```c
int fibonnaci(int n) {
	if (n==0) return 0;
	if (n==1) return 1;
	return fibonnaci(n-1) + fibonnaci(n-2);
}
```

Can be transformed to a tail recursive method as follows:

```c
int fibonnaci(int n) {
	return fibonnaciTail(n, 0, 1);
}

int fibonnaciTail(int n, int a, int b) {
	if (n==0) return a;
	if (n==1) return b;
	return fibonnaci_(n-1, b, a+b);
}
```

Similar to the factorial one, the original method is broken into two, the caller and the helper one. The caller method is using as initial values the return values of the stop-conditions we previously had. The result or the addition is now accumulated inside the recursive call and because the recursion is called on the previous value of our current value as well in the initial method – we need to keep track the previous value of the initial call using an additional parameter.

### Continuation Passing Style

1. The continuation passing style (passing accumulators as parameter values) might be the **only generic way** that allows you to transform the method into a form that uses **only-tail calls**.

2. Not all logics can be made tail-recursive, non-linear recursions for instance maintain a variable somewhere and that somewhere is the **Stack Memory**.

### Stack Visualization

Take a look at how the stack frame with tail call optimization and without.

<div style="text-align: center;">
  <img src="/assets/post_images/tco.png">
</div>


When `factorial(5)` let’s say is called without tail call optimization — these calls will be respectively added to the stack, and we won’t be able to start popping from this stack before we reach the stop condition of the recursion. 

When that happens the values we have will start to get replaced and the stack calls will be popped. 

$$ \text{factorial}(5) \rightarrow \text{factorial}(4) \rightarrow \text{factorial}(3) \rightarrow \text{factorial}(2) \rightarrow \text{factorial}(1)
$$

However, if we try this on the tail recursive method, the first call will be the call to the factorial method `factorial` and the second call will be the call to the `factorialTail`:

$$\text{factorial}(5) \rightarrow \text{factorialTail}(5,1)$$

When `factorial(4)` is called, because no variables are required to be stored in the frame left by the parent call of the same recursive method, and because both of these calls will have the same return value — then this same frame will be used to store the new call details:

$$\text{factorial}(5) \rightarrow \text{factorialTail}(1, 120)$$

Now, if we go back and compare the stacks corresponding to the recursive methods at their maximum depths, we will see that the space complexity of recursion was reduced from $$O(n)$$ to $$O(1)$$ thanks to tail-call elimination.

<div style="text-align: center;">
  <img src="/assets/post_images/tco_complexity.png" style="width: 40%;">
</div>

### Summary

#### What are the benefits ot Tail Recursion Methods?

- Tail recursive methods are **optimized** by certain compilers. These compilers usually execute calls with the help of a stack.
- In case of tail-recursive methods, there is no need to reserve a place in the stack for the recursive call because there is nothing left to do in the current method and we won’t be returning to the parent call.
- The space complexity reduces from $$O(n)$$ to $$O(1).$$

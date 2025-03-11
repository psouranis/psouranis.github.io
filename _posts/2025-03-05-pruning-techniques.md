---
layout: post
title: Pruning techniques - Optimizing machine learning models Part 1
date: 2025-03-04 12:40:16
description: Pruning Techniques
tags: Inference Optimizations Performance ML
categories: Optimizations
thumbnail: assets/post_images/pruned_tree.png
giscus_comments: true
toc: 
  beginning: true
---

In this article, we are going to focus in optimizing machine learning models after training ends. 

### Pruning

<div style="text-align: center;">
  <img src="/assets/post_images/pruned_tree.png">
</div>

Pruning is the practice of removing parameters (which may entail removing individual parameters, or parameters in groups such as by neurons) from an existing artificial neural networks.[1] The goal of this process is to maintain accuracy of the network while increasing its efficiency. This can be done to reduce the computational resources required to run the neural network [[1]](https://en.wikipedia.org/wiki/Pruning_(artificial_neural_network)). 

<div style="text-align: center;">
  <img src="/assets/post_images/pruned_network.png">
</div>

There are 2 distinctions in pruning: **Global** and **Local** pruning.

#### Global vs Local pruning

The difference between global and local pruning lies in whether structures are removed from a subset or all available structures of a network. 
A major limitation of local pruning is that setting a pre-defined prune ratio for each layer can be complex and lead to sub-optimal sparsity. 
To simplify, local pruning often uses a consistent prune ratio across layers. In contrast, global pruning automatically generates a varying prune ratio for each layer. 
However, global pruning poses great challenges (particularly for LLMs), due to significant variations in layer magnitudes. 
For instance, some outlier features may have magnitudes up to 20 times larger than others, leading to incomparability issues.

In the following post, we are going to focus on **Local** pruning since it poses an advantage compared to global pruning.


Let's review some of the most common techniques for pruning:

#### Random Unstructured

As the name suggest, in random unstructured pruning we prune each tensor of our network by randomly removing weights (connections between neurons) of its layers.
In PyTorch the most common way to apply random unstructured pruning is via the `torch.nn.utils.random_unstrucuted`.

We can apply it to each layer of our network like this:

```python
from torch.nn.utils import random_unstructured, remove

model = OurAwesomeNetwork() # nn.Module()
pruning_ratio = 0.5 # Remove half of neurons in every tensor

for name, module in model.named_modules():
	random_unstructured(module, name="weight", amount=pruning_ratio)
	remove(module, name="weight") # The parameter named name+'_orig' is removed from the parameter list.
```

In the script above there are couple of things happening: 

First, we iterate through each layer of our network using the `.named_modules()` method. Then we pass to the `random_unstructured` function the `module`, the `name` and the `amount` parameters.
We do that because we want to target the `weight` attribute of our `Module` which contains the weights (it could also be applied to `bias` as well).

Finally, we apply the `remove` method to the `module` because the `random_unstructured` method applies pruning to each tensor but also keeps the original weights in the attribute `weight_orig`. If we want
to make the pruning permanent and discard the original weights, we have to remove them by using the `remove` function as we did above.

> 💡 **Important note**
> <br><br>
> When we apply pruning through this method, essentially what we are doing is applying a random mask of zeroes and ones on every layer in order to cancel them out. You can actually access those masks via 
> the `weight_mask` attribute. 
> <br><br>
> It is important to make this clear since masking a layer does nothing to the network parameters and hence its size or to how many operations does.
> <br><br>
> The total number of parameters remains the same (it actually increases because we have to also store the masks along with the actual weights) but we will see later how we can actually prune the layer and 
> indeed remove parameters from it.

- *Pros of Random Unstructured Pruning* 
  - Simple to implement.
  - Extremely flexible, as it can prune any element in any dimension.
  - Generally has better performance in terms of accuracy.
<br><br>
- *Cons of Random Unstructured Pruning* 
  - Pruning is not guaranteed to be uniform across the tensor.
  - Pruning is not guaranteed to be uniform across the dimensions.
  - Very hard to accelerate with hardware acceleration.

#### Random Structured

Structured pruning is very similar with unstructured pruning only in this case we don't only remove weights (like previously) but we remove entire neurons (or entire structural components like
filters, channels in Convolutional networks). This type of pruning tends to reduce accuracy of the model but it has the benefit that it achieves better performance in terms of inference speed.
In this case by cutting entire rows or columns of our tensors we dramatically reduce the number of calculations that need to be executed.

<div style="text-align: center;">
  <img src="/assets/post_images/structured_2d_pruning.png" style="max-width: 50%">
</div>

In PyTorch to apply random structured pruning, we can do so via the `torch.nn.utils.random_structured`.

```python
from torch.nn.utils import random_structured, remove

model = OurAwesomeNetwork() # nn.Module()
pruning_ratio = 0.5 # Remove half of neurons in every tensor

for name, module in model.named_modules():
	random_structured(module, name="weight", amount=pruning_ratio)
	remove(module, name="weight") # The parameter named name+'_orig' is removed from the parameter list.
```

- *Pros of Random Structured Pruning* 
  - Simple to implement.
  - Can be used to prune any dimension of a tensor.
  - Easier to accelerate (just a smaller matrix)
  - Generally it offers lower accuracy but better performance in terms of inference speed.
<br><br>
- *Cons of Random Sstructured Pruning* 
  - Less flexible as it prunes entire rows or columns of tensors
  - Pruning is not guaranteed to be uniform across the tensor.

#### Ln Unstructured

In $$\ell_{n}$$ unstructured, we prune each tensor of our network by removing the lowest weights according to the $$\ell_{n}$$ norm. 
Let's see the following example using the $$\ell_1$$ norm to calculate the importance of the tensor weights:

{::nomarkdown}
{% assign jupyter_path = 'assets/jupyter/l1_unstructured.ipynb' | relative_url %}
{% capture notebook_exists %}{% file_exists assets/jupyter/blog.ipynb %}{% endcapture %}
{% if notebook_exists == 'true' %}
  {% jupyter_notebook jupyter_path %}
{% else %}
  <p>Sorry, the notebook you are looking for does not exist.</p>
{% endif %}
{:/nomarkdown}

We start with a `torch.Tensor` that has no pruned weights. We then apply `l1_unstructured` pruning to prune the lowest weights.
To do so we first calculate the $$\ell_{1}$$ norm of the tensor:

$$
\ell_{1}(t) = \sum_{i=1}^{n} |w_{i}|
$$

Where $$n$$ is the total number of elements in the Tensor. Then we remove the $$40\%$$ of the lowest weights of our Tensor.

Similar with random unstructured pruning, $$\ell_n$$ unstructured purning shares the same pros and cons.

- *Pros of Ln Unstructured Pruning* 
  - Simple to implement.
  - Extremely flexible, as it can prune any element in any dimension.
  - Generally has better performance in terms of accuracy compared to Ln Structured Pruning.
  - Focuses more in the importance of weights compared to Random Unstructured and hence can result in higher performance compared to Random Unstructured with higher compression ratio.
<br><br>
- *Cons of Ln Unstructured Pruning* 
  - Pruning is not guaranteed to be uniform across the dimensions.
  - Very hard to accelerate with hardware acceleration.
  - As the pruning ratio increases, it can result in suboptimal networks since it focuses more on the weights with lower importance and the distribution of weights will change dramatically.

#### Ln Structured

Finally in $$\ell_{n}$$ structured, we prune across the dimensions of our tensors, dropping weights with the lowest $$\ell_{n}$$ norm.
Like previously, we will use the $$\ell_1$$ norm in order to calculate the importance. 

It is interesting to see what happens when we prune across the dimensions of a weight matrix. Let's take as an example a Linear layer and see what happens for `dim=0`. 

{::nomarkdown}
{% assign jupyter_path = 'assets/jupyter/l1_structured_dim0.ipynb' | relative_url %}
{% capture notebook_exists %}{% file_exists assets/jupyter/blog.ipynb %}{% endcapture %}
{% if notebook_exists == 'true' %}
  {% jupyter_notebook jupyter_path %}
{% else %}
  <p>Sorry, the notebook you are looking for does not exist.</p>
{% endif %}
{:/nomarkdown}

Now let's take the case where `dim=1`.

{::nomarkdown}
{% assign jupyter_path = 'assets/jupyter/l1_structured_dim1.ipynb' | relative_url %}
{% capture notebook_exists %}{% file_exists assets/jupyter/blog.ipynb %}{% endcapture %}
{% if notebook_exists == 'true' %}
  {% jupyter_notebook jupyter_path %}
{% else %}
  <p>Sorry, the notebook you are looking for does not exist.</p>
{% endif %}
{:/nomarkdown}

Suppose that we have a N-layer network. We can see that in the first case where `dim=0` for each layer $$L$$ we prune the neurons of the $$L$$ layer, whereas if `dim=1` 
we prune the neurons of the $$L+1$$ layer.


- *Pros of Ln Structured Pruning* 
  - Simple to implement.
  - Can be used to prune any dimension of a tensor.
  - Easier to accelerate (just a smaller matrix).
  - It can provide easy quick wins to get a compressed model.
  - Provides very high compression ratio.
<br><br>
- *Cons of Ln Structured Pruning* 
  - Less flexible as it prunes entire rows or columns of tensors.
  - Pruning is not guaranteed to be uniform across the tensor.
  - It is more aggressive than Ln Unstructured and should be not used with high pruning ratios.


In the rest of the post, we will explore each of those techniques on the ImageNet validation dataset (50000) by using as a base model the Vision Transformer model that you can find [[here]](https://huggingface.co/google/vit-base-patch16-224)

#### Results in ImageNet1000 with VIT

After all is said and done, its now time to evaluate the different methods described above in the ImageNet dataset using as our base model the VisionTransformer.
You can find the code that was used to run those experiments here: [link](https://github.com/psouranis/pruning). 

We pruned each layer of the network equally (with the same pruning ratio in each iteration) except the `LayerNorm` layers.

<div style="display: grid; grid-template-columns: repeat(2, 1fr); grid-template-rows: repeat(2, 1fr); gap: 10px; width: fit-content;">
  <img src="/assets/post_images/imagenet1000/pruning_results_random_unstructured.png" alt="Image 1" style="width: 100%; height: auto;">
  <img src="/assets/post_images/imagenet1000/pruning_results_random_structured.png" alt="Image 1" style="width: 100%; height: auto;">
  <img src="/assets/post_images/imagenet1000/pruning_results_l1_unstructured.png" alt="Image 1" style="width: 100%; height: auto;">
  <img src="/assets/post_images/imagenet1000/pruning_results_ln_structured.png" alt="Image 1" style="width: 100%; height: auto;">
</div>

As we can see from the results above, our best compression-ratio / accuracy comes from the `l1_unstructured` method which is somewhat expected since it removes
weights with the lowest importance. As we also mentioned above, `ln_structured` is extremely aggressive and immediatelly drives our accuracy close to 20% which is not acceptable.

Finally we can try to recover some of our lost accuracy by finetuning our model. Let's see if we can recover the accuracy for the case where pruning ratio is 0.5 and 
for `l1_unstructured` pruning method.

#### Finetuning with pruning 

In our experiments, we finetuned the pruned model in order to achieve some of the lost accuracy (around 71%). 
So, we managed to reduce our model size by 50% and lose only 12.5% of its initial accuracy.







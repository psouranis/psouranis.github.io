---
layout: post
title: Pruning techniques - Optimizing machine learning models Part 1
date: 2025-03-04 12:40:16
description: Pruning Techniques
tags: Inference Optimizations Performance ML
categories: Optimizations
thumbnail: assets/post_images/tree.webp
giscus_comments: true
toc: 
  beginning: true
---

In this article, we are going to focus in optimizing machine learning models after training ends. 

### Pruning

<br>
<div style="text-align: center;">
  <img src="/assets/post_images/tree.webp" style="width: 30%; height: auto;">
</div>
<br>

Pruning is the practice of removing parameters (which may entail removing individual parameters, or parameters in groups such as by neurons) from an existing artificial neural networks.[1] The goal of this process is to maintain accuracy of the network while increasing its efficiency. This can be done to reduce the computational resources required to run the neural network [[1]](https://en.wikipedia.org/wiki/Pruning_(artificial_neural_network)). 

<div style="text-align: center;">
  <img src="/assets/post_images/pruned_network.png">
</div>

There are 2 distinctions in pruning: **Global** and **Local** pruning.

#### Global vs Local pruning

The key distinction between local and global pruning lies in their scope: local methods remove connections within individual layers, while global methods consider the entire network. A significant drawback of local pruning is the difficulty in determining an optimal, layer-specific pruning ratio, often leading to a uniform ratio across all layers for simplicity. In contrast, global pruning dynamically assigns varying prune ratios per layer. However, this approach faces considerable hurdles, particularly with Large Language Models (LLMs), due to substantial disparities in layer magnitudes. For example, some features can be up to 20 times larger than others, creating issues with direct comparison during the pruning process.

In the following post, we are going to focus on **Local** pruning since it poses an advantage compared to global pruning.

#### Structured vs Unstructured

Structured pruning removes entire groups of connected parameters like neurons or filters, resulting in a smaller, more regular network that is generally easier for hardware to process efficiently. In contrast, unstructured pruning removes individual weights, leading to a sparse network with the same overall structure but with many zeroed connections, which can achieve higher compression but may require specialized hardware to fully realize speedups.

<div style="display: grid; grid-template-columns: repeat(2, 1fr); grid-template-rows: repeat(1, 1fr); gap: 10px; width: fit-content;">
  <img src="/assets/post_images/unstructured.png" style="width: 52%; height: auto;">
  <img src="/assets/post_images/structured_2d_pruning.png" style="width: 100%; height: auto;">
</div>


Let's review some of the most common techniques for pruning:

#### Random unstructured

As the name suggests, random unstructured pruning involves randomly removing individual weights (the connections between neurons) within each layer of your network. In PyTorch, a common way to achieve this is using the `torch.nn.utils.prune.random_unstructured` function. 

You can apply it to each layer like this:

```python
from torch.nn.utils import random_unstructured, remove

model = OurAwesomeNetwork() # nn.Module()
pruning_ratio = 0.5 # Remove half of neurons in every tensor

for name, module in model.named_modules():
	random_unstructured(module, name="weight", amount=pruning_ratio)
	remove(module, name="weight") # The parameter named name+'_orig' is removed from the parameter list.
```

In the provided script, we're performing the following steps:

1. We iterate through each layer of our network using `.named_modules()` to access both the layer's name and the layer itself.
2. For each layer, we apply the `random_unstructured` pruning function. This function requires three key arguments: the module (the layer), the name of the parameter to prune (in our case, `'weight'` to target the layer's weights – you could also prune biases), and the amount of pruning.
3. Crucially, `random_unstructured` doesn't immediately delete weights. It masks them and stores the original weights in `weight_orig`. To finalize the pruning and remove the original weights, we then call the `remove()` method on the module.


> 💡 Important note
> <br><br>
> When we apply pruning using this method, we're essentially creating a binary mask (composed of zeros and ones) for each layer's weight tensor. A zero in the mask indicates a weight that should be treated as zero during calculations, effectively "canceling it out" in the forward pass. You can inspect these masks directly through the `weight_mask` attribute of the pruned layer.
> <br><br>
> It's crucial to understand that masking alone doesn't reduce the actual number of parameters or the size of the network in memory, nor does it decrease the number of computations performed. The underlying weight values are still present. In fact, the total number of parameters might even increase slightly because we now have to store these masks alongside the original weights (especially in the case where finetuning is applied).


- **Pros of Random unstructured pruning** 
  - *Ease of implementation*: This method is straightforward to implement, making it a good starting point for exploring network pruning techniques.
  - *High granularity and flexibility*: Random unstructured pruning offers the ultimate flexibility by allowing the removal of any individual weight connection within the network, regardless of its layer or position..
  - *Potential for high accuracy*: For a given level of sparsity, this method often achieves better or comparable accuracy compared to more structured pruning approaches.

- **Cons of Random unstructured pruning**
  - *Sparsity distribution challenges*: The random nature of pruning doesn't guarantee a uniform distribution of removed weights within a single layer's tensor. This can lead to some parts of the layer being more sparse than others.
  - *Dimensional imbalance*: Similarly, the sparsity level might not be consistent across different dimensions (e.g., input vs. output channels) within a layer, potentially impacting the importance of certain features.
  - *Limited hardware acceleration*: A significant drawback is that the resulting sparse network has an irregular structure, making it very difficult to exploit the efficiencies of specialized hardware accelerators designed for structured sparsity patterns.

#### Random Structured

Structured pruning differs from unstructured pruning in that instead of removing individual weights, it eliminates entire structural units within the network. This could mean removing whole neurons or, in the context of Convolutional Neural Networks (CNNs), entire filters or channels. While this approach generally leads to a greater reduction in accuracy compared to unstructured pruning, it offers a significant advantage: improved inference speed. By removing entire rows or columns of weight tensors, we drastically reduce the number of computations required. 

In PyTorch, you can perform random structured pruning using the `torch.nn.utils.prune.random_structured` function.

```python
from torch.nn.utils import random_structured, remove

model = OurAwesomeNetwork() # nn.Module()
pruning_ratio = 0.5 # Remove half of neurons in every tensor

for name, module in model.named_modules():
	random_structured(module, name="weight", amount=pruning_ratio)
	remove(module, name="weight") # The parameter named name+'_orig' is removed from the parameter list.
```

- **Pros of Random structured pruning** 
  - *Simplicity*: Randomly selecting entire rows, columns, or other structured units to prune is conceptually and often computationally simple to implement.
  - *Potential for dense substructures*: By removing entire structures, the remaining tensor might still be relatively dense, making it potentially easier to leverage existing efficient dense matrix multiplication (GEMM) implementations compared to highly sparse unstructured matrices.
  - *Parameter reduction*: It effectively reduces the number of parameters and computations required, leading to greater speedups in inference.

- **Cons of Random structured pruning**
  - *Suboptimal pruning granularity*: Pruning at the level of entire structures (rows, columns) can be less fine-grained than unstructured pruning, potentially leading to a larger drop in accuracy for the same sparsity level.
  - *Limited flexibility in sparsity distribution*: The random selection of structures to prune might not result in an optimal distribution of sparsity across the tensor, potentially leaving important weights unpruned while removing less critical ones.
  - *Accuracy sensitivity*: The random nature of the pruning can lead to variability in the final accuracy depending on which specific structures were removed.
  - *Difficulty in leveraging structure*: Even though structured pruning offers us some structure in the resulting pruned layer, its difficult to drop the dimensions entirely and requires very careful stitching.

#### L1 unstructured
Okay, continuing with the explanation of $$\ell_1$$​ unstructured pruning:

In $$\ell_1$$​ unstructured pruning, the importance of each weight in a tensor is determined by its magnitude as measured by the $$\ell_1$$​ norm. We then prune the weights with the smallest importance (lowest norm values). This method aims to remove the weights that have the least impact on the network's output.

Let's delve into an example using the $$\ell_1$$​ norm. The $$\ell_1$$​ norm of a single weight is simply its absolute value. Therefore, in $$\ell_1$$​ unstructured pruning, we would identify the weights with the smallest absolute values as the least important and remove them.

Here's how we might think about calculating the importance of tensor weights using the $$\ell_1​$$ norm:

$$
||w_{ij}||_1 = |w_{ij}|
$$

We would then calculate this value for every weight in the tensor and prune those with the smallest magnitudes.

{::nomarkdown}
{% assign jupyter_path = 'assets/jupyter/l1_unstructured.ipynb' | relative_url %}
{% capture notebook_exists %}{% file_exists assets/jupyter/blog.ipynb %}{% endcapture %}
{% if notebook_exists == 'true' %}
  {% jupyter_notebook jupyter_path %}
{% else %}
  <p>Sorry, the notebook you are looking for does not exist.</p>
{% endif %}
{:/nomarkdown}

In the example above we start with a `torch.Tensor` that has no pruned weights. We then apply `l1_unstructured` pruning to prune the weights with the lowest $$ell_1$$ norm.
To do so we first calculate the *Importance* of the tensor:

$$
\text{Importance} = |W|
$$

Then we remove the $$40\%$$ of the lowest weights of our Tensor.

Similar with random unstructured pruning, $$\ell_1$$ unstructured purning shares the same pros and cons.

- **Pros of L1 unstructured pruning**
  - *Relatively simple implementation*: Similar to random unstructured pruning, implementing $$\ell_1$$​ norm-based unstructured pruning is generally straightforward.
  - *Deterministic pruning*: Unlike random methods, $$\ell_1$$​ pruning is deterministic. Given the same network and pruning ratio, the same weights will always be pruned, making experiments reproducible.
  - *High flexibility*: It offers high flexibility by allowing the pruning of any individual weight within any layer of the network.
  - *Generally better accuracy than $$\ell_1$$​ structured pruning*: For a given sparsity level, $$\ell_1$$​ unstructured pruning typically achieves higher accuracy compared to its structured counterpart due to its fine-grained nature.
  - *Importance-driven pruning*: By focusing on pruning weights with the lowest $$\ell_1$$​ norm (which indicates lower magnitude and potentially less importance), this method can often achieve better performance than random unstructured pruning, especially at higher compression ratios, as it strategically removes less influential connections.

- **Cons of L1 unstructured pruning**
  - *Non-Uniform sparsity across dimensions*: The pruning pattern is not guaranteed to be uniform across different dimensions within a layer's weight tensor. This can potentially lead to imbalances in the importance of different features or connections.
  - *Limited hardware acceleration*: The irregular sparsity pattern resulting from unstructured pruning makes it very challenging to efficiently accelerate inference on standard hardware or specialized accelerators, which typically benefit from more structured sparsity.
  - *Potential for suboptimal networks at high sparsity*: At very high pruning ratios, even low-magnitude weights can be crucial. Removing too many based solely on magnitude can disrupt the network's learned weight distribution and lead to performance degradation.

#### L1 structured

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

- **Pros of L1 structured pruning** 
  - *Relatively simple implementation*: Similar to random structured pruning, implementing $$\ell_1$$​ norm-based structured pruning is generally straightforward.
  - *Deterministic pruning*: Unlike random structured methods, $$\ell_1$$ structured pruning is deterministic. Given the same network and pruning criteria, the same structural units will always be pruned. 
  - *Flexibility in pruning dimensions*: It allows for pruning along different dimensions of a tensor (e.g., rows, columns, channels, filters), enabling various forms of structured sparsity.
  - *Potential for easier acceleration*: By removing entire structural components, the resulting weight tensors have reduced dimensions, which can make them more amenable to basic hardware acceleration compared to unstructured sparsity.
  - *Quick path to model compression*: $$\ell_1$$​ structured pruning can provide a fast and effective way to obtain a compressed model with reduced size and computational cost.
  - *Achieves high compression ratios*: Due to the removal of entire structural units, this method can often achieve significant reductions in the number of parameters, leading to high compression ratios.

- **Cons of L1 structured pruning**
  - *The effectiveness* of structured pruning can be highly dependent on the specific architecture of the neural network. Some architectures might be more sensitive to the removal of certain structural units than others.
  - *Lower flexibility* Compared to Unstructured Pruning: Pruning entire rows, columns, or other structural units is less flexible than removing individual weights, potentially limiting the ability to fine-tune the sparsity pattern for optimal accuracy retention.
  - *More aggressive accuracy reduction at high sparsity*: Compared to $$\ell_1$$ unstructured pruning, $$\ell_1$$ structured pruning tends to be more aggressive in reducing accuracy, especially at high pruning ratios, as the removal of entire structures can have a more significant impact on the network's representational capacity. 
  - *Difficulty in leveraging structure*: Again as previously, even though structured pruning offers us some structure in the resulting pruned layer, its difficult to drop the dimensions entirely and requires very careful stitching.


In the rest of the post, we will explore each of those techniques on the ImageNet validation dataset (50000) by using as a base model the Vision Transformer model that you can find [[here]](https://huggingface.co/google/vit-base-patch16-224)

### Visual Transformers in ImageNet1000

Now that we've explored the different pruning methods, it's time to see how they perform in practice. We evaluated these techniques on the challenging ImageNet dataset, using the popular Vision Transformer architecture as our foundation. In our experiments, we adopted a strategy of applying the same pruning ratio to each layer throughout the pruning process, with the exception of the `LayerNorm` layers.

<div style="display: grid; grid-template-columns: repeat(2, 1fr); grid-template-rows: repeat(2, 1fr); gap: 10px; width: fit-content;">
  <img src="/assets/post_images/imagenet1000/pruning_results_random_unstructured.png" alt="Image 1" style="width: 80%; height: auto;">
  <img src="/assets/post_images/imagenet1000/pruning_results_random_structured.png" alt="Image 1" style="width: 80%; height: auto;">
  <img src="/assets/post_images/imagenet1000/pruning_results_l1_unstructured.png" alt="Image 1" style="width: 80%; height: auto;">
  <img src="/assets/post_images/imagenet1000/pruning_results_ln_structured.png" alt="Image 1" style="width: 80%; height: auto;">
</div>

The results indicate that the `l1_unstructured` pruning method yielded the most favorable trade-off between compression ratio and accuracy, which aligns with the expectation that removing weights with the lowest importance (as determined by the L1 norm) would be effective. As previously noted, the ln_structured method proved to be highly aggressive, leading to an immediate and significant drop in accuracy to an unacceptable level of approximately 20%. 

To mitigate the accuracy loss observed during pruning, we will investigate investigate in a bit the impact of fine-tuning. Specifically, we will attempt to recover accuracy for the `l1_unstructured` method at a pruning ratio of 0.5

We can also analyze the impact of pruning and potential fine-tuning, we can visualize the weight distribution of the classifier layer by examining its histograms.

#### Visualizing Weight Distributions (Classifier layer) across the different methods

<div style="display: grid; grid-template-columns: repeat(2, 1fr); grid-template-rows: repeat(3, 1fr); gap: 10px; width: fit-content;">
  <img src="/assets/post_images/cls_histogram_base.png" style="width: 100%; height: auto;">
  <img src="/assets/post_images/cls_histogram_l1_unstructured.png" style="width: 80%; height: auto;">
  <img src="/assets/post_images/cls_histogram_ln_structured.png" style="width: 80%; height: auto;">
  <img src="/assets/post_images/cls_histogram_random_structured.png" style="width: 80%; height: auto;">
  <img src="/assets/post_images/cls_histogram_random_unstructured.png" style="width: 80%; height: auto;">
</div>

#### Sensitivity analysis for L1 structured (further examining structured pruning)

Before we proceed to the last part of this post, which covers fine-tuning, we'll first conduct a sensitivity analysis with $$\ell_1$$​ structured pruning. The goal is to see if we can gain better accuracy by carefully choosing which layers to prune. We'll start by examining how the model performs when we try different pruning ratios for each layer.

<div style="display: grid; grid-template-columns: repeat(2, 1fr); grid-template-rows: repeat(2, 1fr); gap: 10px; width: fit-content;">
  <img src="/assets/post_images/l1_structured_cl_vs_projection.png" style="width: 80%; height: auto;">
  <img src="/assets/post_images/l1_structured_l11_layer.png" style="width: 80%; height: auto;">
  <img src="/assets/post_images/l1_structured_l0_layer.png" style="width: 80%; height: auto;">
  <img src="/assets/post_images/l1_structured_l5_layer.png" style="width: 80%; height: auto;">
</div>

It's interesting to observe that the effect of pruning varies significantly across different layers. Our results indicate that the embedding and classifier layers are the most important, which makes sense as they are the initial and final stages of the network. Heavily pruning these two layers can severely impact the network's performance. 

Additionally, the first Attention layer appears to be quite sensitive, especially when its intermediate dense module is heavily pruned. On the other hand, the last Attention layer and the 6th Attention layer, which is around the middle, seem to be less affected by pruning, even with substantial pruning ratios.

> 💡 **Important note**
> <br><br>
> It's important to remember that while sensitivity analysis offers valuable insights into the pruneability of individual layers, it doesn't account for the inter-dependencies between them.

Based on our initial experiments, we first attempted to apply pruning to all layers except the 1st, 12th, the Embeddings Projection (Convolutional layer), and the Classifier layer. However, we observed that the inter-dependencies between layers significantly impacted the outcome, with even modest pruning of the Attention layers leading to a decline in performance.

The most effective pruning configuration we identified involved applying $$\ell_1$$​ structured pruning to all layers except:

- The 1st layer
- The 12th layer
- The Embeddings Projection (Convolutional layer)
- The Classifier layer

Furthermore, within the Attention blocks of the remaining layers, we specifically pruned only the intermediate and output tensors, while preserving the *query*, *key*, and *value* tensors.

This configuration yielded an accuracy of 44% with a model sparsity of 20% in the $$\ell_1$$ structured pruning. Our next step will be to investigate whether fine-tuning can help recover some of the accuracy lost during the pruning process.

### Finetuning with pruning 

In a final effort to reclaim some of the performance lost during pruning, we proceeded to fine-tune the resulting sparse network. 
This step proved fruitful, allowing us to recover a significant portion of the original accuracy. 
Starting from an initial accuracy of 80.39%, we reached a respectable 67.24% with $$\ell_1$$ structured after fine-tuning. 

This represents a recovery of approximately 83% of the accuracy lost due to pruning. 
Considering that our initial aggressive application of $$\ell_1$$ structured pruning had severely impacted performance 
(dropping accuracy to around 20%, as noted earlier), the jump to 67.24% through careful layer selection and subsequent fine-tuning demonstrates the potential of this approach. 
This outcome suggests that while aggressive structured pruning can initially lead to a substantial accuracy drop, a well-executed fine-tuning strategy can effectively bridge the performance gap.

>💡 **Important Consideration for Finetuning**
> <br><br>
> In our earlier steps, we used the `remove()` function to solidify the weight mask and effectively eliminate pruned elements. <br> <br>
> This made the pruning permanent. However, when finetuning, we take a different approach. We deliberately don't immediately make the pruned elements permanent. The reason for this is that we want the weight masks to remain active, allowing the backward pass to specifically focus on updating the weights of the unpruned elements.


### Iterative approach

So, building on what we've learned, another way to really optimize our model is to try an iterative approach – kind of like a cycle of pruning and fine-tuning. 

We can try snipping away a little bit of the network, and then giving it a quick tune-up to make sure it hasn't lost its edge. 
By doing this over and over, we can often make the model even smaller and still keep it performing really well, sometimes even better than if we just pruned it all at once. 

<br>

<div style="text-align: center;">
  <img src="/assets/post_images/iterative_pruning.png" style="width: 40%;">
</div>

<br>

This gradual process helps the network get used to having fewer connections and learn to work efficiently with what it has. Since we already saw some good results with just one round of pruning and tuning, giving this iterative method a shot looks like a promising next step to really make our Vision Transformer shine when we want to put it to work! It's all about finding that sweet spot of size and performance.

### Results

Now that we've covered the basics, let's dive into the results! The figure below compares the performance of the various pruning methods and strategies we experimented with.

<br>

<div style="text-align: center;">
  <img src="/assets/post_images/ln_structured_base_finetuned_iter.png" style="width: 80%;">
</div>

- For the Random unstructured, Random structured, and L1 structured methods, we only achieved sensible results using a 22% pruning ratio, and only when restricting pruning to the layers mentioned previously.

- When it comes to preserving accuracy, L1 unstructured pruning proved to be the most effective approach. In fact, we were able to restore the network to its original level of performance using this method.


### In Conclusion: Pruning for Efficiency

Our exploration into post-training model optimization highlighted pruning as a valuable technique for creating more efficient machine learning models. 
We observed that L1 unstructured pruning excelled at maintaining accuracy, while structured methods offer potential speedups with careful application. 
Fine-tuning proved essential for recovering performance after pruning, and iterative approaches hold promise for further optimization.

Ultimately, pruning is a key tool in the ongoing effort to deploy powerful AI models efficiently across various platforms. 

You can find the code available here: [Github](https://github.com/psouranis/pruning)

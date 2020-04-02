# OpenAI Scholars: Third Steps - Making a Pytorch Dataset and Dataloader

This blog is about two very nifty `PyTorch` tools, the `dataset` and the `dataloader`. My main learning resource was [this tutorial](https://pytorch.org/tutorials/beginner/data_loading_tutorial.html), which you might find useful too.

## The dataset

I don't know the person who invented the `PyTorch dataset`, but I picture that their thought process was as follows: "I train a lot of neural networks on a lot of different kinds of data. Each time, I have to write a new set of functions for reading, processing, and feeding that data into a neural network. I just wish I could work with, and index into, any dataset as easily as you index into a [Python list](https://www.youtube.com/watch?v=zEyEC34MY1A)." And so, the `dataset` was born. (I conjured this story up in my mind: I make no claims to veracity.)

For me, it really is intuitive to think of a `dataset` as a list, where each element is an example from your data. In my data, an element example will be an image stored as a numpy array, and an associated label.

**Two critical features** of a `dataset` are the **`__len__`** and **`__getitem__`** methods. The fact that your `dataset` has a **`__len__`** method, means that you can simply call `len(my_dataset)`, and it will return the number of examples you have, as if it were a list. The fact that your `dataset` has a **`__getitem__`** means that you can index into it, again as if it were a list. For example, if you call `my_dataset[253]`, this would give you the 254th example in your dataset.

**An additional cool feature** of a `dataset` is that you can apply transforms to your examples on the fly, for example converting numpy arrays to PyTorch tensors, z-scoring the data, etc. See [the tutorial](https://pytorch.org/tutorials/beginner/data_loading_tutorial.html) for examples.

### How do you make a dataset?

The very simplest way is to not create one at all. If you just want to practice training a neural network, you can use an existing `dataset` that is available in `PyTorch`. CIFAR10 is one example, and here is a [good tutorial](https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html).

But if you have some of your own data that you want to try applying a neural network to, the simplest way that you can make a `dataset` for images (or other data that can be treated as an image) is by using `ImageFolder`. In this case, all you need to do is to organize your images into a particular folder structure, by label, and then [`ImageFolder`](https://pytorch.org/docs/stable/torchvision/datasets.html#imagefolder) does the rest of the work for you (see [the tutorial](https://pytorch.org/tutorials/beginner/data_loading_tutorial.html) for an example). A more general version of `ImageFolder` is [`DatasetFolder`](https://pytorch.org/docs/stable/torchvision/datasets.html#datasetfolder), which should work for any type of data, again provided that you organize your data into folders the same way as for `ImageFolder`.

In my case, the data that I wanted to feed into the CNN from CIFAR10 tutorial above, contained examples that were way larger than I was looking for. I didn't want to manually reshape them all in case I would want to change those hyperparameters (width and height of the final images). Hence, I wrote my own dataset class, inheriting from the general PyTorch `dataset` class. If you do this, the central point is that you will want to write your own, custom `__len__` and `__getitem__` functions, so that they return the length of the dataset, and the correct data example, when called. I ended up creating a csv of file annotations, similarly to [the tutorial](https://pytorch.org/tutorials/beginner/data_loading_tutorial.html). The annotation csv had a row for each example, where it provided the path to the relevant datafile as well as the starting and ending indices from which to select part of the image. Then the length could just return the full length of the csv, and when you index into the dataset, it would read the relevant file, and select the part of the image that is relevant for the example at hand.

Importantly, this implementation means that I read from disk each time I draw an example image. I learned from my mentor that this will substantially slow down network training. Hence, moving forward, I will rewrite my dataset so that it loads all the data into memory at once. To load an example, the dataloader would then index into this preloaded array, which is a much faster operation than reading from disk.

## The dataloader

OK, so you have a `dataset` that allows you to get the length of your dataset, and read individual items from it as simply as if it were a list. What do you want to do next? That's right, you want to draw examples from your dataset to train your neural network. Specifically, you would want to draw samples in [batches](https://machinelearningmastery.com/difference-between-a-batch-and-an-epoch/); you might want to randomize the order of the presentation of batches to your neural network; and you might want to parallelize these operations, because neural network training is computationally heavy.

A `dataloader` allows you to do all of these things. You can think of it as a **fancy iterator** for your dataset.

### How do you make a dataloader?

You can create a very simple dataloader with the following line of code:

`dataloader = dataloader(my_dataset, batch_size=1, shuffle=False, num_workers=1)`

This dataloader is literally just a simple iterator for your dataset. If you want to use some of its fancier features, you can just change some of the arguments, like so:

`dataloader = dataloader(my_dataset, batch_size=4, shuffle=True, num_workers=4)`

## Conclusion

That's it. If you have successes or struggles with these tools, perhaps if you think of cool"power user" hacks for them, or if you have feedback on this blog post, I would love to hear from you.

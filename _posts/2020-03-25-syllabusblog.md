# Kata's OpenAI Scholars Curriculum: A Living Document

## Approach

My goals are currently centered on moving towards a project as fast as possible. I'm focusing on learning how to train neural networks in pytorch, and exploring a candidate dataset. My two goals with that are to: (1) assess the feasibility of a candidate project, and (2) learn enough pytorch syntax to be able to execute the project. It's a very hands-on approach.

Other scholars have made beautiful learning-oriented syllabi, where they're working through large swaths of online course materials and readings. They're making highly personalized in-depth courses in their favorite deep learning topics. Thus far, this is not a central part of my plan. That might be a mistake. I'd love to hear your thoughts about it.

The main argument in favor of a learning-oriented curriculum is: "When will I ever have this opportunity again?" Most jobs don't allow for a lot of space for learning and self-development. It's extremely generous of OpenAI to provide us with this incubation period, and I should take advantage of it. It is possible that a goal-oriented learning syllabus is the most efficient way to absorb a new set of topics and skills. Plus, it would be so much fun.

The main argument in favor of a project-oriented curriculum is: "I will inevitably learn a lot anyway." If I start on a candidate project, I will necessarily get stuck, and come up with questions along the way. I should tailor my studying to those questions. I hope that this will, in an organic way, grow into the right curriculum for me.

I still want to maintain an element of strategy in my learning. I plan to dedicate a certain amount of time each day specifically to learning (as opposed to producing). However, I want to allow the content selection to be curiosity-driven. In other words, I want to be able to select my reading for tomorrow morning, based on something that piqued my interest today.

## Disposition
This is what I ask from myself for my Scholars period: Work on your most important goal early in the week. If you can only finish one thing this whole week, what would it be? If it is unclear what is most important, assume that the thing you feel most inclined to avoid is the most important. It will typically be something that you feel you don't have the skills to do yet. Do it anyway.

## Questions for the reader
- To what extent should my Scholars time be a time for *learning* versus *producing*?
- What is the best implementation of either approach?
- How can I hold myself accountable to my "disposition" above?

## Daily time division

**Mornings** are my best opportunity to learn new things, i.e., sustain attention on new information. They are designated for "heavy learning": papers, blogs, and books. Because these materials don't have an obvious associated motor task (unlike coding), I have to use more mental discipline to maintain focus. This is best done in the morning. The purpose of any one morning's learning can be to understand a topic that I was left wondering about the day before: Blog posts with visualizations are great for this. Another purpose can be to inspire and motivate me for my work: Historical papers are great for noticing the beauty of a field, by observing how much work and thought other people have put into it. I plan to spend 1-4 [pomodoros](https://en.wikipedia.org/wiki/Pomodoro_Technique) on this kind of learning per morning, depending on how exciting the material is versus how urgently I need to move onto my coding tasks.

**Afternoons** are for coding; making progress on my candidate project; or trying toy problems. Because coding naturally involves motor actions, which are easier to sustain even when I'm somewhat tired, it's a great task for the afternoon. Afternoons also aren't amazing and magical the same way that mornings are, so you might as well meet your error messages and do your debugging at that time.

**Evenings** are for: overflow work (caused by over-optimistic time estimates); watching light video explanations of concepts that I'm studying; or writing and reflection. Sometimes evenings are for hanging out with friends or knitting.

**Update**: I received feedback that I shouldn't underestimate the challenges that I will encounter with the coding component of my work. This rings especially true to me since my background is not in software engineering. Honestly, I don't yet have a good enough feel for the magnitude and type of challenges I will encounter when learning how to run neural networks on GPUs. I will probably need to rethink how to integrate this with my day-to-day time usage: It might be that I should divide programming into *hard programming* for mornings versus *easy programming* for afternoons. I'll let you know how it goes.

## Goals

### Project goals
I want to apply deep learning models to publicly available neuroscience data. Since my background is in neuroscience, I am interested in the ways that artificial and biological neural networks can meet. One way is by using artificial neural networks to help solve problems in neuroscience. I am especially interested in models that work well for sequences, such as transformers.

I am also interested in interpretability, and OpenAI is a [fantastic place](https://distill.pub/2018/building-blocks/) for delving into that.

### Learning goals
My learning will be guided by my project on a week-to-week and day-to-day basis. It might be worth being more goal-oriented about my learning, and I would love to hear your feedback about that.

### Social-ish goals

I am very lucky to have the opportunity to work on-site at OpenAI. I want to make the most of that experience.
- Get to know people and projects at OpenAI.
    - Sit with a new person each day at lunch. Learn something about them as person. Learn what they're working on.
- Ask a random person about a current conundrum (maybe a question that came up in your reading or coding.)

## Resources
(If a resource is marked with '\*\*', it means that I have at least looked at it.)

### Reading

#### Background for reading papers more effectively
- ** The Deep Learning Book Chapter 1 ([link](https://www.deeplearningbook.org/))
- Andrew Ng's Deep Learning Course ([link](https://www.deeplearning.ai/))

#### Fundamentals

##### Deep learning
- ** A brief overview of deep learning by Ilya Sutskever ([link](http://yyue.blogspot.com/2015/01/a-brief-overview-of-deep-learning.html))
- **Next 0:** (read this a second time to understand convolutions in multiple dimensions and how the dimensionality decreases depending on how the convolution is done): Andrej Karpathy's course on Convolutional Neural Networks for Visual Recognition ([link](http://cs231n.github.io/)): Seems to be a generally good resource to draw from along the way.

##### Backpropagation
- ** Lecture notes on backpropagation by Andrej Karpathy ([link](http://cs231n.github.io/optimization-2/))
- ** Classical backpropagation paper, Rumelhart et al. (1986) ([link](http://www.iro.umontreal.ca/~vincentp/ift3395/lectures/backprop_old.pdf))

##### Convolutional Neural Networks (CNNs)
- ** Lecture notes on CNNs by Andrej Karpathy, ([link](http://cs231n.github.io/convolutional-networks/))

- ** AlexNet/ImageNet paper by Krizhevsky, Sutskever, and Hinton  ([link](http://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf))
- ** The ResNet paper by He et al. ([link](http://openaccess.thecvf.com/content_cvpr_2016/papers/He_Deep_Residual_Learning_CVPR_2016_paper.pdf))
- ResNet blog post ([link](http://torch.ch/blog/2016/02/04/resnets.html))

##### Batches and batch size
- ImageNet
    - Very efficient ImageNet training by Goyal et al. ([link](https://research.fb.com/wp-content/uploads/2017/06/imagenet1kin1h5.pdf))
    - Training ImageNet in four minutes ([link](https://arxiv.org/abs/1807.11205))

##### Other classical papers
- ** The variational autoencoder paper by Kingma and Welling ([link](https://arxiv.org/abs/1312.6114))

#### Attention and transformers
- ** Attention is All You Need paper by Vaswani et al. (2017) ([link](https://arxiv.org/pdf/1706.03762.pdf))
- ** The annotated transformer blog post ([link](http://nlp.seas.harvard.edu/2018/04/03/attention.html)): Walks through the entire attention paper above, including a complete and explained code implementation in pytorch.
- ** The illustrated transformer blog post by Jay Alammar ([link](http://jalammar.github.io/illustrated-transformer/))
- ** Lilian Weng's blog post on language models: Focus on the differences between GPT-2 and BERT ([link](https://lilianweng.github.io/lil-log/2019/01/31/generalized-language-models.html))
- **Next 1:**: Read GPT-2 and perplexity sections from this post.
- **Next 2:** Evaluation metrics for language models, such as "perplexity" ([link](https://thegradient.pub/understanding-evaluation-metrics-for-language-models/))
- **Next 3:** Byte pair encoding: Useful blog with code ([link](https://leimao.github.io/blog/Byte-Pair-Encoding/))
- Lilian Weng's post about word embeddings ([link](https://lilianweng.github.io/lil-log/2017/10/15/learning-word-embedding.html))
- Byte pair encoding paper ([link](https://arxiv.org/abs/1508.07909))
- BERT paper by Devlin et al. (2019) ([link](https://arxiv.org/pdf/1810.04805.pdf))
- GPT paper by Radford et al. (2018) ([link](https://s3-us-west-2.amazonaws.com/openai-assets/research-covers/language-unsupervised/language_understanding_paper.pdf))
- The illustrated GPT-2 blog post by Jay Alammar: I should focus on the differences between GPT-2 and BERT ([link](http://jalammar.github.io/illustrated-gpt2/))
- The Reformer paper, supposedly a "better" implementation of the transformer ([link](https://arxiv.org/abs/2001.04451)): Interesting, but complicated. Not the first thing to try to implement.

#### Neural networks for speech recognition and audio data
- ** wav2vec paper by Schneider et al. ([link](https://arxiv.org/abs/1904.05862))
- ** wav2letter paper by Collobert, Puhrsch, and Synnaeve ([link](https://arxiv.org/abs/1609.03193))

#### Interpretability
- ** The building blocks of interpretability on Distill ([link](https://distill.pub/2018/building-blocks/))
- Feature visualization on Distill ([link](https://distill.pub/2017/feature-visualization/))
- Differentiable image parameterizations on Distill ([link](https://distill.pub/2018/differentiable-parameterizations/))
- The public slack channel for the Distill community: [slack.distill.pub](slack.distill.pub)
- Lucid colab notebooks on interpretability ([link](https://github.com/tensorflow/lucid#notebooks))

#### AI safety (personal interest)
- Concrete problems in AI safety paper ([link](https://arxiv.org/abs/1606.06565))

#### Links between AI and neuroscience (personal interest)
- ** DeepMind's Psychlab paper by Leibo et al. ([link](https://arxiv.org/abs/1801.08116))
- Definitions of intelligence (personal interest)
  - Shane Legg's thesis [link](http://www.vetta.org/documents/Machine_Super_Intelligence.pdf)
  - More recent paper, "On the measure of intelligence" [link](https://arxiv.org/abs/1911.01547)

### Coding
#### Fundamentals
- ** Pytorch 60-minute blitz (this is a misnomer :-)) ([link](https://pytorch.org/tutorials/beginner/deep_learning_60min_blitz.html))
- Pytorch beginner neural network tutorial ([link](https://pytorch.org/tutorials/beginner/nn_tutorial.html))

#### Attention and transformers
- Pytorch transformer tutorial ([link](https://pytorch.org/tutorials/beginner/transformer_tutorial.html))


### Videos
#### Attention and transformers
- ** fast.ai video on Transformers by Rachel Thomas #1 ([link](https://www.youtube.com/watch?v=AFkGPmU16QA))
- ** fast.ai video on Transformers by Rachel Thomas #2 ([link](https://www.youtube.com/watch?v=KzfyftiH7R8))
- ** Video walk-through of Attention is All You Need paper by Yannic Kilcher ([link](https://www.youtube.com/watch?v=iDulhoQ2pro))

### Logistics
- ** The fast(.ai) way of setting up your free blog on github ([link](https://www.fast.ai/2020/01/16/fast_template/)): And you can, too!

## Weekly goals

### Week 1

- Train a simple classifier on a typical dataset, like MNIST or CIFAR-10. (I ended up working on [this](https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html#sphx-glr-beginner-blitz-cifar10-tutorial-py) CIFAR-10 tutorial).
- Explore a candidate dataset:
    - Find some publicly available data that I might use.
    - Summarize the dataset properties (number subjects, samples, etc.)
    - Visualize the data.
- Think through how I might apply the simple network that I trained to the candidate dataset.
- Create a syllabus to share online. (You're looking at it!)
- Draft a first blog post.
- Read every morning: See "Approach" and "Daily time division" above.

### Week 2

- Train a transformer (e.g., BERT) and use it for a classification task.
- Secondary programming goal: Play with an existing neural network for audio data.
- Take a look at the BERT paper.
- Illustrated GPT-2 blog post by Jay Alammar.
- Read every morning: See "Approach" and "Daily time division" above.
- Blog post.

### Week 3
- Train a transformer-based classifier on audio data or neural data.
- Read every morning: See "Approach" and "Daily time division" above.

### Week 4
- Train a generative, autoregressive transformer to generate new data on Wikitext103, or audio data.
- Secondary programming goal: Try audio synthesis ([link](https://arxiv.org/abs/1802.08435)).
- Read every morning: See "Approach" and "Daily time division" above.
- Blog post.

### Week 5
- Train a generative, autoregressive transformer on neural data to generate new data.
- Try wavenet ([link](https://arxiv.org/abs/1901.08810)).
- Read every morning: See "Approach" and "Daily time division" above.

### Week 6
- Implement some early interpretability tools for my existing model.
- Blog post.

## Implementation
For full disclosure, there is an *even more* living version of my syllabus, in the form of a spreadsheet that I update several times a day. (I suppose it's a syllabus-and-to-do-list.) Its rows are the days of the week, with a header row for each week. Its columns are:
- Reading (heavy learning: papers, blogs, books)
- Programming (toy problems)
- Project progress
- Writing and reflection
- Watching and talking (light learning: videos, conversations)
- Backlog (Leftovers that I didn't get to that day, and that aren't sufficiently high priority to be moved to the next day. Good to revisit from time to time.)

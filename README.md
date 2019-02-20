# Eyewire Validation

Project By: Brian Lim

## Abstract
Eyewire is a citizen science game where players map individual neurons in a rat's retina. Using a relatively new method of deep learning called Bayesian Deep Learning, I create an agent to approximate the individual tasks (a.k.a. cubes) that human players complete as part of Eyewire. The architecture used in the model is a recurrent encoder-decoder network for 3D images. 3D images pose a huge challenge in memory/size restrictions, so methods dealing with those had to be engineered.

## Summary of Eyewire and Terms

### Terms
* Volume: A 3D image stack
* Seed: The initial segments in a cube
* Cube: A 3D image stack and seed given to players as a task
* Task: See Cube
* Trace: The act of propogating the seed to find missing segments within a cube
* Scythe: A high-ranking, established Eyewire player, given additional responsibilities of validating individual cubes
* Reap: Using a scythe's override powers to fix a cube
* Merger: A segment that contains volumes of two different neurons


### The Eyewire Task Pipeline
In Eyewire, players are given a 256x256x256 segmented 3D image (called a cube) containing a set of seed segments propogated from previous tasks. Players are to propogate (a.k.a. tracing) the seed and find the missing segments in the cube, giving Eyewire the nickname "adult coloring." As players play individual cubes, their traces are aggregated into a consensus and marked as the combined trace of the playerbase.

<p align="center">
  <img src="http://wiki.eyewire.org:88/images/thumb/4/4a/1.png/800px-1.png" title="Now that you know how to play, go join the Eyewire community" />
</p>

After 3 players have played a single cube, the cube is then marked to be checked by more established players known as Scythes. Scythes have full override powers of tasks, being able to fix any mistake that the consensus has. It takes 2 Scythes to mark a cube as complete. After a cube is marked as complete and a whole neuron has been traced, 1-2 administrators (Eyewire/Seung Lab employees) are tasked with validating the full trace again to make sure there are no mistakes. 

### The Eyewire Task Pipeline Summary
TL;DR
* 3 Players play a cube
* 2 Scythes validate a cube and reap it if necessary
* 1-2 Admins validate a whole neuron when it is done

## Motivation
Here are a list of problems that are present in the current Eyewire Task Pipeline

* As seen from the Eyewire Task Pipeline, 6-7 people are required at minimum to look at a single task. This is a very time consuming process used to anonate petabytes of data, resulting in less than 10,000 total neurons being traced over the course of 7 years. 

* The segmentation algorithm used to generate the cubes is, although quite robust, still rudimentary as it does not take into account the full range of data that is provided by the Eyewire Dataset. 

* The segmentation algorithm's errors propogate to the players. There are many instances of mergers through the dataset, causing players to mistakenly add additional segments to a trace. The default procedure is to remove all segments that contain significant mergers to avoid multiple segments being classified as part of 2 different neurons.

<p align="center">
  <img src="http://wiki.eyewire.org:88/images/thumb/9/9c/No_borders2.png/800px-No_borders2.png" title="Tracing mergers is the chaotic evil of Eyewire" />
</p>

## Challenges
Here are a list of challenges associated with this project

* 3D images are incredibly large and consumes large amounts of memory. A 5 layer neural network transforming a 256^3 image into an output of size 256^3 does not fit on 12 GB of RAM.

* There are 2 million separate tasks across 56,300 volumes, across 3300 cells. Gathering the data is extremely time consuming and slows the servers hosting Eyewire, totaling 500 GB of data. The full E2198 Dataset is several TB.

* Images are stored as 256 individual 2D images, making file transfers and image loading an extremely costly operation.

* There is a subset of cells and images from the brain stems of a zebrafish that are not the same size as rat's retina images.

* External hard drive transfer speeds are capped so running scripts off it is time consuming. Windows also has this amazing feature that shuts off a hard drive if it uses too much power.

## Solutions Summary

* To remedy the cost of memory, I downsample both images and segmentation by a factor of 4 on each dimension to retain some detail. I am only looking for an approximation of the final annotation.

* Using a combination of asyncio and multiprocessing, I can set the transfer speed of the data gathered through the Eyewire API.

* After gathering the images, I merge them all into 3D numpy arrays and compress them using the lzma compression library.

* I remove all zebrafish cells from the data. Eyewire is already developing an agent for this called MSTY.

* I use my secondary laptop with lots of disk space to help preprocess some of the data

## Solutions

### Downsample Images

Segmentations of volumes are downsampled to a size of 64x64x64 by taking the max segment count in each subvolume of 4x4x4. This causes some tasks to not have any segments in the seed, so we have to remove those tasks. 

Images are downsampled by taking the mean of each 4x4x4 block. Using skimage's downsample_local_mean function, we can quickly do that. 

<p float="left">
  <img src="https://storage.googleapis.com/e2198_compressed/Volume-71141-71142/jpg/0.jpg" width="49%" title="20/20 vision" />
  <img src="https://i.imgur.com/NTVp4Hx.jpg" width="49%" title="20/200 vision" /> 
</p>

### Using asyncio and multiprocessing

Processing large amounts of data sequentially is extremely time consuming especially with the additional cost of HTTP requests. To combat that, we have to run parallel processes using python's asyncio and multiprocessing libraries.

Asyncio skeleton code
```python
def func_(*args):
  # do things
  return result

async def func(list, *args):
  loop = asyncio.get_event_loop()
  futures = [
    loop.run_in_executor(
      None,
      func_,
      item,
      args*
    )
    for item in list
  ]
  results = [await f for f 
             in tqdm.tqdm(asyncio.as_completed(tasks), 
             total=len(tasks))]
             
  for result in results:
    #do things

if __name__ == "__main__":
  list = [#things]
  loop = asyncio.get_event_loop()
  loop.run_until_complete(func(list))
```

Multiprocessing skeleton code
```python
def func_(val):
  # do things
  time.sleep(n)
  return result
  
list = [#things]
with multiprocessing.Pool(np) as p:
  results = list(tqdm.tqdm(p.imap(func, list), total=len(list))
```

Asyncio is not controllable, so it is impossible to throttle the speed. Since the Eyewire API is connected to the game, HTTP requests slow down the game and data collection. This is when multiprocessing has to be used. Asyncio and multiprocessing were used in the data gathering step and preprocessing steps.

tqdm is a library that displays pretty progress bars. I highly recommend it for any script dealing with large loops.
## Bayesian Deep Learning

Bayesian deep learning is new method of deep learning that deals with the problem of uncertainty. Take a model that tries to classify dogs and cats. What would the model output if it came across this image?

<p align="center">
  <img src="https://github.com/kyle-dorman/bayesian-neural-network-blogpost/blob/master/blog_images/catdog.png?raw=true" title="Dogcat? Catdog? Let's just call it a monster" />
</p>

Ideally, there would be a 3rd category that has a label of neither dog nor cat, but adding that label would cost additional data collection. The solution is for the model to output both the classification and an uncertainly value. We can train the model with a loss function that penalizes uncertainty (u) by adding log(u) to the loss function when the original prediction is adjusted for u. 

In the case of binary classification, our output (ŷ) is within the range (0,1), and our uncertainty (u) is within the range \[1, inf). Our equations are given as follows:

* Probability of output given ŷ, u

<p align="center">
  <img src="https://latex.codecogs.com/gif.latex?P(\^{y},&space;u)&space;=&space;\sigma\(\frac{\^{y}}{u}\)" title="Oh god complex equations" />
</p>

* Loss of output given ŷ, u, and ground truth (y)

<p align="center">
  <img src="https://i.imgur.com/UKOWaJb.gif" title="AHHHH MORE COMPLEX EQUATIONS" />
</p>

Where σ is the sigmoid function


The result of these equations is that as uncertainty increases, P(ŷ, u) approaches 0.5 and cancels out the logit output ŷ. The sigmoid function is a very sensitive function, making inputs just outside 0 approach either a 0 or 1 probability. Therefore, learning an uncertainty term allows the model to make a guess without being penalized too heavily. We can also derive insight from the uncertainty term itself.

## Future Work

* Fix the recurrent portion of the model
* Train the model on AWS
* Train the model using several n-grams of images, tuning for best accuracy
* Gather metrics from training
* Built resulting visuals

## Credits

* Eyewire: Providing the resources for this project
  * Amy Sterling: Connecting me to the development team, and for supporting me in building this project
  * Chris Jordan: Creating the OAuth API authentication for me to connect to the API without using a logged in browser
  * Doug Bland: Moral support
* Princeton University - Seung Laboratory: Maintaining Eyewire to be an awesome citizen science game and for developers to work on my requests
* MIT: Building Eyewire
* Professor Yu-Xiang Wang: Providing insight into segmentation and how to create a model

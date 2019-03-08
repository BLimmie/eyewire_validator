# Eyewire Validation

Project By: Brian Lim

Note on the repository: This repository is meant to be a record of the various scripts used. It does not contain the data files necessary to run, nor does this repository note the required location of said data files. Some of the files are not runnable from the folder containing them in the repository and need to be placed in the same directory as the data. The tools directory is within the modeling folder; it was used with many of the data gathering scripts.

<p align="center">
  <a href="https://blog.eyewire.org" target="_blank">
    <img src="https://upload.wikimedia.org/wikipedia/commons/8/8d/EyeWire-Logo-Blue.png" title="Imagine your eyes hooked up to jumper cables" width= "50%"/>
  </a>
</p>

## Abstract
Eyewire is a citizen science game where players map individual neurons in a rat's retina. Using a relatively new method of deep learning called Bayesian Deep Learning, I create an agent to approximate the individual tasks (a.k.a. cubes) that human players complete as part of Eyewire. The architecture used in the model is a recurrent encoder-decoder network for 3D images. 3D images pose a huge challenge in memory/size restrictions, so methods dealing with those had to be engineered.

## Purpose
This project is designed as an introduction to Bayesian Deep Learning and am invitation to join citizen science efforts such as Eyewire.

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
* Harold: The rat's name

### Eyewire

Eyewire is a citizen science game designed to collect data about individual neurons in a rat's retina. Originally designed to create a segmentation algorithm for mapping the human brain, it has evolved into a complex data collection system based on the idea that getting ordinary people to play a game designed around annotating data creates a cost-effective way of annotating data that doesn't require the time and effort of people within the laboratories using the data nor the costly delegation of data annotation to services like Mechanical Turk. Currently, there are just over 3000 rat's neurons fully mapped within a 350×300×60 µm³ volume. At this point in time, the data collected by Eyewire has discovered 6 new specific neuron types and has made a new classification scheme for retinal neurons.

<p align="center">
  <img src="http://wiki.eyewire.org:88/images/thumb/0/05/Countdown.png/800px-Countdown.png" title= "Remove the cell bodies and you get random noise" />
  <br>
  245 of the 3000+ neurons mapped with blood vessels
  <br><br>
  <img src="https://i.imgur.com/LxxHDTS.jpg" title= "This image is of utmost importance. There will be a test on it." width="70%" />
  <br>
  Vertical cross section of the dataset with a sample neuron
</p>

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

The asyncio is not controllable, so it is difficult to throttle the speed. Since the Eyewire API is connected to the game, HTTP requests slow down the game and data collection. This is when multiprocessing has to be used. Asyncio and multiprocessing were used in the data gathering step and preprocessing steps.

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
  <img src="https://i.imgur.com/1CVpdVU.gif" title="Oh god complex equations" />
</p>

* Loss of output given ŷ, u, and ground truth (y)

<p align="center">
  <img src="https://i.imgur.com/UKOWaJb.gif" title="AHHHH MORE COMPLEX EQUATIONS" />
  <br>
  Where σ is the sigmoid function
</p>

The gradients of the loss function for backpropagation are as follows:
<p align="center">
  <img src="https://i.imgur.com/EDo6yQl.gif" title="As the limit of the line number increases to infinity, the complexity of the math equations increases" />
  <br> <br>
  <img src="https://i.imgur.com/BI0dQyI.gif" title="To be honest, I'm just adding the partial derivatives to make this seem more complicated than it really is" />
</p>

The result of these equations is that as uncertainty increases, P(ŷ, u) approaches 0.5 and cancels out the logit output ŷ. The sigmoid function is a very sensitive function, making inputs just outside 0 approach either a 0 or 1 probability. Therefore, learning an uncertainty term allows the model to make a guess without being penalized too heavily. We can also derive insight from the uncertainty term itself.

The uncertainty term can even further be used to set a "maybe" threshold. If the uncertainty is higher than a certain value, it gets assigned a maybe label, which would be the exact scenario we want from the merged dog and cat image. This creates an artificial 3rd label that doesn't require extra data specifically for that label.

## The Model

The model architecture is a recurrent lightweight U-Net. It consists of 2 Conv3D/ReLU/MaxPool layers, of which the outputs gets concatenated to the 2 ConvTranspose3D/ReLU layers. The output of that is fed into two 1x1x1 Conv3D layers to calculate the predicted output and uncertainty of each pixel.

<p align="center">
  <img src="https://i.imgur.com/0lV3Si8.png" title="This is so lightweight, it can actually fit on 12 GB of GPU RAM" />
</p>

The input confidence is defaulted to 1 for the seed and 10 for everywhere else as it hasn't been explored yet. The hope for this model is that on each loop, it will explore the surroundings of the current seed and decide what belongs, what doesn't belong, and what it is still uncertain about. We also input p previous ground truths, of which the null equivalent is all zeros, so that the model gets previous cubes for extra context.

We loop n (default=4) number of times through the U-Net model plugging in the logits as the new seed and sigma as the new confidence. We hope the high confidence value on the initial iteration for non-seed voxels causes the model to ignore the initial values of 0 in the seed.   

## Results and Analysis

On the test set, we get the following results:
```json
{
  "loss": 1.0000137219926748, 
  "precision": 0.9998482499519548, 
  "recall": 0.9997848688364814, 
  "iou": 0.9996852392996376, 
  "seedless_precision": 0.9997686750150999, 
  "seedless_recall": 0.9997559129969251, 
  "seedless_iou": 0.9995771374986273
}
```
### Definitions

#### Loss

This is just a number used in the optimization of the model. Lower loss means the model fits better. This doesn't say much about how well the model does by itself, so we can ignore this value for most intents and purposes.

#### Precision

Precision is the percentage of voxels the model predicted to be positive in the ground truth that are actually positive in the ground truth.

<p align="center">
  <img src="https://i.imgur.com/LVko80D.gif" title="How many arrows did you hit?"/>
</p>

#### Recall

Recall is the percentage of ground truth positive voxels that the model managed to predict

<p align="center">
  <img src="https://i.imgur.com/Z1uDiuc.gif" title="Of all the arrows that hit the target, how many are yours?"/>
</p>

#### Intersection over Union (IOU)

IOU is the volumetric intersection of positive guesses and ground truth positive voxels divided by the union of the two.

<p align="center">
  <img src="https://i.imgur.com/HTWhRdF.gif" title="Over all the arrows you shot and arrows that hit the target, how many did you hit?"/>
</p>

### Analysis

#### Uncertainty

First, I want to look at the uncertainty of differently shaded voxels, since there is a known phenomenon within the dataset called a "black spill." A black spill is a volume of voxels where the dye accidentally spilled into the surrounding area instead of just the cell membrane. These voxels can be extremely difficult to trace through because there is no indication of where membranes are.

Because there are actually very few voxels where uncertainty is more than the minimum of 1, I opt for a scatter plot to show where any uncertainties lie at all. This doesn't show the concentration of uncertainties, but there is a clear range where uncertainties lie.

<p align="center">
  <img src="https://github.com/BLimmie/eyewire_validator/blob/master/images/uncertainty.png?raw=true" title="To be honest, I expected cooler results than \"The model works really ****ing well\"/>
</p>

Besides the anomaly of darkness = 0, we notice that the model only has uncertainty within a certain range of darker values and no uncertainty within much lighter values. This is somewhat strange since another potential area of error are membranes not being dyed, causing merger segments. I think the additional information of parent tasks makes the model certain about the direction of the neuron's growth.

This matches the theory that black spills cause uncertainty.

#### Loops = 1

What happens if we only loop through the model once?

```
{
  "loss": 1.0000137200649821, 
  "precision": 0.9999090572150231, 
  "recall": 0.9999214975016473, 
  "iou": 0.9998205810386009, 
  "seedless_precision": 0.9998980110984516, 
  "seedless_recall": 0.9999177439668899, 
  "seedless_iou": 0.9998057813872722
}
```

We get better results? I guess the model is good enough on its first try that any additional tries, it second-guesses itself. We get such a small uncertainty on the first loop that a float rounds it to the minimum possible uncertainty. I think I like it when the model loops more because it can be more uncertain. The first loop essentially negates the necessity of bayesian deep learning, and we want our model to know when it can be wrong.

## Future Work

* Switch model/loss to Monte Carlo Simulations and test

## Credits

* Eyewire: Providing the resources for this project
  * Amy Sterling: Connecting me to the development team, and for supporting me in building this project
  * Chris Jordan: Creating the OAuth API authentication for me to connect to the API without using a logged in browser and providing API documentation support
  * Doug Bland: Moral support 
* Princeton University - Seung Laboratory: Maintaining Eyewire to be an awesome citizen science game and for developers to work on my requests
* MIT: Building Eyewire
* Professor Yu-Xiang Wang: Providing insight into segmentation and how to create a model for the purposes of propagating a seed

<p align="center">
  <img src="https://i0.wp.com/blog.eyewire.org/wp-content/uploads/2018/03/ARTPRINT_ForScience_BLACK.jpg?resize=700%2C700&ssl=1" title="科学のために！" width = 70%>
</p>

# Eyewire Validation

Project By: Brian Lim

Note on the repository: This repository is meant to be a record of the various scripts used. It does not contain the data files necessary to run, nor does this repository note the required location of said data files. Some of the files are not runnable from the folder containing them in the repository and need to be placed in the same directory as the data. The tools directory is within the modeling folder; it was used with many of the data gathering scripts.

<p align="center">
  <a href="https://blog.eyewire.org" target="_blank">
    <img src="https://upload.wikimedia.org/wikipedia/commons/8/8d/EyeWire-Logo-Blue.png" title="This image takes you to Eyewire. Please join citizen science efforts such as this!" width= "50%"/>
  </a>
</p>

## Abstract
Eyewire is a citizen science game where players map individual neurons in a mouse's retina. Using a relatively new method of deep learning called Bayesian Deep Learning, I create an agent to approximate the individual tasks (a.k.a. cubes) that human players complete as part of Eyewire. 3D images pose a huge challenge in memory/size restrictions, so methods dealing with those had to be engineered.  The architecture used in the model is a recurrent lightweight U-Net for 3D images, and showed extremely good results.

## Purpose
This project is designed as an introduction to Bayesian Deep Learning and an invitation to join citizen science efforts such as Eyewire.

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
* Aggregate: The ground truth of a task based on the combined responses of many players. Interchangeable with "Ground Truth"
* Harold: The mouse's name

### Eyewire

Eyewire is a citizen science game designed to collect data about individual neurons in a mouse's retina. Originally designed to create a segmentation algorithm for mapping the human brain, it has evolved into a complex data collection system based on the idea that getting ordinary people to play a game designed around annotating data creates a cost-effective way of annotating data that doesn't require the time and effort of people within the laboratories using the data nor the costly delegation of data annotation to services like Mechanical Turk. Currently, there are just over 3000 mouse neurons fully mapped within a 350×300×60 µm³ volume. At this point in time, the data collected by Eyewire has discovered 6 new specific neuron types and has made a new classification scheme for retinal neurons.

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
In Eyewire, players are given a 256x256x256 segmented 3D image (called a cube) containing a set of seed segments propagated from previous tasks. Players are to propagate (a.k.a. tracing) the seed and find the missing segments in the cube, giving Eyewire the nickname "adult coloring." As players play individual cubes, their traces are aggregated into a consensus and marked as the combined trace of the playerbase.

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
Here are a list of problems that are present in the current Eyewire Task Pipeline:

* As seen from the Eyewire Task Pipeline, 6-7 people are required at minimum to look at a single task. This is a very time consuming process used to anonate petabytes of data, resulting in less than 10,000 total neurons being traced over the course of 7 years. 

* The segmentation algorithm used to generate the cubes is, although quite robust, still rudimentary as it does not take into account the full range of data that is provided by the Eyewire Dataset. 

* The segmentation algorithm's errors propagate to the players. There are many instances of mergers through the dataset, causing players to mistakenly add additional segments to a trace. The default procedure is to remove all segments that contain significant mergers to avoid multiple segments being classified as part of 2 different neurons.

<p align="center">
  <img src="http://wiki.eyewire.org:88/images/thumb/9/9c/No_borders2.png/800px-No_borders2.png" title="Tracing mergers is the chaotic evil of Eyewire" />
</p>

## The Original Dataset

The Eyewire API contains several endpoints that are reachable with either an access code or an account with sufficient permissions.

Relevant endpoints:
```
http://eyewire.org/1.0/task/[:task_id:] # Get data about the task, used for getting volume id, filepath, seed, parent
http://eyewire.org/1.0/task/[:task_id:]/aggregate # Get data about the task's aggregate
http://eyewire.org/1.0/cell/[:cell_id:]/tasks # Get a list of tasks associated with a cell. Very useful in picking out tasks to gather data on
```

There is also a cell registry csv file circulated among the hobby devs within the community which I could parse through to get the available cells to download their data.

Using the task endpoint, there are related urls to download the 256x256x256 image stack of each volume and the lzma compressed 1D uint16 segmentation file of each volume.

## Challenges
Here are a list of challenges associated with this project

* 3D images are incredibly large and consume large amounts of memory. A 5 layer neural network transforming a 256³ image into an output of size 256³ does not fit on 12 GB of RAM.

* There are 2 million separate tasks across 56,300 volumes, across 3300 cells. Gathering the data is extremely time consuming and slows the servers hosting Eyewire, totaling 500 GB of data. The full E2198 Dataset is several TB.

* Images are stored as 256 individual 2D images, making file transfers and image loading an extremely costly operation.

* There is a subset of cells and images from the brain stem of a zebrafish that are not the same size as those of a mouse retina.

* External hard drive transfer speeds are capped so running scripts off it is time consuming. Windows also has this amazing feature that shuts off a hard drive if it uses too much power.

## Solutions Summary

* To remedy the cost of memory, I downsample both images and segmentation by a factor of 4 on each dimension to retain some detail. I am only looking for an approximation of the final annotation.

* Using a combination of asyncio and multiprocessing, I can set the transfer speed of the data gathered through the Eyewire API.

* After gathering the images, I merge them all into 3D numpy arrays and compress them using the lzma compression library.

* I remove all zebrafish cells from the data. Eyewire is already developing an agent for this called MSTY.

* I use my secondary laptop with lots of disk space to help preprocess some of the data.

## Solutions

### Downsample Images

Segmentations of volumes are downsampled to a size of 64x64x64 by taking the max segment count in each subvolume of 4x4x4. This causes some tasks to not have any segments in the seed, so we have to remove those tasks. 

Images are downsampled by taking the mean of each 4x4x4 block. This is done using skimage's downsample_local_mean function. 

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
    pass

if __name__ == "__main__":
  list = [things]
  loop = asyncio.get_event_loop()
  loop.run_until_complete(func(list))
```

Multiprocessing skeleton code
```python
def func_(val):
  # do things
  time.sleep(n)
  return result
  
list = [things]
with multiprocessing.Pool(np) as p:
  results = list(tqdm.tqdm(p.imap(func, list), total=len(list))
```

The asyncio is not controllable, so it is difficult to throttle the speed. Since the Eyewire API is connected to the game, HTTP requests slow down the game and data collection. This is when multiprocessing is a better alternative to asyncio. Asyncio and multiprocessing were used in the data gathering step and preprocessing steps.

tqdm is a library that displays pretty progress bars. I highly recommend it for any script dealing with large loops.

## The Final Dataset

After downsampling all the images/segmentations and removing tasks containing an empty seed, we are left with ~1.7 million data points across 56,326 3D images. Images are grayscale images of size 64x64x64, stored as lzma compressed 1D uint8 binary files. Segmentation files are lzma compressed 1D uint16 binary files. We also have two additional files used to store the data on individual tasks, task_data.json, and task_vol.json. These files are used to quickly get task data without needing to access the API, significantly speeding up training.

After splitting up the dataset, ~1.5 million tasks are used for training, and ~145,700 tasks are used as a testing dataset. When training, we randomly sample a quarter of the training dataset because it takes too much time (128 hrs) to train on 1.5 million tasks.  

### task_vol.json

This file is a json object containing the mapping of tasks to volume IDs to easily index the images from the data directory. One sample data point in this file looks like:

```json
{
  "588174": "123276"
}
```

### task_data.json

This file contains information about a task's seed, aggregate, and parent task. One sample data point in this file looks like:

```json
{
  "1415863": {
    "seed": [1473, 1528],
    "parent": 1415862,
    "aggregate": [1407, 1472, 1473, 1528]
  }
}
```

## Bayesian Deep Learning

Bayesian deep learning is a new method of deep learning that deals with the problem of uncertainty. Take a model that tries to classify dogs and cats. What would the model output if it came across this image?

<p align="center">
  <img src="https://github.com/kyle-dorman/bayesian-neural-network-blogpost/blob/master/blog_images/catdog.png?raw=true" title="Dogcat? Catdog? Let's just call it a monster" />
</p>

Ideally, there would be a 3rd category that has a label of neither dog nor cat, but adding that label would cost additional data collection. The solution is for the model to output both the classification and an uncertainty value. We can train the model with a loss function that penalizes uncertainty (u) by adding log(u) to the loss function when the original prediction is adjusted for u. 

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

The model architecture is a recurrent lightweight U-Net ("lwunet" for short). It consists of 2 Conv3D/ReLU/MaxPool layers, of which the outputs get concatenated to the 2 ConvTranspose3D/ReLU layers. The output of that is fed into two 1x1x1 Conv3D layers to calculate the predicted output and uncertainty of each pixel.

<p align="center">
  <img src="https://i.imgur.com/0lV3Si8.png" title="This is so lightweight, it can actually fit on 12 GB of GPU RAM" />
</p>

The input confidence is defaulted to 1 for the seed and 10 for everywhere else as it hasn't been explored yet. The hope for this model is that on each loop, it will explore the surroundings of the current seed and decide what belongs, what doesn't belong, and what it is still uncertain about. We also input p previous ground truths, of which the null equivalent is all zeros, so that the model gets previous cubes for extra context.

We loop n (default=4) number of times through the U-Net model plugging in the logits as the new seed and sigma as the new confidence. We hope the high confidence value on the initial iteration for non-seed voxels causes the model to ignore the initial values of 0 in the seed.

The model trained uses a previous size of 2 (trigram cubes) for additional information. 

## Results

On the test set, we get the following results:
```json
{
  "loss": 1.0000137143301788, 
  "precision": 0.9996331187884362, 
  "recall": 0.999607809239787, 
  "iou": 0.9994934872954645, 
  "seedless_precision": 0.7455424629880024, 
  "seedless_recall": 0.9998307692043708, 
  "seedless_iou": 0.9865256681720843,
  "count_guesses": 272979968, 
  "count_gt": 272978784, 
  "count_intersection": 272942176, 
  "count_union": 273016352, 
  "count_seedless_guesses": 78767688, 
  "count_seedless_gt": 78762184, 
  "count_seedless_intersection": 78731656, 
  "count_seedless_union": 78798056
}

Aggregate Metrics:
{
  "agg_precision": 0.999861557607040,
  "agg_recall": 0.999865894339979,
  "agg_iou": 0.999728309313868,
  "agg_seedless_precision": 0.999542553540482,
  "agg_seedless_recall": 0.999612402825193,
  "agg_seedless_iou": 0.999157339617616
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

Recall is the percentage of ground truth positive voxels that the model managed to predict.

<p align="center">
  <img src="https://i.imgur.com/Z1uDiuc.gif" title="Of all the arrows that hit the target, how many are yours?"/>
</p>

#### Intersection over Union (IOU)

IOU is the volumetric intersection of positive guesses and ground truth positive voxels divided by the union of the two.

<p align="center">
  <img src="https://i.imgur.com/HTWhRdF.gif" title="Over all the arrows you shot and arrows that hit the target, how many did you hit?"/>
</p>

#### Seedless Metrics

Seedless metrics are ignoring any voxel in either the ground truth or the guess that belonged to the seed. It should be easy to determine that the seed stays, so we ignore it.

#### Aggregate Metrics

Instead of averaging over all trials, we take the total counts and calculate the individual metrics. This will show a more accurate representation of the metrics.

## Analysis

### Uncertainty

First, I want to look at the uncertainty of differently shaded voxels, since there is a known phenomenon within the dataset called a "black spill." A black spill is a volume of voxels where the dye accidentally spilled into the surrounding area instead of just the cell membrane. These voxels can be extremely difficult to trace through because there is no indication of where membranes are.

Because there are actually very few voxels where uncertainty is more than the minimum of 1, I opt for a scatter plot to show where any uncertainties lie at all. This doesn't show the concentration of uncertainties, but there is a clear range where uncertainties lie.

<p align="center">
  <img src="https://github.com/BLimmie/eyewire_validator/blob/master/images/uncertainty.png?raw=true" title="To be honest, I expected cooler results than &quot;The model works really ****ing well&quot;"/>
</p>

Besides the anomaly of darkness = 0, we notice that the model only has uncertainty within a certain range of darker values and no uncertainty within much lighter values. This is somewhat strange since another potential area of error are membranes not being dyed, causing merger segments. I think the additional information of parent tasks makes the model certain about the direction of the neuron's growth.

This matches the theory that black spills cause uncertainty.

**Uncertainty seems a bit low. Can you explain that?**

The math behind the loss function makes small differences very important at both extremes of the scale. Dividing the logit output by a value of 1.002 can decrease the value of loss from a value of greater than 3 to around 1.5. When optimizing the model, this significant decrease in the loss function is where uncertainty plays the largest role, as we penalize larger uncertainties.

### Metrics Distribution

Note on the following graphs: All integer counts are on a logarithmic scale to show detail of less accurate points which can be misleading. Proportions as a decimal between 0 and 1 are not logarithmically scaled.

#### Precision/Recall/IOU Distribution

<p float="center">
  <img src="https://github.com/BLimmie/eyewire_validator/blob/master/images/precision.png?raw=true" width="32%" title="Sorry, there are too many of these graphs to create clever title texts" />
  <img src="https://github.com/BLimmie/eyewire_validator/blob/master/images/recall.png?raw=true" width="32%" title="" />
  <img src="https://github.com/BLimmie/eyewire_validator/blob/master/images/iou.png?raw=true" width="32%" title="" />
</p>

These distributions show that there are very few tasks where lwunet scores near 0. The scales of these graphs can be misleading, but less than 0.1% of data points have a recall below 0.95. The IOU distribution is a combination of precision and recall. Overall, the model is very accurate. There are a handful of tasks that have a seed but an empty GT as they were removed for being mergers. There are very few of these in the dataset, so they are left in for dataset noise, and they result in precisions of 0.

#### Seedless Distribution

<p float="center">
  <img src="https://github.com/BLimmie/eyewire_validator/blob/master/images/s_precision.png?raw=true" width="32%" title="This graph originally had the same number of 0's as 1's, but I changed how precision is calculated for 0/0" />
  <img src="https://github.com/BLimmie/eyewire_validator/blob/master/images/s_recall.png?raw=true" width="32%" title="" />
  <img src="https://github.com/BLimmie/eyewire_validator/blob/master/images/s_iou.png?raw=true" width="32%" title="So did this" />
</p>

Taking out the seeds shows a more interesting underlying distribution that more accurately tests lwunet on voxels outside the seed. We notice that there are significantly more tasks with a precision of 0. This can be explained. The segmentation algorithm used in Eyewire causes a significant number of mergers, big and small, and players are encouraged to avoid tracing these in either neuron to avoid what's known as a duplicate cube, where a segment appears in the aggregate of 2 or more tasks. So on many occassions, there are large chunks of voxels that are not part of the aggregate but part of the neuron. The increase in precision scores of 0 are due to mergers occurring in the seed, where no new segments were added to the aggregate, but there are voxels that do belong to the neuron. The overall decrease in precision can be due to small mergers in an otherwise larger trace. Recall does not show the same reduction, and IOU is a combination of the two scores.

#### Comparing General Metrics to Seedless Metrics

<p float="center">
  <img src="https://github.com/BLimmie/eyewire_validator/blob/master/images/sp_p_scatter.png?raw=true" width="32%" title="" />
  <img src="https://github.com/BLimmie/eyewire_validator/blob/master/images/sr_r_scatter.png?raw=true" width="32%" title="Ooga Booga" />
  <img src="https://github.com/BLimmie/eyewire_validator/blob/master/images/siou_iou_scatter.png?raw=true" width="32%" title="" />
</p>

Like our analysis in the seedless distributions in the section above, precision takes a large hit overall and can be explained. Recall shows a graph more consistent with our expectations, where there appears to be a line through (recall) = (seedless_recall). There are also a large number of tasks that increased their recall to 1, which means those tasks did not add additional segments to the seed, but lwunet didn't think all parts of the seed belonged to the ground truth even without a merger. IOU, again, is a combination of the two graphs.

#### Comparing Volume of GT to Precision/Recall

<p float="center">
  <img src="https://github.com/BLimmie/eyewire_validator/blob/master/images/gt_precision.png?raw=true" width="49%" title="" />
  <img src="https://github.com/BLimmie/eyewire_validator/blob/master/images/gt_recall.png?raw=true" width="49%" title="" />
</p>

As we expect, precision and recall are unstable metrics when there are fewer voxels in the aggregate. We notice a weird anomaly in recall where there are several hundred points with lower recall than the curve. I have a hunch this is due to downsampling the image and thus some aggregates' volumes becoming smaller due to the lack of information about nubs (small branches in a neuron, not extending past the cube). Nub filled tasks tend to have a neuron that extends through the cube, which results in gt volumes similar to that patch in recall.

#### Comparing Seedless Volume of GT to Seedless Precision/Recall

<p float="center">
  <img src="https://github.com/BLimmie/eyewire_validator/blob/master/images/gt_precision_seedless.png?raw=true" width="49%" title="" />
  <img src="https://github.com/BLimmie/eyewire_validator/blob/master/images/gt_recall_seedless.png?raw=true" width="49%" title="I would fix the scaling on this stupid graph, but that means modifying my code" />
</p>

Precision, like past analysis gets pulled down a lot due to mergers, especially in smaller traces. Recall does not show a noticeable difference in the graph. 

#### Comparing Volume Guessed to Correctly Guessed Volume

<p float="center">
  <img src="https://github.com/BLimmie/eyewire_validator/blob/master/images/guess_intersection.png?raw=true" width="49%" title="Put enough dots on a line, and you create, well, a line" />
  <img src="https://github.com/BLimmie/eyewire_validator/blob/master/images/guess_intersection_seedless.png?raw=true" width="49%" title="Oh no! The line broke :(" />
</p>

This is additional confirmation that lwunet is filling in the voxels not in the aggregate due to seed mergers. We can see that there is an almost perfect correlation between total ground truth volume and guessed volume, but removing the seed shows a pattern of data points with 0 additional voxels in the aggregate, even if lwunet thinks there should be more voxels in the aggregate.

### Loops = 1

What happens if we only loop through the model once?

```json
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

## Conclusion

In this project we introduce a lightweight U-Net that can be used on smaller 3D images. We also introduce the concept of Bayesian Deep Learning, which allows us to calculate the model's uncertainty. Using Bayesian Deep Learning on our lwunet, we manage to create a model that is able to propagate the seed of tasks in Eyewire to match as close as possible the trace of the neuron designated by the seed. We also realize how many mergers the current Eyewire segmentation algorithm actually creates, as removing them shows that lwunet appears to be less precise with its guesses. We also get confirmation that black spills cause lots of confusion in not only humans, but the model too. 

There is a nontrivial mismatch between how this model creates output and how Eyewire actually plays, so automation attempts will have to do additional research on how to convert downsampled images to the segments displayed in Eyewire.

## Future Work

* Switch model/loss to Monte Carlo Simulations and test
* Update with aggregate metrics for loop=1 (the p2.xlarge server used to run this is starting to empty my wallet)

## How to use the model independently

1. Import pytorch, numpy, and the model arch
```python
import torch
import numpy as np
from model.model import lwunet
```

2. Load the state on a machine that can access a CUDA GPU. CPU is highly not recommended and might cause problems when loading the model (Please create an issue if this happens. It should be fixed.)
```python
state_dict = torch.load('state_dict.pth')
```

3. Instantiate the model, load the weights, and prepare the model
```python
model = lwunet(3,4)

# load state dict
if config['n_gpu'] > 1:
    model = torch.nn.DataParallel(model)
model.load_state_dict(state_dict)

# prepare model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)
model.eval()
```

4. Gather inputs and feed into model
```python
def run(img, seed, prev, target=None):
    """
    inputs are np.ndarrays, and the batch size is n (1 unless running multiple inputs at once)
    Sizes:
    img [nx64x64x64]
    seed [nx64x64x64]
    prev [nx2x64x64x64]
    target [nx1x64x64x64] or None if there is no use for the aggregate
    """
    img = torch.tensor(img, dtype=torch.float)
    img = img.to(device)
    seed = torch.tensor(seed, dtype=torch.float)
    seed = seed.to(device)
    
    conf = np.where(seed==1, 1, 10)
    conf = torch.tensor(conf, dtype=torch.float)
    conf = conf.to(device)
    if target is not None:
        target = torch.tensor(target, dtype=torch.float)
        target = target.to(device)
    prev = torch.tensor(prev, dtype=torch.float)
    prev = prev.to(device)
    
    logits, sigma = model(img, seed, conf, prev)
    
    return logits, sigma
```

Prediction is based on the sign of logits.

See [downsample_images.py](data_scripts/downsample_images.py) and [downsample_segment.py](data_scripts/downsample_segment.py) for examples in how to downsample segmentation and images from 256x256x256 to 64x64x64.

See [create_data.py](modeling/tools/create_data.py) for examples in how to generate the image, seed, ground truth, and previous arrays.

5. Return to numpy array
```python
logits, sigma = logits.cpu().numpy(), sigma.cpu().numpy()
```

## Credits

* Eyewire: Providing the resources for this project
  * Amy Sterling: Connecting me to the development team, and for supporting me in building this project
  * Chris Jordan: Creating the OAuth API authentication for me to connect to the API without using a logged in browser and providing API documentation support
  * Doug Bland: Moral support
  * KrzysztofKruk: Providing the Cell Registry
* Princeton University - Seung Laboratory: Maintaining Eyewire to be an awesome citizen science game and for developers to work on my requests
* MIT: Building Eyewire
* Professor Yu-Xiang Wang: Providing insight into segmentation and how to create a model for the purposes of propagating a seed

<p align="center">
  <img src="https://i0.wp.com/blog.eyewire.org/wp-content/uploads/2018/03/ARTPRINT_ForScience_BLACK.jpg?resize=700%2C700&ssl=1" title="科学のために！" width = 70%>
</p>

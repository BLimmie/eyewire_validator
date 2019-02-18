# Eyewire Validation

## Abstract
Eyewire is a citizen science game where players map individual neurons in a rat's retina. Using a relatively new method of deep learning called Bayesian Deep Learning, I create an agent to approximate the individual tasks (a.k.a. cubes) that human players complete as part of Eyewire. The architecture used in the model is a recurrent encoder-decoder network for 3D images. 3D images pose a huge challenge in memory/size restrictions, so methods dealing with those had to be engineered.


## Summary of Eyewire and Terms

#### Terms
* Cube: A 3D image stack given to players as a task
* Trace: The act of propogating the seed to find missing segments within a cube
* Scythe: A high-ranking, established Eyewire player, given additional responsibilities of validating individual cubes
* Reap: Using a scythe's override powers to fix a cube
* Merger: A segment that contains volumes of two different neurons

#### The Eyewire Task Pipeline
In Eyewire, players are given a 256x256x256 segmented 3D image (called a cube) containing a set of seed segments propogated from previous tasks. Players are to propogate (a.k.a. tracing) the seed and find the missing segments in the cube, giving Eyewire the nickname "adult coloring." As players play individual cubes, their traces are aggregated into a consensus and marked as the combined trace of the playerbase.

![Eyewire: How to Play](http://wiki.eyewire.org:88/images/thumb/4/4a/1.png/800px-1.png)

After 3 players have played a single cube, the cube is then marked to be checked by more established players known as Scythes. Scythes have full override powers of tasks, being able to fix any mistake that the consensus has. It takes 2 Scythes to mark a cube as complete. After a cube is marked as complete and a whole neuron has been traced, 1-2 administrators (Eyewire/Seung Lab employees) are tasked with validating the full trace again to make sure there are no mistakes. 

#### The Eyewire Task Pipeline Summary
TL;DR
* 3 Players play a cube
* 2 Scythes validate a cube and reap it if necessary
* 1-2 Admins validate a whole neuron when it is done

## Motivation
Here are a list of problems that are present in the current Eyewire Task Pipeline

* As seen from the Eyewire Task Pipeline, 6-7 people are required at minimum to look at a single task. This is a very time consuming process used to anonate petabytes of data, resulting in less than 100,000 total neurons being traced over the course of 7 years. 

* The segmentation algorithm used to generate the cubes is, although quite robust, still rudimentary as it does not take into account the full range of data that is provided by the Eyewire Dataset. 

* The segmentation algorithm's errors propogate to the players. There are many instances of mergers through the dataset, causing players to mistakenly add additional segments to a trace. The default procedure is to remove all segments that contain significant mergers to avoid multiple segments being classified as part of 2 different neurons.

![Eyewire: Merger Example](http://wiki.eyewire.org:88/images/thumb/9/9c/No_borders2.png/800px-No_borders2.png)


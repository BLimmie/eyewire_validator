# Eyewire Validation

## Abstract
Eyewire is a citizen science game where players map individual neurons in a rat's retina. Using a relatively new method of deep learning called Bayesian Deep Learning, I create an agent to approximate the individual tasks (a.k.a. cubes) that human players complete as part of Eyewire. The architecture used in the model is a recurrent encoder-decoder network for 3D images. 3D images pose a huge challenge in memory/size restrictions, so methods dealing with those had to be engineered.


## Summary of Eyewire and Terms

#### The Eyewire Task Pipeline
In Eyewire, players are given a 256x256x256 segmented 3D image containing a set of seed segments propogated from previous tasks. 

![Eyewire: How to Play](http://wiki.eyewire.org:88/images/thumb/4/4a/1.png/1200px-1.png)

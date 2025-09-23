CUDA Path Tracer
================

**University of Pennsylvania, CIS 565: GPU Programming and Architecture, Project 2**

* Jacqueline (Jackie) Li
  * [LinkedIn](https://www.linkedin.com/in/jackie-lii/), [personal website](https://sites.google.com/seas.upenn.edu/jacquelineli/home), [Instagram](https://www.instagram.com/sagescherrytree/), etc.
* Tested on: Windows 10, 11th Gen Intel(R) Core(TM) i7-11800H @ 2.30GHz, NVIDIA GeForce RTX 3060 Laptop GPU (6 GB)

# Basic Diffuse Pathtracer

## Bugs During Implementation

Missing a sampling dimension from sphere to hemisphere cosine sampling caused an artifact that did not allow throughput to accumulate properly on the vertical. 

## Stream Compaction Optimization for Base Pathtracer.

Used thrust/partition to read in number of currently active paths (light rays) to device, then partition them based on whether or not path is currently active. Partition will sort the rays into currently active in the front, and terminated rays after, and it returns the end pointer to the reduced dev_path array containing currently active rays. 

Stream compaction reduces iteration time from ~880 ms/frame to ~500 ms/frame.

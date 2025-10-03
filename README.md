CUDA Path Tracer
================

**University of Pennsylvania, CIS 565: GPU Programming and Architecture, Project 2**

* Jacqueline (Jackie) Li
  * [LinkedIn](https://www.linkedin.com/in/jackie-lii/), [personal website](https://sites.google.com/seas.upenn.edu/jacquelineli/home), [Instagram](https://www.instagram.com/sagescherrytree/), etc.
* Tested on: Windows 10, 11th Gen Intel(R) Core(TM) i7-11800H @ 2.30GHz, NVIDIA GeForce RTX 3060 Laptop GPU (6 GB)

# Basic Diffuse Pathtracer

## Features Implemented

1. Basic Pathtracer w/ diffuse shading.
2. Stream Compaction on terminated rays.
3. Material sorting.
4. Stochastic sampled antialiasing. 
5. Reflective and refractive materials (specular). (2)
6. Depth of field. (2)
7. Mesh loading, OBJ. (2)
8. BVH acceleration. (4)
9. Textures. (3 (?))
10. Bump maps. (Currently implementing)
10. Microfacet Materials. (To be implemented)
11. Subsurface scattering. (To be implemented)

## Bugs During Implementation

#### Diffuse Sampling Implementation

Missing a sampling dimension from sphere to hemisphere cosine sampling caused an artifact that did not allow throughput to accumulate properly on the vertical. 

#### Reflection and Transmissive Material Implementation

Bugs in this domain come primarily from frame of reference errors pertaining to my pathSegment.ray.direction.

##### Reflection Bug

Normals mapped to worldspace not being used in reflecting ray properly.

##### A Curious Transmission Bug

Upon normalizing my normal input from the ray as well as pathSegment.ray.direction, I get rings around the sphere, that simultaneously look cool and creepy. 

Here is the code to replicate the bug:
```
// Specular Transmission.
__device__ glm::vec3 sampleFSpecularTrans(
    const glm::vec3& albedo,
    const glm::vec3& normal,
    const glm::vec3& wo,
    const float& IOR,
    glm::vec3& wiW) {

    // Index of refraction of glass.
    float etaA = 1.f;
    float etaB = IOR;

    glm::vec3 N = glm::normalize(normal);
    glm::vec3 woNormal = glm::normalize(wo); // The ray's direction as read from pathSegment in ScatterRay.

    // Test z coordinate of wo (if z coord > 0, then about to enter transmissive surface.)
    bool entering = wo.z > 0.f;
    float etaI = entering ? etaA : etaB;
    float etaT = entering ? etaB : etaA;

    float eta = etaI / etaT;

    wiW = glm::normalize(glm::refract(-woNormal, N, eta));

    if (glm::length(wiW) < EPSILON) {
        return glm::vec3(0.0f);
    }

    return albedo;
}
```

##### Trasmission Epsilon Bug

Beware of EPSILON values that are too small... When I changed the EPSILON value to 0.0001f from 0.000001f, the glass material started to work. I presume it is b/c the ray is stepping in too small and getting stuck in the reflective material, resulting in the strange albedo throughput b/c the rest of the rays in the scene are never influencing it.

Also, an interesting matter to point out is that within the sampling function, the smaller EPSILON value blurs out the light caustic more than comparing w/ the larger EPSILON.

## Stream Compaction Optimization for Base Pathtracer

Used thrust/partition to read in number of currently active paths (light rays) to device, then partition them based on whether or not path is currently active. Partition will sort the rays into currently active in the front, and terminated rays after, and it returns the end pointer to the reduced dev_path array containing currently active rays. 

Stream compaction reduces iteration time from ~880 ms/frame to ~500 ms/frame.

## BVH Acceleration

##### References

* [Jacco How to Build a BVH] (https://jacco.ompf2.com/2022/04/13/how-to-build-a-bvh-part-1-basics/)
* [Sebastian Lague Coding Adventure: More Raytracing!] (https://www.youtube.com/watch?v=C1H4zIiCOaI)

| ![](img/cornell.2025-09-30_21-48-04z.1071samp.png) | ![](img/cornell.2025-09-30_22-05-25z.1074samp.png) |
|:--:|:--:|
|Running phat_phuck.obj with no basic BVH |Running phat_phuck.obj with basic BVH |
| application average: 732.856 ms/frame, 1.4 fps | application average: 99.365 ms/frame, 10.0 fps |
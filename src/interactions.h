#pragma once

#include "sceneStructs.h"

#include <glm/glm.hpp>

#include <thrust/random.h>

// Utility functions.

inline __device__ float CosTheta(const glm::vec3& w) { return w.z; }
inline __device__ float AbsCosTheta(const glm::vec3& w) { return glm::abs(w.z); }

inline __device__ void coordinateSystem(const glm::vec3& in_vec1, glm::vec3& out_vec2, glm::vec3& out_vec3) {
    if (abs(in_vec1.x) > abs(in_vec1.y))
        out_vec2 = glm::vec3(-in_vec1.z, 0, in_vec1.x) / sqrt(in_vec1.x * in_vec1.x + in_vec1.z * in_vec1.z);
    else
        out_vec2 = glm::vec3(0, in_vec1.z, -in_vec1.y) / sqrt(in_vec1.y * in_vec1.y + in_vec1.z * in_vec1.z);
    out_vec3 = cross(in_vec1, out_vec2);
}

inline __device__ glm::mat3 LocalToWorld(const glm::vec3& nor) {
    glm::vec3 tan;
    glm::vec3 bit;
    coordinateSystem(nor, tan, bit);
    return glm::mat3(tan, bit, nor);
}

// Transpose of localToWorld.
inline __device__ glm::mat3 WorldToLocal(const glm::vec3& nor) {
    return glm::transpose(LocalToWorld(nor));
}

// CHECKITOUT
/**
 * Computes a cosine-weighted random direction in a hemisphere.
 * Used for diffuse lighting.
 */
__host__ __device__ glm::vec3 calculateRandomDirectionInHemisphere(
    glm::vec3 normal, 
    thrust::default_random_engine& rng);

// Additional sampling methods.
__device__ glm::vec3 squareToDiskConcentric(
    const glm::vec2& xi);

__device__ glm::vec3 squareToHemisphereCosine(
    const glm::vec2& xi);

__device__ float squareToHemisphereCosinePDF(
    const glm::vec3& sample);

__device__ glm::vec3 f_diffuse(glm::vec3& albedo);

// Sample functions to call from ScatterRay.
__device__ glm::vec3 sampleFDiffuse(
    const glm::vec3& albedo, 
    const glm::vec3& normal, 
    glm::vec3& wiW, 
    float& pdf,
    thrust::default_random_engine& rng);

__device__ glm::vec3 sampleFSpecularRefl(
    const glm::vec3& albedo,
    const glm::vec3& normal,
    const glm::vec3& wo,
    glm::vec3& wiW);

// Helper functions Refract and FaceForward for transmissive material.
__device__ bool Refract(
    const glm::vec3& wi,
    const glm::vec3 normal,
    const float eta,
    glm::vec3& wt);

__device__ glm::vec3 FaceForward(
    const glm::vec3& normal,
    const glm::vec3& v);

__device__ glm::vec3 sampleFSpecularTrans(
    const glm::vec3& albedo,
    glm::vec3& normal,
    const glm::vec3& wo,
    const float& IOR,
    glm::vec3& wiW);

// Helper functions FresnelDielectricEval for glass transmissive material.
__device__ float FresnelDielectricEval(
    float cosThetaI,
    float IOR);

__device__ float FresnelSchlick(
    float cosTheta,
    float etaI,
    float etaT);

__device__ glm::vec3 sampleFGlass(
    const glm::vec3& albedo,
    glm::vec3& normal,
    const glm::vec3& wo,
    const float& IOR,
    glm::vec3& wiW,
    thrust::default_random_engine& rng);

// Microfacets.

inline __device__ bool SameHemisphere(
    const glm::vec3& w, 
    const glm::vec3& wp) {
    return w.z * wp.z > 0;
}

inline __device__ float Cos2Theta(
    const glm::vec3& w)
{
    return w.z * w.z;
}

inline __device__ float Sin2Theta(
    const glm::vec3& w)
{
    return glm::max(0.f, 1.f - Cos2Theta(w));
}

inline __device__ float Tan2Theta(
    const glm::vec3& w)
{
    return Sin2Theta(w) / Cos2Theta(w);
}


inline __device__ float SinTheta(
    const glm::vec3& w)
{
    return glm::sqrt(Sin2Theta(w));
}

inline __device__ float TanTheta(
    const glm::vec3& w)
{
    return SinTheta(w) / CosTheta(w);
}

inline __device__ float CosPhi(
    const glm::vec3& w)
{
    float sinTheta = SinTheta(w);
    return (sinTheta == 0) ? 0 : glm::clamp(w.x / sinTheta, -1.f, 1.f);
}

inline __device__ float SinPhi(
    const glm::vec3& w)
{
    float sinTheta = SinTheta(w);
    return (sinTheta == 0) ? 0 : glm::clamp(w.y / sinTheta, -1.f, 1.f);
}

inline __device__ float Cos2Phi(
    const glm::vec3& w)
{
    return CosPhi(w) * CosPhi(w);
}

inline __device__ float Sin2Phi(
    const glm::vec3& w)
{
    return SinPhi(w) * SinPhi(w);
}

__device__ glm::vec3 sampleWH(
    const glm::vec3& wo,
    const float& roughness,
    thrust::default_random_engine& rng);

__device__ float TrowbridgeReitzD(
    const glm::vec3& wh,
    const float& roughness);

__device__ float lambda(
    const glm::vec3& w,
    const float& roughness);

__device__ float TrowbridgeReitzG(
    const glm::vec3& wo,
    const glm::vec3& wi,
    const float& roughness);

__device__ float TrowbridgeReitzPdf(
    const glm::vec3& wo,
    const glm::vec3& wh,
    const float& roughness);

__device__ glm::vec3 fMicrofacetRefl(
    const glm::vec3& albedo,
    const glm::vec3& wo,
    const glm::vec3& wi,
    const float& IOR,
    const float& roughness);

__device__ glm::vec3 sampleFMicrofacetRefl(
    const glm::vec3& albedo,
    const glm::vec3& normal,
    const glm::vec3& wo,
    const float& IOR,
    const float& roughness,
    glm::vec3& wiW,
    float& pdf,
    thrust::default_random_engine& rng);

/**
 * Scatter a ray with some probabilities according to the material properties.
 * For example, a diffuse surface scatters in a cosine-weighted hemisphere.
 * A perfect specular surface scatters in the reflected ray direction.
 * In order to apply multiple effects to one surface, probabilistically choose
 * between them.
 *
 * The visual effect you want is to straight-up add the diffuse and specular
 * components. You can do this in a few ways. This logic also applies to
 * combining other types of materias (such as refractive).
 *
 * - Always take an even (50/50) split between a each effect (a diffuse bounce
 *   and a specular bounce), but divide the resulting color of either branch
 *   by its probability (0.5), to counteract the chance (0.5) of the branch
 *   being taken.
 *   - This way is inefficient, but serves as a good starting point - it
 *     converges slowly, especially for pure-diffuse or pure-specular.
 * - Pick the split based on the intensity of each material color, and divide
 *   branch result by that branch's probability (whatever probability you use).
 *
 * This method applies its changes to the Ray parameter `ray` in place.
 * It also modifies the color `color` of the ray in place.
 *
 * You may need to change the parameter list for your purposes!
 */
__device__ void scatterRay(
    PathSegment& pathSegment,
    glm::vec3 intersect,
    glm::vec3 normal,
    const Material& m,
    thrust::default_random_engine& rng);
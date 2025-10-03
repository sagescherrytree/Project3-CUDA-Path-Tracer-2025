#pragma once

#include <cuda_runtime.h>

#include "glm/glm.hpp"

#include <string>
#include <vector>

#define BACKGROUND_COLOR (glm::vec3(0.0f))

enum GeomType
{
    SPHERE,
    CUBE
};

struct Ray
{
    glm::vec3 origin;
    glm::vec3 direction;
};

struct Geom
{
    enum GeomType type;
    int materialid;
    glm::vec3 translation;
    glm::vec3 rotation;
    glm::vec3 scale;
    glm::mat4 transform;
    glm::mat4 inverseTransform;
    glm::mat4 invTranspose;
};

struct Material
{
    glm::vec3 color;
    struct
    {
        float exponent;
        glm::vec3 color;
    } specular;
    float hasReflective;
    float hasRefractive;
    float indexOfRefraction;
    float emittance;

    bool hasTexture = false;
    int textureID = -1;
};

// Struct for textures.
struct Texture 
{
    int width = 0;
    int height = 0;
    int channels;
    unsigned char* data;
};

// Vertex struct for loading obj files.
struct Vertex
{
    int materialID;
    glm::vec3 position;
    glm::vec3 normal;
    glm::vec2 uv; // Vertex uv for tex coords.
};

// Triangle struct for BVH.
struct Triangle {
    Vertex v1;
    Vertex v2;
    Vertex v3;
    glm::vec3 centroid;
    int materialID;
};

struct AABB {
    glm::vec3 min = glm::vec3(FLT_MAX);
    glm::vec3 max = glm::vec3(-FLT_MAX);
};

struct BVHNode {
    AABB aabb;
	int left;
    int right;
    int start;
    int triCount; // Num triangles in leaf.
};

struct Camera
{
    glm::ivec2 resolution;
    glm::vec3 position;
    glm::vec3 lookAt;
    glm::vec3 view;
    glm::vec3 up;
    glm::vec3 right;
    glm::vec2 fov;
    glm::vec2 pixelLength;

    // Depth of field params.
    float aperture;
    float focalDist;
};

struct RenderState
{
    Camera camera;
    unsigned int iterations;
    int traceDepth;
    std::vector<glm::vec3> image;
    std::string imageName;
};

struct PathSegment
{
    Ray ray;
    glm::vec3 color = glm::vec3(0.f, 0.f, 0.f); // I.e. throughput.
    int pixelIndex;
    int remainingBounces;
};

// Struct for stream compaction condition.
struct PathAlive {
    __host__ __device__
        bool operator()(const PathSegment& p) const {
        return (p.remainingBounces > 0);
    }
};

// Use with a corresponding PathSegment to do:
// 1) color contribution computation
// 2) BSDF evaluation: generate a new ray
struct ShadeableIntersection
{
  float t;
  glm::vec3 surfaceNormal;
  int materialId;
  glm::vec2 uv; // For texture mapping.
};

// Struct for material sorting.
struct CompareMat {
    __host__ __device__ bool operator()(const ShadeableIntersection& a, const ShadeableIntersection& b)
    {
        return a.materialId < b.materialId;
    }
};
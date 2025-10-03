#include "pathtrace.h"

#include <cstdio>
#include <cuda.h>
#include <cmath>
#include <thrust/execution_policy.h>
#include <thrust/random.h>
#include <thrust/remove.h>
#include <thrust/device_vector.h>
#include <thrust/partition.h>

#include "sceneStructs.h"
#include "scene.h"
#include "glm/glm.hpp"
#include "glm/gtx/norm.hpp"
#include "utilities.h"
#include "intersections.h"
#include "interactions.h"

#define ERRORCHECK 1
#define STREAM_COMPACTION 1
#define MATERIAL_SORTING 1
#define NAIVE_MESH_LOADING 0
#define BVH_ACCELERATION 1

#define FILENAME (strrchr(__FILE__, '/') ? strrchr(__FILE__, '/') + 1 : __FILE__)
#define checkCUDAError(msg) checkCUDAErrorFn(msg, FILENAME, __LINE__)
void checkCUDAErrorFn(const char* msg, const char* file, int line)
{
#if ERRORCHECK
    cudaDeviceSynchronize();
    cudaError_t err = cudaGetLastError();
    if (cudaSuccess == err)
    {
        return;
    }

    fprintf(stderr, "CUDA error");
    if (file)
    {
        fprintf(stderr, " (%s:%d)", file, line);
    }
    fprintf(stderr, ": %s: %s\n", msg, cudaGetErrorString(err));
#ifdef _WIN32
    getchar();
#endif // _WIN32
    exit(EXIT_FAILURE);
#endif // ERRORCHECK
}

__host__ __device__
thrust::default_random_engine makeSeededRandomEngine(int iter, int index, int depth)
{
    int h = utilhash((1 << 31) | (depth << 22) | iter) ^ utilhash(index);
    return thrust::default_random_engine(h);
}

//Kernel that writes the image to the OpenGL PBO directly.
__global__ void sendImageToPBO(uchar4* pbo, glm::ivec2 resolution, int iter, glm::vec3* image)
{
    int x = (blockIdx.x * blockDim.x) + threadIdx.x;
    int y = (blockIdx.y * blockDim.y) + threadIdx.y;

    if (x < resolution.x && y < resolution.y)
    {
        int index = x + (y * resolution.x);
        glm::vec3 pix = image[index];

        glm::ivec3 color;
        color.x = glm::clamp((int)(pix.x / iter * 255.0), 0, 255);
        color.y = glm::clamp((int)(pix.y / iter * 255.0), 0, 255);
        color.z = glm::clamp((int)(pix.z / iter * 255.0), 0, 255);

        // Each thread writes one pixel location in the texture (textel)
        pbo[index].w = 0;
        pbo[index].x = color.x;
        pbo[index].y = color.y;
        pbo[index].z = color.z;
    }
}

static Scene* hst_scene = NULL;
static GuiDataContainer* guiData = NULL;
std::vector<cudaTextureObject_t> texObjects;
static glm::vec3* dev_image = NULL;
static Geom* dev_geoms = NULL;
static Material* dev_materials = NULL;
static PathSegment* dev_paths = NULL;
static ShadeableIntersection* dev_intersections = NULL;
// TODO: static variables for device memory, any extra info you need, etc
// ...
// Mesh loading variables.
static Vertex* dev_vertices = NULL;
static Triangle* dev_triangles = NULL;
static int* dev_triIndices = NULL; // Indices into triangle array for BVH.
static BVHNode* dev_bvhNodes = NULL;
static Texture* dev_textures = NULL;

// GPU textures.
static cudaTextureObject_t* dev_texObjects = NULL;
static std::vector<cudaArray_t> dev_texArrays;

void InitDataContainer(GuiDataContainer* imGuiData)
{
    guiData = imGuiData;
}

// Helper function to fill in texture objects. Referenced from 
// https://github.com/NVIDIA/cuda-samples/blob/master/Samples/0_Introduction/simpleTexture/simpleTexture.cu.
void createTextureObjects(int i) 
{
    cudaResourceDesc resDesc = {};
    memset(&resDesc, 0, sizeof(resDesc));
    resDesc.resType = cudaResourceTypeArray;
    resDesc.res.array.array = dev_texArrays[i];

    // Texture object params.
    cudaTextureDesc texDesc = {};
    texDesc.normalizedCoords = 1;
	texDesc.filterMode = cudaFilterModeLinear;
    texDesc.addressMode[0] = cudaAddressModeWrap;
	texDesc.addressMode[1] = cudaAddressModeWrap;
	texDesc.readMode = cudaReadModeNormalizedFloat;

    // Create texture object.
    cudaTextureObject_t texObject = 0;
	cudaCreateTextureObject(&texObject, &resDesc, &texDesc, NULL);

    // Allocate result in device memory.
	cudaMemcpy(&dev_texObjects[i], &texObject, sizeof(cudaTextureObject_t), cudaMemcpyHostToDevice);
}


void pathtraceInit(Scene* scene)
{
    hst_scene = scene;

    const Camera& cam = hst_scene->state.camera;
    const int pixelcount = cam.resolution.x * cam.resolution.y;

    cudaMalloc(&dev_image, pixelcount * sizeof(glm::vec3));
    cudaMemset(dev_image, 0, pixelcount * sizeof(glm::vec3));

    cudaMalloc(&dev_paths, pixelcount * sizeof(PathSegment));

    cudaMalloc(&dev_geoms, scene->geoms.size() * sizeof(Geom));
    cudaMemcpy(dev_geoms, scene->geoms.data(), scene->geoms.size() * sizeof(Geom), cudaMemcpyHostToDevice);

    cudaMalloc(&dev_materials, scene->materials.size() * sizeof(Material));
    cudaMemcpy(dev_materials, scene->materials.data(), scene->materials.size() * sizeof(Material), cudaMemcpyHostToDevice);

    cudaMalloc(&dev_intersections, pixelcount * sizeof(ShadeableIntersection));
    cudaMemset(dev_intersections, 0, pixelcount * sizeof(ShadeableIntersection));

    // TODO: initialize any extra device memeory you need
	cudaMalloc(&dev_vertices, scene->vertices.size() * sizeof(Vertex));
    cudaMemcpy(dev_vertices, scene->vertices.data(), scene->vertices.size() * sizeof(Vertex), cudaMemcpyHostToDevice);

    cudaMalloc(&dev_triangles, scene->triangles.size() * sizeof(Triangle));
    cudaMemcpy(dev_triangles, scene->triangles.data(), scene->triangles.size() * sizeof(Triangle), cudaMemcpyHostToDevice);
    
	cudaMalloc(&dev_triIndices, scene->triIndices.size() * sizeof(int));
	cudaMemcpy(dev_triIndices, scene->triIndices.data(), scene->triIndices.size() * sizeof(int), cudaMemcpyHostToDevice);

	cudaMalloc(&dev_bvhNodes, scene->bvhNodes.size() * sizeof(BVHNode));
	cudaMemcpy(dev_bvhNodes, scene->bvhNodes.data(), scene->bvhNodes.size() * sizeof(BVHNode), cudaMemcpyHostToDevice);

    cudaMalloc(&dev_textures, scene->textures.size() * sizeof(Texture));
    cudaMemcpy(dev_textures, scene->textures.data(), scene->textures.size() * sizeof(Texture), cudaMemcpyHostToDevice);

    // Allocate texture objects on GPU. 
	cudaMalloc(&dev_texObjects, scene->textures.size() * sizeof(cudaTextureObject_t));
	dev_texArrays.resize(scene->textures.size());

    for (int i = 0; i < scene->textures.size(); i++) {
        // Channel.
        cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<uchar4>();

        int channels = scene->textures[i].channels;

        if (channels == 1) {
            channelDesc = cudaCreateChannelDesc<unsigned char>();
        }
        else if (channels == 4) {
            channelDesc = cudaCreateChannelDesc<uchar4>();
        }
        else if (channels == 3) {
            // 3-channel images are padded to 4 channels
            channelDesc = cudaCreateChannelDesc<uchar4>();
        }
        else {
            printf("Unsupported texture channel count: %d\n", channels);
            continue;
        }

		cudaMallocArray(&dev_texArrays[i], &channelDesc, scene->textures[i].width, scene->textures[i].height);
        cudaMemcpyToArray(dev_texArrays[i], 
            0, 0, 
            scene->textures[i].data,
            scene->textures[i].width * scene->textures[i].height * scene->textures[i].channels * sizeof(unsigned char),
			cudaMemcpyHostToDevice);

        createTextureObjects(i);
    }

    checkCUDAError("pathtraceInit");
}

void pathtraceFree()
{
    cudaFree(dev_image);  // no-op if dev_image is null
    cudaFree(dev_paths);
    cudaFree(dev_geoms);
    cudaFree(dev_materials);
    cudaFree(dev_intersections);
    // TODO: clean up any extra device memory you created
	cudaFree(dev_vertices);
	cudaFree(dev_triangles);
    cudaFree(dev_triIndices);
    cudaFree(dev_bvhNodes);
	cudaFree(dev_textures);

    for (int i = 0; i < texObjects.size(); i++) {
		cudaDestroyTextureObject(texObjects[i]);
		cudaFreeArray(dev_texArrays[i]);
    }

    checkCUDAError("pathtraceFree");
}

__device__ glm::vec3 sampleAperture(float apertureRadius, thrust::default_random_engine& rng) {
    thrust::uniform_real_distribution<float> u01(0, 1);
    float r = apertureRadius * sqrt(u01(rng));
    float theta = 2 * PI * u01(rng);
    // Lens offset.
    return glm::vec3(r * cos(theta), r * sin(theta), 0.0f);
}

/**
* Generate PathSegments with rays from the camera through the screen into the
* scene, which is the first bounce of rays.
*
* Antialiasing - add rays for sub-pixel sampling
* motion blur - jitter rays "in time"
* lens effect - jitter ray origin positions based on a lens
*/
__global__ void generateRayFromCamera(Camera cam, int iter, int traceDepth, PathSegment* pathSegments)
{
    int x = (blockIdx.x * blockDim.x) + threadIdx.x;
    int y = (blockIdx.y * blockDim.y) + threadIdx.y;

    if (x < cam.resolution.x && y < cam.resolution.y) {
        int index = x + (y * cam.resolution.x);
        PathSegment& segment = pathSegments[index];

        //segment.ray.origin = cam.position;
        //segment.color = glm::vec3(1.0f, 1.0f, 1.0f);

        thrust::default_random_engine rng = makeSeededRandomEngine(iter, index, 0);
        thrust::uniform_real_distribution<float> u01(0, 1);

        // Create jitter.
        float jitterX = u01(rng);
        float jitterY = u01(rng);

        // TODO: implement antialiasing by jittering the ray
        //segment.ray.direction = glm::normalize(cam.view
        //    - cam.right * cam.pixelLength.x * ((float)x + jitterX - (float)cam.resolution.x * 0.5f)
        //    - cam.up * cam.pixelLength.y * ((float)y + jitterY - (float)cam.resolution.y * 0.5f)
        //);

        // Redo ray direction calc w/ depth of field.
        glm::vec3 pixelPoint = cam.view
            - cam.right * cam.pixelLength.x * ((float)x + jitterX - (float)cam.resolution.x * 0.5f)
            - cam.up * cam.pixelLength.y * ((float)y + jitterY - (float)cam.resolution.y * 0.5f);
        glm::vec3 rayDir = glm::normalize(pixelPoint);

        // Compute focal point.
        glm::vec3 focalPoint = cam.position + rayDir * cam.focalDist;

        // Depth of field: sample aperture.
        glm::vec3 apertureOffset = sampleAperture(cam.aperture, rng);

        // Updated ray calcs based on depth of field aperture.
        segment.ray.origin = cam.position + apertureOffset;
        segment.color = glm::vec3(1.f);
        segment.ray.direction = glm::normalize(focalPoint - segment.ray.origin);

        segment.pixelIndex = index;
        segment.remainingBounces = traceDepth;
    }
}

// TODO:
// computeIntersections handles generating ray intersections ONLY.
// Generating new rays is handled in your shader(s).
// Feel free to modify the code below.
__global__ void computeIntersections(
    int depth,
    int num_paths,
    PathSegment* pathSegments,
    Geom* geoms,
    int geoms_size,
    ShadeableIntersection* intersections,
    Vertex* vertices,
    Triangle* triangles,
    int* triIndices,
    BVHNode* bvhNodes,
    int numVerts)
{
    int path_index = blockIdx.x * blockDim.x + threadIdx.x;

    if (path_index < num_paths)
    {
        PathSegment pathSegment = pathSegments[path_index];

        float t;
        glm::vec3 intersect_point;
        glm::vec3 normal;
        float t_min = FLT_MAX;
        int hit_geom_index = -1; // -1 for cube, sphere, -2 for triangle, 0 o.w.
        int hit_material_id = -1; 
        bool outside = true;
        
        glm::vec2 hitUV = glm::vec2(0.0f, 0.0f);
        int hitTriIndex = -1;

        glm::vec3 tmp_intersect;
        glm::vec3 tmp_normal;

        glm::vec3 tmp_dpdu;
        glm::vec3 tmp_dpdv;

        // naive parse through global geoms

        for (int i = 0; i < geoms_size; i++)
        {
            Geom& geom = geoms[i];

            if (geom.type == CUBE)
            {
                t = boxIntersectionTest(geom, pathSegment.ray, tmp_intersect, tmp_normal, outside);
            }
            else if (geom.type == SPHERE)
            {
                t = sphereIntersectionTest(geom, pathSegment.ray, tmp_intersect, tmp_normal, outside);
            }
            // TODO: add more intersection tests here... triangle? metaball? CSG?

            // Compute the minimum t from the intersection tests to determine what
            // scene geometry object was hit first.
            if (t > 0.0f && t_min > t)
            {
                t_min = t;
                hit_geom_index = geoms[i].materialid;
                intersect_point = tmp_intersect;
                normal = tmp_normal;

                hitUV = glm::vec2(0.0f, 0.0f);
                hitTriIndex = -1;
                hit_material_id = hit_geom_index;
            }
        }

#if NAIVE_MESH_LOADING
        for (int i = 0; i < numVerts; i += 3) {
            Vertex& v0 = vertices[i + 0];
            Vertex& v1 = vertices[i + 1];
            Vertex& v2 = vertices[i + 2];

            // Call intersectTriangle for mesh triangle intersection.
            float t, u, v;
            if (intersectTriangle(pathSegment.ray, v0.position, v1.position, v2.position, t, u, v)) {
                if (t > 0.0f && t < t_min) {
                    t_min = t;
                    hit_geom_index = -2; 
                    intersect_point = pathSegment.ray.origin + t * pathSegment.ray.direction;

                    // Interpolate normals (if available).
                    if (glm::length(v0.normal) < 1e-6f || glm::length(v1.normal) < 1e-6f || glm::length(v2.normal) < 1e-6f) {
                        normal = glm::normalize(glm::cross(v1.position - v0.position, v2.position - v0.position));
                    }
                    else {
                        normal = glm::normalize((1 - u - v) * v0.normal + u * v1.normal + v * v2.normal);
                    }

                    // Interpolate UVs.
                    hitUV = (1 - u - v) * v0.uv + u * v1.uv + v * v2.uv;

                    // Store material directly from vertex (assuming per-triangle consistent).
                    hit_material_id = v0.materialID;
                }
            }
        }
#endif

#if BVH_ACCELERATION
        // Material id for BVH.
		int material_id_bvh = -1;
        float t_bvh = bvhMeshIntersectionTest(
            pathSegment.ray,
            tmp_intersect,
            tmp_normal,
            outside,
            material_id_bvh,
            hitUV,
            hitTriIndex,
            triangles,
            triIndices,
            bvhNodes,
            tmp_dpdu,
            tmp_dpdv);

        if (t_bvh > 0.0f && t_bvh < t_min) {
            t_min = t_bvh;
            hit_geom_index = -2;
            intersect_point = tmp_intersect;
            normal = tmp_normal;
            hit_material_id = material_id_bvh;
        }
#endif

        if (hit_geom_index == -1)
        {
            intersections[path_index].t = -1.0f;
        }
        else
        {
            if (glm::dot(pathSegment.ray.direction, normal) > 0.0f) {
                normal = -normal; // flip to make it face the ray origin.
            }
            // The ray hits something
            intersections[path_index].t = t_min;
            intersections[path_index].materialId = (hit_geom_index == -2) ? hit_material_id : hit_geom_index;
            intersections[path_index].surfaceNormal = normal;
            // only write UV for triangle hits
            if (hit_geom_index == -2) {
                intersections[path_index].uv = hitUV;
                intersections[path_index].dpdu = tmp_dpdu;
                intersections[path_index].dpdv = tmp_dpdv;
            }
            else {
                // set a default for non-triangle hits (optional)
                intersections[path_index].uv = glm::vec2(0.0f, 0.0f);
            }
        }
    }
}

// LOOK: "fake" shader demonstrating what you might do with the info in
// a ShadeableIntersection, as well as how to use thrust's random number
// generator. Observe that since the thrust random number generator basically
// adds "noise" to the iteration, the image should start off noisy and get
// cleaner as more iterations are computed.
//
// Note that this shader does NOT do a BSDF evaluation!
// Your shaders should handle that - this can allow techniques such as
// bump mapping.
__global__ void shadeFakeMaterial(
    int iter,
    int num_paths,
    ShadeableIntersection* shadeableIntersections,
    PathSegment* pathSegments,
    Material* materials)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_paths)
    {
        ShadeableIntersection intersection = shadeableIntersections[idx];
        if (intersection.t > 0.0f) // if the intersection exists...
        {
          // Set up the RNG
          // LOOK: this is how you use thrust's RNG! Please look at
          // makeSeededRandomEngine as well.
            thrust::default_random_engine rng = makeSeededRandomEngine(iter, idx, 0);
            thrust::uniform_real_distribution<float> u01(0, 1);

            Material material = materials[intersection.materialId];
            glm::vec3 materialColor = material.color;

            // If the material indicates that the object was a light, "light" the ray
            if (material.emittance > 0.0f) {
                pathSegments[idx].color *= (materialColor * material.emittance);
            }
            // Otherwise, do some pseudo-lighting computation. This is actually more
            // like what you would expect from shading in a rasterizer like OpenGL.
            // TODO: replace this! you should be able to start with basically a one-liner
            else {
                float lightTerm = glm::dot(intersection.surfaceNormal, glm::vec3(0.0f, 1.0f, 0.0f));
                pathSegments[idx].color *= (materialColor * lightTerm) * 0.3f + ((1.0f - intersection.t * 0.02f) * materialColor) * 0.7f;
                pathSegments[idx].color *= u01(rng); // apply some noise because why not
            }
            // If there was no intersection, color the ray black.
            // Lots of renderers use 4 channel color, RGBA, where A = alpha, often
            // used for opacity, in which case they can indicate "no opacity".
            // This can be useful for post-processing and image compositing.
        }
        else {
            pathSegments[idx].color = glm::vec3(0.0f);
        }
    }
}

// Helper function to sample texture.
__device__ glm::vec3 sampleTexture(int texID, glm::vec2 uv, cudaTextureObject_t* texObjects, int numTexture) {
    if (texID < 0 || texID >= numTexture) {
        return glm::vec3(1.0f, 0.0f, 1.0f);
    }

    float4 col = tex2D<float4>(texObjects[texID], uv.x, 1.f - uv.y);
    return glm::vec3(col.x, col.y, col.z);
}

// helper: sample height (single channel) assuming texture is RGBA where height stored in .x
__device__ float sampleHeight(int texID, glm::vec2 uv, cudaTextureObject_t* texObjects, int numTexture) {
    if (texID < 0 || texID >= numTexture) return 0.0f;
    float4 col = tex2D<float4>(texObjects[texID], uv.x, 1.f - uv.y);
    return col.x;
}

__global__ void kernShadeMaterialProper(
    int iter,
    int num_paths,
    ShadeableIntersection* shadeableIntersections,
    PathSegment* pathSegments,
    Material* materials,
    cudaTextureObject_t* dev_texObjects,
    Texture* dev_textures,
    int numTextures)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_paths)
    {
        // Check if ray terminated.
        if (pathSegments[idx].remainingBounces <= 0) {
            return;
        }

        ShadeableIntersection intersection = shadeableIntersections[idx];


        if (intersection.t > 0.0f) // if the intersection exists...
        {

            Material material = materials[intersection.materialId];
            glm::vec3 materialColor = material.color;

            // If textured, sample texture.
            if (material.hasTexture) {
				glm::vec3 texColor = sampleTexture(material.textureID, intersection.uv, dev_texObjects, numTextures);
                materialColor = texColor;
            }
            material.color = materialColor;

            // If the material indicates that the object was a light, "light" the ray
            if (material.emittance > 0.0f) {
                // Is light.
                pathSegments[idx].color *= (materialColor * material.emittance);
                pathSegments[idx].remainingBounces = 0;
            }
            
            else {
                // Use thrust to scatter ray.
                thrust::default_random_engine rng = makeSeededRandomEngine(iter, pathSegments[idx].pixelIndex, pathSegments[idx].remainingBounces);
                thrust::uniform_real_distribution<float> u01(0, 1);

                // Get the ray
                Ray& ray = pathSegments[idx].ray;

                // Get the intersection point
                glm::vec3 intersect = ray.origin + ray.direction * intersection.t;

                glm::vec3 ng = intersection.surfaceNormal;  // geometric / interpolated normal
                glm::vec3 dpdu = intersection.dpdu;
                glm::vec3 dpdv = intersection.dpdv;
                glm::vec2 uv = intersection.uv;

                glm::vec3 shadingNormal = ng; // default
                if (material.hasBumpMap && material.bumpID >= 0) {
                    int bumpTexID = material.bumpID;

                    int texWidth = dev_textures[bumpTexID].width;
                    int texHeight = dev_textures[bumpTexID].height;

                    float du = 1.0f / float(texWidth);
                    float dv = 1.0f / float(texHeight);

                    float h = sampleHeight(bumpTexID, uv, dev_texObjects, numTextures);
                    float hU = sampleHeight(bumpTexID, glm::vec2(uv.x + du, uv.y), dev_texObjects, numTextures);
                    float hV = sampleHeight(bumpTexID, glm::vec2(uv.x, uv.y + dv), dev_texObjects, numTextures);

                    float dhdu = (hU - h) / du;
                    float dhdv = (hV - h) / dv;

                    float scale = material.bumpScale; // e.g., 0.05

                    // perturb dp/du and dp/dv
                    glm::vec3 dpdu_p = dpdu + scale * dhdu * ng;
                    glm::vec3 dpdv_p = dpdv + scale * dhdv * ng;

                    // compute new normal
                    shadingNormal = glm::normalize(glm::cross(dpdu_p, dpdv_p));

                    // keep it in same hemisphere as original
                    if (glm::dot(shadingNormal, ng) < 0.0f)
                        shadingNormal = -shadingNormal;
                }

                scatterRay(pathSegments[idx], intersect, shadingNormal, material, rng);
            }
            // If there was no intersection, color the ray black.
            // Lots of renderers use 4 channel color, RGBA, where A = alpha, often
            // used for opacity, in which case they can indicate "no opacity".
            // This can be useful for post-processing and image compositing.
        }
        else {
            pathSegments[idx].color = glm::vec3(0.0f);
            pathSegments[idx].remainingBounces = 0;
        }
    }
}

// Add the current iteration's output to the overall image
__global__ void finalGather(int nPaths, glm::vec3* image, PathSegment* iterationPaths)
{
    int index = (blockIdx.x * blockDim.x) + threadIdx.x;

    if (index < nPaths)
    {
        PathSegment iterationPath = iterationPaths[index];
        image[iterationPath.pixelIndex] += iterationPath.color;
    }
}

/**
 * Wrapper for the __global__ call that sets up the kernel calls and does a ton
 * of memory management
 */
void pathtrace(uchar4* pbo, int frame, int iter)
{
    const int traceDepth = hst_scene->state.traceDepth;
    const Camera& cam = hst_scene->state.camera;
    const int pixelcount = cam.resolution.x * cam.resolution.y;

    // 2D block for generating ray from camera
    const dim3 blockSize2d(8, 8);
    const dim3 blocksPerGrid2d(
        (cam.resolution.x + blockSize2d.x - 1) / blockSize2d.x,
        (cam.resolution.y + blockSize2d.y - 1) / blockSize2d.y);

    // 1D block for path tracing
    const int blockSize1d = 128;

    ///////////////////////////////////////////////////////////////////////////

    // Recap:
    // * Initialize array of path rays (using rays that come out of the camera)
    //   * You can pass the Camera object to that kernel.
    //   * Each path ray must carry at minimum a (ray, color) pair,
    //   * where color starts as the multiplicative identity, white = (1, 1, 1).
    //   * This has already been done for you.
    // * For each depth:
    //   * Compute an intersection in the scene for each path ray.
    //     A very naive version of this has been implemented for you, but feel
    //     free to add more primitives and/or a better algorithm.
    //     Currently, intersection distance is recorded as a parametric distance,
    //     t, or a "distance along the ray." t = -1.0 indicates no intersection.
    //     * Color is attenuated (multiplied) by reflections off of any object
    //   * TODO: Stream compact away all of the terminated paths.
    //     You may use either your implementation or `thrust::remove_if` or its
    //     cousins.
    //     * Note that you can't really use a 2D kernel launch any more - switch
    //       to 1D.
    //   * TODO: Shade the rays that intersected something or didn't bottom out.
    //     That is, color the ray by performing a color computation according
    //     to the shader, then generate a new ray to continue the ray path.
    //     We recommend just updating the ray's PathSegment in place.
    //     Note that this step may come before or after stream compaction,
    //     since some shaders you write may also cause a path to terminate.
    // * Finally, add this iteration's results to the image. This has been done
    //   for you.

    // TODO: perform one iteration of path tracing

    generateRayFromCamera<<<blocksPerGrid2d, blockSize2d>>>(cam, iter, traceDepth, dev_paths);
    checkCUDAError("generate camera ray");

    int depth = 0;
    PathSegment* dev_path_end = dev_paths + pixelcount;
    int num_paths = dev_path_end - dev_paths;

    // --- PathSegment Tracing Stage ---
    // Shoot ray into scene, bounce between objects, push shading chunks

    bool iterationComplete = false;
    while (!iterationComplete)
    {
        // clean shading chunks
        cudaMemset(dev_intersections, 0, pixelcount * sizeof(ShadeableIntersection));

        // tracing
        dim3 numblocksPathSegmentTracing = (num_paths + blockSize1d - 1) / blockSize1d;
        computeIntersections<<<numblocksPathSegmentTracing, blockSize1d>>> (
            depth,
            num_paths,
            dev_paths,
            dev_geoms,
            hst_scene->geoms.size(),
            dev_intersections,
            dev_vertices,
            dev_triangles,
            dev_triIndices,
            dev_bvhNodes,
			hst_scene->vertices.size()
        );
        checkCUDAError("trace one bounce");
        cudaDeviceSynchronize();
        depth++;

        // TODO:
        // --- Shading Stage ---
        // Shade path segments based on intersections and generate new rays by
        // evaluating the BSDF.
        // Start off with just a big kernel that handles all the different
        // materials you have in the scenefile.
        // TODO: compare between directly shading the path segments and shading
        // path segments that have been reshuffled to be contiguous in memory.


#if MATERIAL_SORTING
        // Sort the paths by material via stream compaction (thrust).
        thrust::device_ptr<ShadeableIntersection> dev_intersections_ptr(dev_intersections);
        thrust::device_ptr<PathSegment> dev_paths_ptr(dev_paths);
        thrust::stable_sort_by_key(dev_intersections_ptr, dev_intersections_ptr + num_paths, dev_paths_ptr, CompareMat());
#endif

        //shadeFakeMaterial << <numblocksPathSegmentTracing, blockSize1d >> > (
        kernShadeMaterialProper <<<numblocksPathSegmentTracing, blockSize1d>>>(
            iter,
            num_paths,
            dev_intersections,
            dev_paths,
            dev_materials,
            dev_texObjects,
            dev_textures,
			hst_scene->textures.size()
        );
        cudaDeviceSynchronize();

#if STREAM_COMPACTION
        // Call thrust for stream compaction.
        thrust::device_ptr<PathSegment> dev_thrust_paths(dev_paths);
        thrust::device_ptr<PathSegment> dev_new_ends =
            thrust::stable_partition(thrust::device, dev_thrust_paths, dev_thrust_paths + num_paths, PathAlive());

        num_paths = dev_new_ends.get() - dev_paths;
#endif

        // TODO: should be based off stream compaction results.
        if (num_paths == 0) {
            iterationComplete = true;
        }
        if (depth >= traceDepth) {
            iterationComplete = true;
        }

        if (guiData != NULL)
        {
            guiData->TracedDepth = depth;
        }
    }

    // Assemble this iteration and apply it to the image
    dim3 numBlocksPixels = (pixelcount + blockSize1d - 1) / blockSize1d;
    finalGather<<<numBlocksPixels, blockSize1d>>>(pixelcount, dev_image, dev_paths);

    ///////////////////////////////////////////////////////////////////////////

    // Send results to OpenGL buffer for rendering
    sendImageToPBO<<<blocksPerGrid2d, blockSize2d>>>(pbo, cam.resolution, iter, dev_image);

    // Retrieve image from GPU
    cudaMemcpy(hst_scene->state.image.data(), dev_image,
        pixelcount * sizeof(glm::vec3), cudaMemcpyDeviceToHost);

    checkCUDAError("pathtrace");
}

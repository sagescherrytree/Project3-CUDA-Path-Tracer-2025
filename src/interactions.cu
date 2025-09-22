#include "interactions.h"

#include "utilities.h"

#include <thrust/random.h>

__host__ __device__ glm::vec3 calculateRandomDirectionInHemisphere(
    glm::vec3 normal,
    thrust::default_random_engine &rng)
{
    thrust::uniform_real_distribution<float> u01(0, 1);

    float up = sqrt(u01(rng)); // cos(theta)
    float over = sqrt(1 - up * up); // sin(theta)
    float around = u01(rng) * TWO_PI;

    // Find a direction that is not the normal based off of whether or not the
    // normal's components are all equal to sqrt(1/3) or whether or not at
    // least one component is less than sqrt(1/3). Learned this trick from
    // Peter Kutz.

    glm::vec3 directionNotNormal;
    if (abs(normal.x) < SQRT_OF_ONE_THIRD)
    {
        directionNotNormal = glm::vec3(1, 0, 0);
    }
    else if (abs(normal.y) < SQRT_OF_ONE_THIRD)
    {
        directionNotNormal = glm::vec3(0, 1, 0);
    }
    else
    {
        directionNotNormal = glm::vec3(0, 0, 1);
    }

    // Use not-normal direction to generate two perpendicular directions
    glm::vec3 perpendicularDirection1 =
        glm::normalize(glm::cross(normal, directionNotNormal));
    glm::vec3 perpendicularDirection2 =
        glm::normalize(glm::cross(normal, perpendicularDirection1));

    return up * normal
        + cos(around) * over * perpendicularDirection1
        + sin(around) * over * perpendicularDirection2;
}

// Implementation for additional sampling.
__device__ glm::vec3 squareToDiskConcentric(const glm::vec2& xi) {
    float x;
    float y;
    if (xi.x == 0.f && xi.y == 0.f) {
        x = 0.f;
        y = 0.f;
    }
    else {
        float theta = 0.f;
        float radius = 1.f;
        float a = (2.f * xi.x) - 1.f;
        float b = (2.f * xi.y) - 1.f;

        if ((a * a) > (b * b)) {
            radius *= a;
            theta = PI_OVER_FOUR * (b / a);
        }
        else {
            radius *= b;
            theta = PI_OVER_TWO - (PI_OVER_FOUR * (a / b));
        }
        x = radius * glm::cos(theta);
        y = radius * glm::sin(theta);
    }

    return glm::vec3(x, y, 0.f);
}

__device__ glm::vec3 squareToHemisphereCosine(const glm::vec2& xi) {
    glm::vec3 disk = squareToDiskConcentric(xi);
    float z = glm::sqrt(glm::max(0.f, 1.f - (disk.x * disk.x)));
    return glm::vec3(disk.x, disk.y, z);
}

__device__ float squareToHemisphereCosinePDF(const glm::vec3& sample) {
    float cosTheta = glm::dot(sample, glm::vec3(0.f, 0.f, 1.f));
    return cosTheta * INV_PI;
}

__device__ glm::vec3 f_diffuse(const glm::vec3& albedo) {
    return albedo * INV_PI;
}

// Diffuse material sampling.
__device__ glm::vec3 sampleFDiffuse(
    const glm::vec3& albedo,
    const glm::vec3& normal,
    glm::vec3& wiW,
    float& pdf,
    thrust::default_random_engine& rng) {

    // Generate random number first.
    thrust::uniform_real_distribution<float> u01(0, 1);
    const glm::vec2 xi = glm::vec2(u01(rng), u01(rng));

    glm::vec3 wi = squareToHemisphereCosine(xi);
    glm::mat3 worldSpace = LocalToWorld(normal);
    wiW = glm::normalize(worldSpace * wi);
    pdf = squareToHemisphereCosinePDF(wi);
    return f_diffuse(albedo);
}

__host__ __device__ void scatterRay(
    PathSegment & pathSegment,
    glm::vec3 intersect,
    glm::vec3 normal,
    const Material &m,
    thrust::default_random_engine &rng)
{
    // TODO: implement this.
    // A basic implementation of pure-diffuse shading will just call the
    // calculateRandomDirectionInHemisphere defined above.

    // Establish sampling vector, bsdf (material base), and pdf.
    glm::vec3 wiW;
    glm::vec3 bsdf;
    float pdf;

    // Set bsdf.
    bsdf = m.color / PI;

    // Sampled vector.
    wiW = calculateRandomDirectionInHemisphere(normal, rng);
    
    // Calculate pdf.
    float cosTheta = glm::dot(wiW, normal);
    pdf = glm::cos(glm::acos(cosTheta)) / PI;
    
    // Le ray.
    // Basic diffuse mat implementation.
    pathSegment.ray.direction = glm::normalize(wiW);
    pathSegment.color = bsdf * glm::abs(glm::dot(wiW, normal)) / pdf;
    pathSegment.ray.origin = intersect + pathSegment.ray.direction * EPSILON;

    // Subtract number of bounces.
    pathSegment.remainingBounces -= 1;

    // Terminate any rays that never reach a light
    if (pathSegment.remainingBounces == 0)
    {
        pathSegment.color = glm::vec3(0.0f);
    }
}

__device__ void scatterRay_F(
    PathSegment& pathSegment,
    glm::vec3 intersect,
    glm::vec3 normal,
    const Material& m,
    thrust::default_random_engine& rng)
{
    // Establish sampling vector, bsdf (material base), and pdf.
    glm::vec3 wiW;
    glm::vec3 bsdf_diffuse;
    float pdf;

    // Call different types of material:
    // Diffuse for testing.
    bsdf_diffuse = sampleFDiffuse(m.color, normal, wiW, pdf, rng);

    // Le ray.
    pathSegment.ray.direction = glm::normalize(wiW);
    pathSegment.ray.origin = intersect + pathSegment.ray.direction * EPSILON;

    // AbsDot term.
    float cosTheta = glm::max(0.0f, glm::dot(normal, wiW));

    // Throughput accum.
    if (pdf > 0.0f) {
        pathSegment.color *= (bsdf_diffuse * cosTheta) / pdf;
    }
    else {
        pathSegment.remainingBounces = 0;
        return;
    }

    // Subtract number of bounces.
    pathSegment.remainingBounces -= 1;
}

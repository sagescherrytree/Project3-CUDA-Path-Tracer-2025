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
// Range: 0-1.
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
    float z = glm::sqrt(glm::max(0.f, 1.0f - (disk.x * disk.x) - (disk.y * disk.y)));
    return glm::vec3(disk.x, disk.y, z);
}

__device__ float squareToHemisphereCosinePDF(const glm::vec3& sample) {
    return sample.z / PI;
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

// Specular Reflection.
__device__ glm::vec3 sampleFSpecularRefl(
    const glm::vec3& albedo,
    const glm::vec3& normal,
    const glm::vec3& wo,
    glm::vec3& wiW) {
    wiW = glm::reflect(wo, normal);
    return albedo;
}

__device__ bool Refract(
    const glm::vec3& wi,
    const glm::vec3 normal,
    const float eta,
    glm::vec3& wt) {
    // Compute cos(theta) using Snell's Law.
    float cosThetaI = glm::dot(normal, wi);
    float sin2ThetaI = glm::max(0.f, float(1.f - (cosThetaI * cosThetaI)));
    float sin2ThetaT = eta * eta * sin2ThetaI;

    // Total internal reflection.
    if (sin2ThetaT >= 1) {
        return false;
    }
    float cosThetaT = glm::sqrt(1.f - sin2ThetaT);
    wt = eta * -wi + (eta * cosThetaI - cosThetaT) * normal;
    return true;
}

__device__ glm::vec3 FaceForward(
    const glm::vec3& normal,
    const glm::vec3& v) {
    return glm::dot(normal, v) < 0.f ? -normal : normal;
}

// Specular Transmission.
__device__ glm::vec3 sampleFSpecularTrans(
    const glm::vec3& albedo,
    glm::vec3& normal,
    const glm::vec3& wo,
    const float& IOR,
    glm::vec3& wiW) {

    // Index of refraction of glass.
    float etaA = 1.f;
    float etaB = IOR;
    
    // The ray's direction as read from pathSegment in ScatterRay.

    // Test z coordinate of wo (if z coord > 0, then about to enter transmissive surface.)
    bool entering = glm::dot(wo, normal) < 0.f;
    float etaI = entering ? etaA : etaB;
    float etaT = entering ? etaB : etaA;

    float eta = etaI / etaT;

    normal = entering ? normal : -normal;
    wiW = glm::refract(wo, normal, eta);

    if (glm::length(wiW) < EPSILON) {
        return glm::vec3(0.0f);
    }

    float eta2 = (eta * eta);
    return albedo * eta2;
}

// Glass material.
// Fresnel dielectric coefficient calc.

__device__ float FresnelDielectricEval(float cosThetaI, float IOR) {
    float etaI = 1.f;
    float etaT = IOR;
    cosThetaI = glm::clamp(cosThetaI, -1.f, 1.f);

    if (cosThetaI > 0.f) {
        float temp = etaI;
        etaI = etaT;
        etaT = temp;
    }
    cosThetaI = glm::abs(cosThetaI);

    float sinThetaI = glm::sqrt(glm::max(0.f, 1.f - cosThetaI * cosThetaI));
    float sinThetaT = etaI / etaT * sinThetaI;
    float cosThetaT = glm::sqrt(glm::max(0.f, 1.f - sinThetaT * sinThetaT));
    float Rparl = ((etaT * cosThetaI) - (etaI * cosThetaT)) /
        ((etaT * cosThetaI) + (etaI * cosThetaT));
    float Rperp = ((etaI * cosThetaI) - (etaT * cosThetaT)) /
        ((etaI * cosThetaI) + (etaT * cosThetaT));
    
    return (Rparl * Rparl + Rperp * Rperp) * 0.5f;
}

// Fresnel-Schlick.
__device__ float FresnelSchlick(float cosTheta, float etaI, float etaT) {
    float R0 = (etaI - etaT) / (etaI + etaT);
    R0 = R0 * R0;
    return R0 + (1.0f - R0) * powf(1.0f - cosTheta, 5.0f);
}

// Specular glass.
__device__ glm::vec3 sampleFGlass(
    const glm::vec3& albedo,
    glm::vec3& normal,
    const glm::vec3& wo,
    const float& IOR,
    glm::vec3& wiW,
    thrust::default_random_engine& rng) {

    // Generate random number first.
    thrust::uniform_real_distribution<float> u01(0, 1);

    float random = u01(rng);

    float cosTheta = glm::dot(wo, normal);
    float fresnel = FresnelDielectricEval(cosTheta, IOR);

    if (random < fresnel) {
        wiW = glm::reflect(wo, normal);
        return albedo; // Reflection.
    }
    else {
        glm::vec3 T = sampleFSpecularTrans(albedo, normal, wo, IOR, wiW);
        if (glm::length(wiW) < EPSILON) {
            // Total internal reflection.
            wiW = glm::reflect(wo, normal);
            return albedo;
        }
        else {
            return T;
        }
    }
}

// Updated scatterRay to call all sampling materials.
__device__ void scatterRay(
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
    float pdf = 1.0f;

    // -- Glass --
    if (m.hasRefractive > 0.0f && m.hasReflective > 0.0f) {
        bsdf = sampleFGlass(m.color, normal, pathSegment.ray.direction, m.indexOfRefraction, wiW, rng);
        pathSegment.ray.direction = glm::normalize(wiW);
        pathSegment.ray.origin = intersect + normal * EPSILON;

        // Throughput accum.
        pathSegment.color *= bsdf;
    }

    // -- Reflective --
    else if (m.hasReflective > 0.0f) {
        bsdf = sampleFSpecularRefl(m.color, normal, pathSegment.ray.direction, wiW);
        pathSegment.ray.direction = glm::normalize(wiW);
        pathSegment.ray.origin = intersect + normal * EPSILON;
        pathSegment.color *= bsdf;
    }

    // -- Transmissive --
    else if (m.hasRefractive > 0.0f) {
        bsdf = sampleFSpecularTrans(m.color, normal, pathSegment.ray.direction, m.indexOfRefraction, wiW);
        pathSegment.ray.direction = glm::normalize(wiW);
        pathSegment.ray.origin = intersect + normal * EPSILON;
        pathSegment.color *= bsdf;
    }

    // -- Diffuse --
    else {
        bsdf = sampleFDiffuse(m.color, normal, wiW, pdf, rng);
        // Le ray.
        // Basic diffuse mat implementation.
        pathSegment.ray.direction = glm::normalize(wiW);
        pathSegment.ray.origin = intersect + normal * EPSILON;

        // Throughput accum.
        float cosTheta = glm::max(0.0f, glm::dot(normal, wiW));
        pathSegment.color *= (bsdf * cosTheta) / pdf;
    }

    // Subtract number of bounces.
    pathSegment.remainingBounces -= 1;
}

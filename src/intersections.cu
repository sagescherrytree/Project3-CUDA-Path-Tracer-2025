#include "intersections.h"

__host__ __device__ float boxIntersectionTest(
    Geom box,
    Ray r,
    glm::vec3 &intersectionPoint,
    glm::vec3 &normal,
    bool &outside)
{
    Ray q;
    q.origin    =                multiplyMV(box.inverseTransform, glm::vec4(r.origin   , 1.0f));
    q.direction = glm::normalize(multiplyMV(box.inverseTransform, glm::vec4(r.direction, 0.0f)));

    float tmin = -1e38f;
    float tmax = 1e38f;
    glm::vec3 tmin_n;
    glm::vec3 tmax_n;
    for (int xyz = 0; xyz < 3; ++xyz)
    {
        float qdxyz = q.direction[xyz];
        /*if (glm::abs(qdxyz) > 0.00001f)*/
        {
            float t1 = (-0.5f - q.origin[xyz]) / qdxyz;
            float t2 = (+0.5f - q.origin[xyz]) / qdxyz;
            float ta = glm::min(t1, t2);
            float tb = glm::max(t1, t2);
            glm::vec3 n;
            n[xyz] = t2 < t1 ? +1 : -1;
            if (ta > 0 && ta > tmin)
            {
                tmin = ta;
                tmin_n = n;
            }
            if (tb < tmax)
            {
                tmax = tb;
                tmax_n = n;
            }
        }
    }

    if (tmax >= tmin && tmax > 0)
    {
        outside = true;
        if (tmin <= 0)
        {
            tmin = tmax;
            tmin_n = tmax_n;
            outside = false;
        }
        intersectionPoint = multiplyMV(box.transform, glm::vec4(getPointOnRay(q, tmin), 1.0f));
        normal = glm::normalize(multiplyMV(box.invTranspose, glm::vec4(tmin_n, 0.0f)));
        return glm::length(r.origin - intersectionPoint);
    }

    return -1;
}

__host__ __device__ float sphereIntersectionTest(
    Geom sphere,
    Ray r,
    glm::vec3 &intersectionPoint,
    glm::vec3 &normal,
    bool &outside)
{
    float radius = .5;

    glm::vec3 ro = multiplyMV(sphere.inverseTransform, glm::vec4(r.origin, 1.0f));
    glm::vec3 rd = glm::normalize(multiplyMV(sphere.inverseTransform, glm::vec4(r.direction, 0.0f)));

    Ray rt;
    rt.origin = ro;
    rt.direction = rd;

    float vDotDirection = glm::dot(rt.origin, rt.direction);
    float radicand = vDotDirection * vDotDirection - (glm::dot(rt.origin, rt.origin) - powf(radius, 2));
    if (radicand < 0)
    {
        return -1;
    }

    float squareRoot = sqrt(radicand);
    float firstTerm = -vDotDirection;
    float t1 = firstTerm + squareRoot;
    float t2 = firstTerm - squareRoot;

    float t = 0;
    if (t1 < 0 && t2 < 0)
    {
        return -1;
    }
    else if (t1 > 0 && t2 > 0)
    {
        t = glm::min(t1, t2);
        outside = true;
    }
    else
    {
        t = glm::max(t1, t2);
        outside = false;
    }

    glm::vec3 objspaceIntersection = getPointOnRay(rt, t);

    intersectionPoint = multiplyMV(sphere.transform, glm::vec4(objspaceIntersection, 1.f));
    normal = glm::normalize(multiplyMV(sphere.invTranspose, glm::vec4(objspaceIntersection, 0.f)));

    return glm::length(r.origin - intersectionPoint);
}

// Triangle intersection.
__host__ __device__ bool intersectTriangle(
    const Ray& r,
    const glm::vec3& v0,
    const glm::vec3& v1,
    const glm::vec3& v2,
    float& tOut, float& uOut, float& vOut) 
{
    glm::vec3 edge1 = v1 - v0;
    glm::vec3 edge2 = v2 - v0;
    glm::vec3 pvec = glm::cross(r.direction, edge2);
    float det = glm::dot(edge1, pvec);
    if (fabs(det) < BABY_EPSILON) {
        return false;
    }
    float invDet = 1.0f / det;

    glm::vec3 tvec = r.origin - v0;
    float u = glm::dot(tvec, pvec) * invDet;
    if (u < 0.0f || u > 1.0f) {
        return false;
    }

    glm::vec3 qvec = glm::cross(tvec, edge1);
    float v = glm::dot(r.direction, qvec) * invDet;
    if (v < 0.0f || (u + v) > 1.0f) return false;

    float t = glm::dot(edge2, qvec) * invDet;
    if (t <= BABY_EPSILON) {
        return false;
    }

    tOut = t; uOut = u; vOut = v;
    return true;
}

// BVH intersection test.
__host__ __device__ float bvhMeshIntersectionTest(
    Ray r,
    glm::vec3& intersectionPoint,
    glm::vec3& normal,
    bool& outside,
    int& materialID,
    // Access verts anc centroid through triangle struct.
    // B/c I read in .obj as just Vertices, do not know if I need Vertex struct or Triangle struct?
    Triangle* triangles,
    int* triIndices,
    BVHNode* nodes)
{
    float t_hit = FLT_MAX;
    bool hitAnything = false;

    int stack[64];
    int stackPtr = 0;
    stack[stackPtr++] = 0;

    while (stackPtr > 0) {
		int nodeIdx = stack[--stackPtr];
        BVHNode& node = nodes[nodeIdx];

        // Ray intersection test.
        if (!aabbIntersectionTest(node.aabb, r)) {
            continue; // No intersection, skip node.
		}

        // Leaf node.
        if (node.triCount > 0 && node.start >= 0) {
            // Means node is leaf.
            for (int i = 0; i < node.triCount; i++) {
                int triIndex = triIndices[node.start + i];
                Triangle& tri = triangles[triIndex];

				Vertex v0 = tri.v1;
                Vertex v1 = tri.v2;
                Vertex v2 = tri.v3;

                float t, u, v;
                if (intersectTriangle(r, v0.position, v1.position, v2.position, t, u, v)) {
                    if (t < t_hit && t > 0.0f) {

						hitAnything = true;
                        t_hit = t;

                        // Intersection point.
                        intersectionPoint = r.origin + t * r.direction;

                        // Normals.
                        if (glm::length(v0.normal) < 1e-6f || glm::length(v1.normal) < 1e-6f || glm::length(v2.normal) < 1e-6f) {
                            normal = glm::normalize(glm::cross(v1.position - v0.position, v2.position - v0.position));
                        }
                        else {
                            normal = glm::normalize((1 - u - v) * v0.normal + u * v1.normal + v * v2.normal);
                        }

                        outside = glm::dot(r.direction, normal) < 0.0f;
                        materialID = v0.materialID;
                    }
                }
            }
        }
        else {
            // Internal node, push children to stack.
            if (node.left >= 0) {
                stack[stackPtr++] = node.left;
            }
            if (node.right >= 0) {
                stack[stackPtr++] = node.right;
			}
        }
    }

    return hitAnything ? t_hit : -1.f;
}

// Helper function to test ray-AABB intersection.
__host__ __device__ bool aabbIntersectionTest(
    const AABB& aabb,
    const Ray& ray) 
{
    float t_min = -FLT_MAX;
    float t_max = FLT_MAX;
    for (int i = 0; i < 3; ++i) {
        float dir = ray.direction[i];
        float origin = ray.origin[i];

        if (glm::abs(dir) < 0.00001f) {
            if (origin < aabb.min[i] || origin > aabb.max[i]) {
                return false;
            }
        }
        else {
            float t1 = (aabb.min[i] - origin) / dir;
            float t2 = (aabb.max[i] - origin) / dir;

            if (t1 > t2) {
                // Swap b/c within bounds.
                float temp = t1;
                t1 = t2;
                t2 = temp;
            }
            if (t1 > t_min) {
                t_min = t1;
            }
            if (t2 < t_max) {
                t_max = t2;
            }
            if (t_min > t_max) {
                // No overlap.
                return false;
            }
        }
    }
    return t_max >= t_min && t_max > 0.f;
}
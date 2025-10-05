#include "scene.h"

#include "utilities.h"

#include <glm/gtc/matrix_inverse.hpp>
#include <glm/gtx/string_cast.hpp>
#include "json.hpp"

#define TINYOBJLOADER_IMPLEMENTATION
#include "tiny_obj_loader.h"

#include "stb_image.h"

#include <fstream>
#include <iostream>
#include <string>
#include <unordered_map>

using namespace std;
using json = nlohmann::json;

Scene::Scene(string filename)
{
    cout << "Reading scene from " << filename << " ..." << endl;
    cout << " " << endl;
    auto ext = filename.substr(filename.find_last_of('.'));
    if (ext == ".json")
    {
        loadFromJSON(filename);
        return;
    }
    else
    {
        cout << "Couldn't read from " << filename << endl;
        exit(-1);
    }
}

Scene::~Scene()
{
    for (auto& tex : textures)
    {
        stbi_image_free(tex.data);
    }
}

void Scene::loadFromJSON(const std::string& jsonName)
{
    std::ifstream f(jsonName);
    json data = json::parse(f);
    const auto& materialsData = data["Materials"];
    std::unordered_map<std::string, uint32_t> MatNameToID;
    for (const auto& item : materialsData.items())
    {
        const auto& name = item.key();
        const auto& p = item.value();
        Material newMaterial{};
        // TODO: handle materials loading differently
        if (p["TYPE"] == "Diffuse")
        {
            const auto& col = p["RGB"];
            newMaterial.color = glm::vec3(col[0], col[1], col[2]);
        }
        else if (p["TYPE"] == "Emitting")
        {
            const auto& col = p["RGB"];
            newMaterial.color = glm::vec3(col[0], col[1], col[2]);
            newMaterial.emittance = p["EMITTANCE"];
        }
        else if (p["TYPE"] == "Glass")
        {
            const auto& col = p["RGB"];
            newMaterial.hasReflective = 1;
            newMaterial.hasRefractive = 1;
            newMaterial.indexOfRefraction = p["IOR"];
            newMaterial.color = glm::vec3(col[0], col[1], col[2]);
        }
        else if (p["TYPE"] == "Reflective")
        {
            const auto& col = p["RGB"];
            newMaterial.hasReflective = 1;
            newMaterial.hasRefractive = 0;
            newMaterial.color = glm::vec3(col[0], col[1], col[2]);
        }
        else if (p["TYPE"] == "Transmissive")
        {
            const auto& col = p["RGB"];
            newMaterial.hasReflective = 0;
            newMaterial.hasRefractive = 1;
            newMaterial.indexOfRefraction = p["IOR"];
            newMaterial.color = glm::vec3(col[0], col[1], col[2]);
        }
        else if (p["TYPE"] == "Microfacet")
        {
            const auto& col = p["RGB"];
            newMaterial.roughness = p["ROUGHNESS"];
            newMaterial.metallic = p["METALLIC"];
            newMaterial.indexOfRefraction = p["IOR"];
            newMaterial.color = glm::vec3(col[0], col[1], col[2]);
        }

        if (p.contains("TEXTURE"))
        {
            size_t lastSlashPos = jsonName.find_last_of("/\\");
            std::string basePath = jsonName.substr(0, lastSlashPos);
            if (!basePath.empty() && basePath.back() != '/' && basePath.back() != '\\') {
                basePath += "/";
            }
			std::string textureName = p["TEXTURE"];
            std::string texturePath = basePath + textureName;
            newMaterial.textureID = loadTexture(texturePath);
			newMaterial.hasTexture = true;
        }

        if (p.contains("BUMP_MAP"))
        {
            size_t lastSlashPos = jsonName.find_last_of("/\\");
            std::string basePath = jsonName.substr(0, lastSlashPos);
            if (!basePath.empty() && basePath.back() != '/' && basePath.back() != '\\') {
                basePath += "/";
            }
            std::string bumpName = p["BUMP_MAP"];
            std::string bumpPath = basePath + bumpName;
            newMaterial.bumpID = loadTexture(bumpPath);
            newMaterial.hasBumpMap = true;

            const auto& bumpScale = p["BUMP_SCALE"];
            newMaterial.bumpScale = bumpScale;
        }

        MatNameToID[name] = materials.size();
        materials.emplace_back(newMaterial);
    }
    const auto& objectsData = data["Objects"];
    for (const auto& p : objectsData)
    {
        const auto& type = p["TYPE"];
        if (type == "obj")
        {
            size_t lastSlashPos = jsonName.find_last_of("/\\");
            std::string basePath = jsonName.substr(0, lastSlashPos);
            std::string objName = p["PATH"];
            std::string objPath = basePath + objName;
            int mat = MatNameToID[p["MATERIAL"]];

            // Transform matrix for obj.
            glm::vec3 trans = glm::vec3(p["TRANS"][0], p["TRANS"][1], p["TRANS"][2]);
            glm::vec3 rot = glm::vec3(p["ROTAT"][0], p["ROTAT"][1], p["ROTAT"][2]);
            glm::vec3 scale = glm::vec3(p["SCALE"][0], p["SCALE"][1], p["SCALE"][2]);

            glm::mat4 transformMatrix = utilityCore::buildTransformationMatrix(trans, rot, scale);
            glm::mat4 invTransposeMatrix = glm::inverseTranspose(transformMatrix);

            loadFromOBJ(objPath, mat, transformMatrix, invTransposeMatrix);
            break;
        }
        Geom newGeom;
        if (type == "cube")
        {
            newGeom.type = CUBE;
        }
        else
        {
            newGeom.type = SPHERE;
        }
        newGeom.materialid = MatNameToID[p["MATERIAL"]];
        const auto& trans = p["TRANS"];
        const auto& rotat = p["ROTAT"];
        const auto& scale = p["SCALE"];
        newGeom.translation = glm::vec3(trans[0], trans[1], trans[2]);
        newGeom.rotation = glm::vec3(rotat[0], rotat[1], rotat[2]);
        newGeom.scale = glm::vec3(scale[0], scale[1], scale[2]);
        newGeom.transform = utilityCore::buildTransformationMatrix(
            newGeom.translation, newGeom.rotation, newGeom.scale);
        newGeom.inverseTransform = glm::inverse(newGeom.transform);
        newGeom.invTranspose = glm::inverseTranspose(newGeom.transform);

        geoms.push_back(newGeom);
    }
    const auto& cameraData = data["Camera"];
    Camera& camera = state.camera;
    RenderState& state = this->state;
    camera.resolution.x = cameraData["RES"][0];
    camera.resolution.y = cameraData["RES"][1];
    float fovy = cameraData["FOVY"];
    state.iterations = cameraData["ITERATIONS"];
    state.traceDepth = cameraData["DEPTH"];
    state.imageName = cameraData["FILE"];
    const auto& pos = cameraData["EYE"];
    const auto& lookat = cameraData["LOOKAT"];
    const auto& up = cameraData["UP"];
    camera.position = glm::vec3(pos[0], pos[1], pos[2]);
    camera.lookAt = glm::vec3(lookat[0], lookat[1], lookat[2]);
    camera.up = glm::vec3(up[0], up[1], up[2]);

    camera.focalDist = glm::length(camera.lookAt - camera.position);
    camera.aperture = cameraData["APERTURE"];

    //calculate fov based on resolution
    float yscaled = tan(fovy * (PI / 180));
    float xscaled = (yscaled * camera.resolution.x) / camera.resolution.y;
    float fovx = (atan(xscaled) * 180) / PI;
    camera.fov = glm::vec2(fovx, fovy);

    camera.right = glm::normalize(glm::cross(camera.view, camera.up));
    camera.pixelLength = glm::vec2(2 * xscaled / (float)camera.resolution.x,
        2 * yscaled / (float)camera.resolution.y);

    camera.view = glm::normalize(camera.lookAt - camera.position);

    //set up render camera stuff
    int arraylen = camera.resolution.x * camera.resolution.y;
    state.image.resize(arraylen);
    std::fill(state.image.begin(), state.image.end(), glm::vec3());

    // Call buildBVH after load from OBJ?
    if (!triangles.empty()) {
        buildBVH();
    }
}

void Scene::loadFromOBJ(const std::string& objName, int materialID, const glm::mat4& transformMatrix, const glm::mat4& invTransposeMatrix)
{
	tinyobj::attrib_t attrib;
	std::vector<tinyobj::shape_t> shapes;
	std::vector<tinyobj::material_t> materials;

    std::string warn;
    std::string err;

    bool ret = tinyobj::LoadObj(&attrib, &shapes, &materials, &warn, &err, objName.c_str());

    if (!warn.empty()) {
        std::cout << "WARN: " << warn << std::endl;
	}

    if (!err.empty()) {
        std::cerr << err << std::endl;
    }

    if (!ret) {
        throw std::runtime_error("Failed to load " + objName);
    }

    // Loop over shapes.
    for (size_t s = 0; s < shapes.size(); s++) {
        // Loop over faces(polygon).
        size_t index_offset = 0;
        for (size_t f = 0; f < shapes[s].mesh.num_face_vertices.size(); f++) {
            size_t fv = size_t(shapes[s].mesh.num_face_vertices[f]);

            // Save faces for calculating normals if none exist from obj file.
            std::vector<Vertex> faceVerts;
            faceVerts.reserve(fv);

            // Loop over vertices in the face.
            for (size_t v = 0; v < fv; v++) {

                // Define new vertex.
                Vertex newVertex;

                // Get current vert.
                tinyobj::index_t idx = shapes[s].mesh.indices[index_offset + v];

                tinyobj::real_t vx = attrib.vertices[3 * size_t(idx.vertex_index) + 0];
                tinyobj::real_t vy = attrib.vertices[3 * size_t(idx.vertex_index) + 1];
                tinyobj::real_t vz = attrib.vertices[3 * size_t(idx.vertex_index) + 2];

                // Transform vert positions.
                glm::vec4 pos = transformMatrix * glm::vec4(vx, vy, vz, 1.0f);
				newVertex.position = glm::vec3(pos);

				// Check if `normal_index` is zero or positive. negative = no normal data.
				glm::vec3 normal = glm::vec3(0.0f, 0.0f, 0.0f);
                if (idx.normal_index >= 0) {
                    tinyobj::real_t nx = attrib.normals[3 * size_t(idx.normal_index) + 0];
                    tinyobj::real_t ny = attrib.normals[3 * size_t(idx.normal_index) + 1];
                    tinyobj::real_t nz = attrib.normals[3 * size_t(idx.normal_index) + 2];
					normal = glm::vec3(nx, ny, nz);

                    // Transform normals.
                    glm::vec4 n = invTransposeMatrix * glm::vec4(nx, ny, nz, 0.0f); 
                    newVertex.normal = glm::normalize(glm::vec3(n));
				}

                // Check for texcoords.
                if (idx.texcoord_index >= 0) {
                    tinyobj::real_t tx = attrib.texcoords[2 * size_t(idx.texcoord_index) + 0];
					tinyobj::real_t ty = attrib.texcoords[2 * size_t(idx.texcoord_index) + 1];
					newVertex.uv = glm::vec2(tx, ty);
                } else {
                    newVertex.uv = glm::vec2(0.0f, 0.0f); // default if no texcoord
                }

                // Set vert attributes.
				newVertex.materialID = materialID;

				faceVerts.push_back(newVertex);
            }

            // If no normals in OBJ, compute flat face normal.
            if (faceVerts.size() >= 3) {
                bool missingNormals = true;
                for (const auto& vtx : faceVerts) {
                    if (glm::length(vtx.normal) > 1e-6f) {
                        missingNormals = false;
                        break;
                    }
                }
                if (missingNormals) {
                    glm::vec3 e1 = faceVerts[1].position - faceVerts[0].position;
                    glm::vec3 e2 = faceVerts[2].position - faceVerts[0].position;
                    glm::vec3 faceNormal = glm::normalize(glm::cross(e1, e2));
                    for (auto& vtx : faceVerts) {
                        vtx.normal = faceNormal;
                    }
                }

                // Triangulate to push back into Triangles buffer.
                for (size_t i = 1; i + 1 < faceVerts.size(); i++) {
                    Triangle tri;
                    tri.v1 = faceVerts[0];
                    tri.v2 = faceVerts[i];
                    tri.v3 = faceVerts[i + 1];

                    // Centroid for BVH splitting
                    tri.centroid = (tri.v1.position + tri.v2.position + tri.v3.position) / 3.0f;

                    // Get material for triangle.
                    tri.materialID = materialID;

                    // Get bump for triangle.
                    computeTriangleTangents(tri);

                    // Push into triangle buffer
                    this->triangles.push_back(tri);
                }
            }

            // Push vertex to vertex buffer.
            for (auto& vtx : faceVerts) {
                if (this->vertices.size() < 20) {
                    std::cout << "Vertex " << this->vertices.size()
                        << " pos: " << glm::to_string(vtx.position)
                        << " normal: " << glm::to_string(vtx.normal)
                        << " matID: " << vtx.materialID << std::endl;
                }
                this->vertices.push_back(vtx);
            }

            index_offset += fv;
        }
	}
}

// Texture loading function, to be called in loadJson after parsing texture.
int Scene::loadTexture(const std::string& texturePath)
{
	int width, height, channels;

	unsigned char* data = stbi_load(texturePath.c_str(), &width, &height, &channels, STBI_rgb_alpha);

    if (!data) {
        std::cerr << "Failed to load texture image: " << texturePath << std::endl;
        return -1;
    }

    channels = 4;

    // Create Texture struct.
	Texture newTexture;
    newTexture.width = width;
    newTexture.height = height;
    newTexture.channels = channels;
	newTexture.data = data;

    std::cout << "Loaded texture " << texturePath << " (" << width << "x" << height << ", " << channels << " channels)" << std::endl;

    int textureID = (int)textures.size();
    textures.push_back(newTexture);

    return textureID;
}

// Calculate Triangle tangents for bump maps.
void Scene::computeTriangleTangents(Triangle& tri)
{
    const glm::vec3& p1 = tri.v1.position;
    const glm::vec3& p2 = tri.v2.position;
    const glm::vec3& p3 = tri.v3.position;

    const glm::vec2& uv1 = tri.v1.uv;
    const glm::vec2& uv2 = tri.v2.uv;
    const glm::vec2& uv3 = tri.v3.uv;

    glm::vec3 dp1 = p2 - p1;
    glm::vec3 dp2 = p3 - p1;

    glm::vec2 duv1 = uv2 - uv1;
    glm::vec2 duv2 = uv3 - uv1;

    float det = duv1.x * duv2.y - duv1.y * duv2.x;

    if (glm::abs(det) < 1e-8f) {
        glm::vec3 n = glm::normalize(glm::cross(dp1, dp2));
        glm::vec3 tangent = glm::normalize(dp1);
        glm::vec3 bitangent = glm::normalize(glm::cross(n, tangent));
        tri.dpdu = tangent;
        tri.dpdv = bitangent;
        return;
    }

    float invDet = 1.0f / det;

    tri.dpdu = (dp1 * duv2.y - dp2 * duv1.y) * invDet;
    tri.dpdv = (-dp1 * duv2.x + dp2 * duv1.x) * invDet;
}

// BVH Helper function.
void Scene::UpdateNodeBounds(int start, int end, std::vector<int>& triIndices, BVHNode& node) {
    AABB aabb;
    for (int i = start; i < end; i++) {
        const Triangle& tri = triangles[triIndices[i]];
        aabb.min = glm::min(aabb.min, tri.v1.position);
        aabb.min = glm::min(aabb.min, tri.v2.position);
        aabb.min = glm::min(aabb.min, tri.v3.position);
        aabb.max = glm::max(aabb.max, tri.v1.position);
        aabb.max = glm::max(aabb.max, tri.v2.position);
        aabb.max = glm::max(aabb.max, tri.v3.position);
    }

    node.aabb = aabb;
}

// BVH Node traversal.
void Scene::buildBVH() {
	bvhNodes.clear();
	triIndices.resize(triangles.size());
    for (int i = 0; i < (int)triangles.size(); i++) {
        triIndices[i] = i;
    }

    if (triangles.empty()) return;

    // Call recursive function to build BVH tree.
    // Start = 0, End = num triangles.
    buildBVHRecursive(0, (int)triangles.size(), triIndices);
}

int Scene::buildBVHRecursive(int start, int end, std::vector<int>& triIndices) {
	// Make new node.
	/*std::cout << "Building BVH node for triangles " << start << " to " << end - 1 << std::endl;*/
	int nodeIndex = (int)bvhNodes.size();
    bvhNodes.push_back(BVHNode{});

    // Compute bounding box over all triangles.
	UpdateNodeBounds(start, end, triIndices, bvhNodes[nodeIndex]);

    // Update node.
    int numTris = end - start;
    const int leafThreshold = 4;
    
    // Leaf check.
    if (numTris <= leafThreshold) {
        // Make leaf node.
        bvhNodes[nodeIndex].start = start;
        bvhNodes[nodeIndex].triCount = numTris;
        bvhNodes[nodeIndex].left = -1;
        bvhNodes[nodeIndex].right = -1;
        return nodeIndex;
	}

    // Otherwise, update internal node.
    AABB centroidBounds;
    for (int i = start; i < end; i++) {
		const Triangle& tri = triangles[triIndices[i]];
        centroidBounds.min = glm::min(centroidBounds.min, tri.centroid);
		centroidBounds.max = glm::max(centroidBounds.max, tri.centroid);
    }
    glm::vec3 extent = centroidBounds.max - centroidBounds.min;
    int axis = 0;

    // Reassign longest axis.
    if (extent.y > extent.x && extent.y > extent.z) {
        axis = 1;
    }
    if (extent.z > extent.x) {
        axis = 2;
    }

	float splitPos = 0.5f * (centroidBounds.min[axis] + centroidBounds.max[axis]);

    // In place partition.
    int mid = start;
    for (int i = start; i < end; i++) {
        int triIndex = triIndices[i];
        if (triangles[triIndex].centroid[axis] < splitPos) {
			std::swap(triIndices[i], triIndices[mid]);
            mid++;
        }
    }

    // Pathological split. Split equally.
    if (mid == start || mid == end) {
		mid = (start + end) / 2;
    }

    // Recurse.
    bvhNodes[nodeIndex].left = buildBVHRecursive(start, mid, triIndices);
    bvhNodes[nodeIndex].right = buildBVHRecursive(mid, end, triIndices);

    bvhNodes[nodeIndex].start = -1;
    bvhNodes[nodeIndex].triCount = 0;

    return nodeIndex;
}
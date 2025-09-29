#include "scene.h"

#include "utilities.h"

#include <glm/gtc/matrix_inverse.hpp>
#include <glm/gtx/string_cast.hpp>
#include "json.hpp"

#define TINYOBJLOADER_IMPLEMENTATION
#include "tiny_obj_loader.h"

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
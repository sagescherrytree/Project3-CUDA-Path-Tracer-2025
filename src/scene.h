#pragma once

#include "sceneStructs.h"
#include <vector>

class Scene
{
private:
    void loadFromJSON(const std::string& jsonName);
	void loadFromOBJ(const std::string& objName, int materialID, const glm::mat4& transformMatrix, const glm::mat4& invTransposeMatrix);
	void UpdateNodeBounds(int start, int end, std::vector<int>& triIndices, BVHNode& node);
    void buildBVH();
	int buildBVHRecursive(int start, int end, std::vector<int>& triIndices);
public:
    Scene(std::string filename);

    std::vector<Geom> geoms;
    std::vector<Material> materials;
    std::vector<Vertex> vertices;
    std::vector<Triangle> triangles;
	std::vector<int> triIndices; // Indices into triangle array for BVH.
	std::vector<BVHNode> bvhNodes;
    RenderState state;
};

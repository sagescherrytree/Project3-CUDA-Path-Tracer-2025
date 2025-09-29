#pragma once

#include "sceneStructs.h"
#include <vector>

class Scene
{
private:
    void loadFromJSON(const std::string& jsonName);
	void loadFromOBJ(const std::string& objName, int materialID, const glm::mat4& transformMatrix, const glm::mat4& invTransposeMatrix);
public:
    Scene(std::string filename);

    std::vector<Geom> geoms;
    std::vector<Material> materials;
    std::vector<Vertex> vertices;
    std::vector<Triangle> triangles;
    RenderState state;
};

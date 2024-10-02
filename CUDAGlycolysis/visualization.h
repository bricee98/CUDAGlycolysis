#ifndef VISUALIZATION_H
#define VISUALIZATION_H

#include <GL/glew.h>
#include <GL/gl.h>
#include <GLFW/glfw3.h>
#include <glm/glm.hpp>
#include <vector>
#include "SimulationSpace.h"
#include "Molecule.cuh"

extern GLFWwindow* window;
extern bool isPaused;

void initVisualization();
void renderSimulation(const SimulationSpace& space, const std::vector<Molecule>& molecules, float total_simulated_time, float deltaTime);
void cleanupVisualization();

#endif // VISUALIZATION_H
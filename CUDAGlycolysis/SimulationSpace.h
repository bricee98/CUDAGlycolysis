#ifndef SIMULATION_SPACE_H
#define SIMULATION_SPACE_H

#include <cuda_runtime.h>

// Define constants
#define MAX_MOLECULE_TYPES 15

// SimulationSpace structure
#pragma once

struct SimulationSpace {
    float width;
    float height;
    float depth;
    int gridSizeX;
    int gridSizeY;
    int gridSizeZ;
    int num_molecules;        // Added this line
    int num_molecule_types;
    int molecule_counts[MAX_MOLECULE_TYPES];
};
#endif // SIMULATION_SPACE_H
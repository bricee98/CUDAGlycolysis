#pragma once

#include <cuda_runtime.h>
#include "SimulationSpace.h"

// Constants for the spatial grid
#define CELL_SIZE 10.0f  // Adjust based on interaction cutoff
#define MAX_MOLECULES_PER_CELL 10000  // Adjust as needed

struct Grid {
    int sizeX;
    int sizeY;
    int sizeZ;
};

struct Cell {
    int count;
    int moleculeIndices[MAX_MOLECULES_PER_CELL];
};

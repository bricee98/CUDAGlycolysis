#ifndef SIMULATION_SPACE_H
#define SIMULATION_SPACE_H

#include <cuda_runtime.h>

// Define constants
#define MAX_MOLECULE_TYPES 15

// SimulationSpace structure
struct SimulationSpace {
    int width, height, depth;
    int num_molecules;
    int num_molecule_types;
    int molecule_counts[MAX_MOLECULE_TYPES];
};

#endif // SIMULATION_SPACE_H
#pragma once

#include <cuda_runtime.h>
#include "Molecule.cuh"
#include "SimulationSpace.h"
#include "Cell.cuh"
#include <curand_kernel.h>
#include "SimulationData.h"

// Declare the variables as extern
extern __constant__ int GRID_SIZE_X;
extern __constant__ int GRID_SIZE_Y;
extern __constant__ int GRID_SIZE_Z;

// Function declarations
__global__ void assignMoleculesToCells(Molecule* molecules, int num_molecules, Cell* cells, SimulationSpace space, Grid grid);

__global__ void computeForcesUsingCells(Molecule* molecules, int num_molecules, Cell* cells, float3* forces, SimulationSpace space, Grid grid);

__global__ void applyForcesAndUpdatePositions(Molecule* molecules, float3* forces, int num_molecules, SimulationSpace space, float dt);

__global__ void initCurand(unsigned long long seed, curandState *state, int num_molecules);

__global__ void handleInteractions(Molecule* molecules, int* num_molecules, int max_molecules, curandState* states, int* reactionCounts, MoleculeCreationInfo* creationBuffer, int* numCreations, int* deletionBuffer, int* numDeletions);

__device__ float3 calculatePairwiseForce(const Molecule& mol1, const Molecule& mol2, float3 r, float distSq);

__device__ float distanceSquared(const Molecule& mol1, const Molecule& mol2);

__device__ bool checkEnzymePresence(Molecule* molecules, int num_molecules, const Molecule& substrate, MoleculeType enzymeType);

__device__ bool shouldReact(curandState* state, float baseProbability, bool enzymePresent);

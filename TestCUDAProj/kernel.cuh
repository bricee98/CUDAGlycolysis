#pragma once

#include <curand_kernel.h>
#include "Molecule.cuh"
#include "SimulationState.h"
#include "SimulationSpace.h"

// Declare all your __global__ and __device__ functions here
__global__ void handleInteractions(Molecule* molecules, int* num_molecules, int max_molecules, curandState* states, int* reactionCounts, MoleculeCreationInfo* creationBuffer, int* numCreations, int* deletionBuffer, int* numDeletions);
__global__ void initCurand(unsigned long long seed, curandState *state, int num_molecules);
__global__ void calculateForces(Molecule* molecules, int num_molecules, float3* forces);
__global__ void applyForcesAndUpdatePositions(Molecule* molecules, float3* forces, int num_molecules, SimulationSpace space, float dt, Atom* atoms);

// Declare other __device__ functions as well
__device__ float distanceSquared(const Molecule& mol1, const Molecule& mol2);
__device__ bool checkEnzymePresence(Molecule* molecules, int num_molecules, const Molecule& substrate, MoleculeType enzymeType);
// ... (declare all other __device__ functions)
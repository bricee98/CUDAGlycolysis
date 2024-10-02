#define KERNEL_CU

// Define the variables here
__device__ __constant__ int GRID_SIZE_X = 10;
__device__ __constant__ int GRID_SIZE_Y = 10;
__device__ __constant__ int GRID_SIZE_Z = 10;

// Also provide host-side copies
int h_GRID_SIZE_X = 10;
int h_GRID_SIZE_Y = 10;
int h_GRID_SIZE_Z = 10;

#include "kernel.cuh"
#include <curand_kernel.h>
#include "Molecule.cuh"
#include "Atom.cuh"
#include "SimulationSpace.h"
#include "kernel.cuh"
#include "Cell.cuh"
#include "SimulationData.h"
#include <cstdio>
#include <assert.h>
// Constants for interaction radius and reaction probabilities
#define INTERACTION_RADIUS 2.0f
#define INTERACTION_RADIUS_SQ (INTERACTION_RADIUS * INTERACTION_RADIUS)
#define BASE_REACTION_PROBABILITY 0.00f
#define ENZYME_CATALYSIS_FACTOR 100.0f
#define NUM_REACTION_TYPES 10 // Update this as you add more reaction types

// Constants for force calculations
#define COULOMB_CONSTANT 138.935456f  // (kJ*nm)/(mol*e^2)
#define CUTOFF_DISTANCE 2.0f          // nm
#define CUTOFF_DISTANCE_SQ (CUTOFF_DISTANCE * CUTOFF_DISTANCE)
#define EPSILON_0 8.854187817e-12f    // Vacuum permittivity
#define K_BOLTZMANN 0.0083144621f     // Boltzmann constant in kJ/(mol*K)
#define TEMPERATURE 310.15f           // Temperature in Kelvin (37°C)
#define SOLVENT_DIELECTRIC 78.5f      // Dielectric constant of water at 37°C

#define CELL_SIZE 2.0f  // Adjust this value based on your simulation requirements

// Add the new kernels and functions

// Kernel to assign molecules to cells
__global__ void assignMoleculesToCells(Molecule* molecules, int num_molecules, Cell* cells, SimulationSpace space, Grid grid) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_molecules) {
        Molecule& mol = molecules[idx];

        // Compute cell indices
        int cellX = static_cast<int>(mol.centerOfMass.x / CELL_SIZE);
        int cellY = static_cast<int>(mol.centerOfMass.y / CELL_SIZE);
        int cellZ = static_cast<int>(mol.centerOfMass.z / CELL_SIZE);

        // Clamp indices to grid bounds
        cellX = min(max(cellX, 0), grid.sizeX - 1);
        cellY = min(max(cellY, 0), grid.sizeY - 1);
        cellZ = min(max(cellZ, 0), grid.sizeZ - 1);

        int cellIndex = cellX + cellY * grid.sizeX + cellZ * grid.sizeX * grid.sizeY;

        // Use atomic operations to safely add molecule index to the cell
        int offset = atomicAdd(&cells[cellIndex].count, 1);
        if (offset < MAX_MOLECULES_PER_CELL) {
            cells[cellIndex].moleculeIndices[offset] = idx;
        } else {
            // Handle overflow (e.g., ignore or handle in another way)
            printf("Overflow detected in cell %d: offset=%d exceeds MAX_MOLECULES_PER_CELL=%d\n", cellIndex, offset, MAX_MOLECULES_PER_CELL);
            // Optionally, you can implement additional handling here
        }
    }
}

// Kernel to compute forces using cells
__global__ void computeForcesUsingCells(Molecule* molecules, int num_molecules, Cell* cells, float3* forces, SimulationSpace space, Grid grid) {
    int cellX = blockIdx.x;
    int cellY = blockIdx.y;
    int cellZ = blockIdx.z;

    int cellIndex = cellX + cellY * grid.sizeX + cellZ * grid.sizeX * grid.sizeY;

    // Assert to ensure cellIndex is within valid range
    assert(cellIndex < (grid.sizeX * grid.sizeY * grid.sizeZ));

    __shared__ int s_moleculeIndices[MAX_MOLECULES_PER_CELL];
    __shared__ int s_count;

    if (threadIdx.x == 0) {
        s_count = cells[cellIndex].count;
        // Log the count of molecules in the current cell
        //printf("Cell (%d, %d, %d) [Index %d] has %d molecules.\n", cellX, cellY, cellZ, cellIndex, s_count);
        for (int i = 0; i < s_count; ++i) {
            s_moleculeIndices[i] = cells[cellIndex].moleculeIndices[i];
            // Assert to ensure molecule indices are within valid range
            assert(s_moleculeIndices[i] < num_molecules);
        }

        // Clamp s_count to MAX_MOLECULES_PER_CELL to prevent out-of-bounds access
        s_count = min(s_count, MAX_MOLECULES_PER_CELL);
        if (s_count < cells[cellIndex].count) {
            printf("Clamped s_count from %d to %d for cellIndex %d to prevent overflow.\n", cells[cellIndex].count, s_count, cellIndex);
        }
    }
    __syncthreads();

    // Loop over neighboring cells
    for (int offsetX = -1; offsetX <= 1; ++offsetX) {
        for (int offsetY = -1; offsetY <= 1; ++offsetY) {
            for (int offsetZ = -1; offsetZ <= 1; ++offsetZ) {
                int neighborX = cellX + offsetX;
                int neighborY = cellY + offsetY;
                int neighborZ = cellZ + offsetZ;

                // Check bounds
                if (neighborX >= 0 && neighborX < grid.sizeX &&
                    neighborY >= 0 && neighborY < grid.sizeY &&
                    neighborZ >= 0 && neighborZ < grid.sizeZ) {
                    int neighborIndex = neighborX + neighborY * grid.sizeX + neighborZ * grid.sizeX * grid.sizeY;

                    // Assert to ensure neighborIndex is within valid range
                    assert(neighborIndex < (grid.sizeX * grid.sizeY * grid.sizeZ));

                    int neighborCount = cells[neighborIndex].count;
                    // Log the neighbor cell details
                    //printf("Processing Neighbor Cell (%d, %d, %d) [Index %d] with %d molecules.\n", neighborX, neighborY, neighborZ, neighborIndex, neighborCount);

                    for (int i = threadIdx.x; i < s_count; i += blockDim.x) {
                        int molIdx = s_moleculeIndices[i];
                        Molecule& mol_i = molecules[molIdx];
                        float3 totalForce = make_float3(0.0f, 0.0f, 0.0f);

                        for (int j = 0; j < neighborCount; ++j) {
                            int molJIdx = cells[neighborIndex].moleculeIndices[j];
                            if (molIdx != molJIdx) {
                                // Assert to ensure molecule indices are within valid range
                                assert(molIdx < num_molecules && molJIdx < num_molecules);

                                Molecule& mol_j = molecules[molJIdx];

                                float3 r;
                                r.x = mol_j.centerOfMass.x - mol_i.centerOfMass.x;
                                r.y = mol_j.centerOfMass.y - mol_i.centerOfMass.y;
                                r.z = mol_j.centerOfMass.z - mol_i.centerOfMass.z;

                                // Apply periodic boundary conditions
                                if (r.x > space.width / 2) r.x -= space.width;
                                if (r.x < -space.width / 2) r.x += space.width;
                                if (r.y > space.height / 2) r.y -= space.height;
                                if (r.y < -space.height / 2) r.y += space.height;
                                if (r.z > space.depth / 2) r.z -= space.depth;
                                if (r.z < -space.depth / 2) r.z += space.depth;

                                float distSq = r.x * r.x + r.y * r.y + r.z * r.z;

                                if (distSq < CUTOFF_DISTANCE_SQ && distSq > 0.0f) {
                                    float3 pairForce = calculatePairwiseForce(mol_i, mol_j, r, distSq);
                                    totalForce.x += pairForce.x;
                                    totalForce.y += pairForce.y;
                                    totalForce.z += pairForce.z;
                                }
                            }
                        }

                        // Update force
                        atomicAdd(&forces[molIdx].x, totalForce.x);
                        atomicAdd(&forces[molIdx].y, totalForce.y);
                        atomicAdd(&forces[molIdx].z, totalForce.z);
                    }
                }
            }
        }
    }
}

// Implement the calculatePairwiseForce function
__device__ float3 calculatePairwiseForce(const Molecule& mol1, const Molecule& mol2, float3 r, float distSq) {
    float3 force = make_float3(0.0f, 0.0f, 0.0f);
    float dist = sqrtf(distSq);
    float invDist = 1.0f / dist;

    // Parameters (adjust as needed)
    float epsilon = 0.1f;
    float sigma = 0.3f;

    // Lennard-Jones potential
    float sigmaOverDist = sigma * invDist;
    float sigmaOverDist6 = powf(sigmaOverDist, 6);
    float sigmaOverDist12 = sigmaOverDist6 * sigmaOverDist6;

    float forceLJ = 24.0f * epsilon * (2.0f * sigmaOverDist12 - sigmaOverDist6) * invDist * invDist;

    // Coulomb force
    float chargeProduct = mol1.totalCharge * mol2.totalCharge;
    float forceCoulomb = COULOMB_CONSTANT * chargeProduct * invDist * invDist;

    float totalForceMultiplier = forceLJ + forceCoulomb;

    force.x = r.x * totalForceMultiplier * invDist;
    force.y = r.y * totalForceMultiplier * invDist;
    force.z = r.z * totalForceMultiplier * invDist;

    return force;
}

// Helper function to calculate distance squared between two molecules
__device__ float distanceSquared(const Molecule& mol1, const Molecule& mol2) {
    float dx = mol1.centerOfMass.x - mol2.centerOfMass.x;
    float dy = mol1.centerOfMass.y - mol2.centerOfMass.y;
    float dz = mol1.centerOfMass.z - mol2.centerOfMass.z;
    return dx*dx + dy*dy + dz*dz;
}

// Helper function to check for enzyme presence
__device__ bool checkEnzymePresence(Molecule* molecules, int num_molecules, const Molecule& substrate, MoleculeType enzymeType) {
    for (int k = 0; k < num_molecules; k++) {
        if (molecules[k].type == enzymeType && distanceSquared(substrate, molecules[k]) <= INTERACTION_RADIUS_SQ) {
            return true;
        }
    }
    return false;
}

// Helper function to check if a reaction should occur
__device__ bool shouldReact(curandState* state, float baseProbability, bool enzymePresent) {
    float reactionProbability = baseProbability;
    if (enzymePresent) {
        reactionProbability *= ENZYME_CATALYSIS_FACTOR;
    }
    return curand_uniform(state) < reactionProbability;
}

// Kernel to initialize curand states
__global__ void initCurand(unsigned long long seed, curandState *state, int num_molecules) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_molecules) {
        curand_init(seed, idx, 0, &state[idx]);
    }
}

// Main interaction kernel
__global__ void handleInteractions(Molecule* molecules, int* num_molecules, int max_molecules, curandState* states,
                                   int* reactionCounts, MoleculeCreationInfo* creationBuffer, int* numCreations,
                                   int* deletionBuffer, int* numDeletions) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Confirm thread index and number of molecules
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        printf("handleInteractions kernel launched with %d threads\n", gridDim.x * blockDim.x);
        printf("Number of molecules: %d\n", *num_molecules);
    }

    if (idx >= *num_molecules) return;

    // Add a print statement to confirm execution
    // printf("Thread %d is processing molecule %d of type %d\n", idx, idx, molecules[idx].type);

    Molecule& mol1 = molecules[idx];
    curandState localState = states[idx];

    for (int j = idx + 1; j < *num_molecules; j++) {
        Molecule& mol2 = molecules[j];

        if (distanceSquared(mol1, mol2) <= INTERACTION_RADIUS_SQ) {
            switch (mol1.type) {
                case GLUCOSE:
                    if (mol2.type == ATP) {
                        bool enzymePresent = checkEnzymePresence(molecules, *num_molecules, mol1, HEXOKINASE);
                        if (shouldReact(&localState, BASE_REACTION_PROBABILITY, enzymePresent)) {
                            int delIdx = atomicAdd(numDeletions, 2);
                            deletionBuffer[delIdx] = idx;
                            deletionBuffer[delIdx + 1] = j;

                            int createIdx = atomicAdd(numCreations, 2);
                            creationBuffer[createIdx] = {GLUCOSE_6_PHOSPHATE, mol1.centerOfMass.x, mol1.centerOfMass.y, mol1.centerOfMass.z};
                            creationBuffer[createIdx + 1] = {ADP, mol2.centerOfMass.x, mol2.centerOfMass.y, mol2.centerOfMass.z};

                            atomicAdd(&reactionCounts[0], 1);
                            return;
                        }
                    }
                    break;

                default:
                    break;
            }
        }
    }

    states[idx] = localState;
}

// Calculate pairwise force between two molecule

// Kernel to calculate forces
__global__ void calculateForces(Molecule* molecules, int num_molecules, float3* forces) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_molecules) {
        float3 totalForce = make_float3(0.0f, 0.0f, 0.0f);

        Molecule& mol_i = molecules[idx];

        for (int j = 0; j < num_molecules; ++j) {
            if (idx != j) {
                Molecule& mol_j = molecules[j];

                float3 r;
                r.x = mol_j.centerOfMass.x - mol_i.centerOfMass.x;
                r.y = mol_j.centerOfMass.y - mol_i.centerOfMass.y;
                r.z = mol_j.centerOfMass.z - mol_i.centerOfMass.z;

                float distSq = r.x * r.x + r.y * r.y + r.z * r.z;

                if (distSq < CUTOFF_DISTANCE_SQ && distSq > 0.0f) {
                    float3 pairForce = calculatePairwiseForce(mol_i, mol_j, r, distSq);
                    totalForce.x += pairForce.x;
                    totalForce.y += pairForce.y;
                    totalForce.z += pairForce.z;
                }
            }
        }

        forces[idx] = totalForce;
    }
}

// Kernel to apply forces and update positions
__global__ void applyForcesAndUpdatePositions(Molecule* molecules, float3* forces, int num_molecules, SimulationSpace space, float dt) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_molecules) {

        // Log info for molecule 500
        if (idx == 500) {
            printf("Molecule 500: Position (%f, %f, %f), Velocity (%f, %f, %f), Force (%f, %f, %f)\n",
                   molecules[500].centerOfMass.x, molecules[500].centerOfMass.y, molecules[500].centerOfMass.z,
                   molecules[500].vx, molecules[500].vy, molecules[500].vz,
                   forces[500].x, forces[500].y, forces[500].z);
        }

        Molecule& mol = molecules[idx];
        float totalMass = mol.getTotalMass();
        float3 force = forces[idx];

        // Apply acceleration
        float ax = force.x / totalMass;
        float ay = force.y / totalMass;
        float az = force.z / totalMass;

        // Update velocity directly
        mol.vx += ax * dt;
        mol.vy += ay * dt;
        mol.vz += az * dt;

        // Update position directly
        mol.centerOfMass.x += mol.vx * dt;
        mol.centerOfMass.y += mol.vy * dt;
        mol.centerOfMass.z += mol.vz * dt;

        // Apply periodic boundary conditions
        mol.centerOfMass.x = fmodf(mol.centerOfMass.x + space.width, space.width);
        mol.centerOfMass.y = fmodf(mol.centerOfMass.y + space.height, space.height);
        mol.centerOfMass.z = fmodf(mol.centerOfMass.z + space.depth, space.depth);

        // Log info for molecule 500
        if (idx == 500) {
            printf("New Molecule 500: Position (%f, %f, %f), Velocity (%f, %f, %f)\n",
                   mol.centerOfMass.x, mol.centerOfMass.y, mol.centerOfMass.z,
                   mol.vx, mol.vy, mol.vz);
        }
    }
}
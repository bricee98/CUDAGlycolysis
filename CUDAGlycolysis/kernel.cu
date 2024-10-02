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
#include "SimulationSpace.h"
#include "kernel.cuh"
#include "Cell.cuh"
#include "SimulationData.h"
#include <cstdio>
#include <assert.h>
// Constants for interaction radius and reaction probabilities
#define INTERACTION_RADIUS 5.0f  // nm
#define INTERACTION_RADIUS_SQ (INTERACTION_RADIUS * INTERACTION_RADIUS)
#define BASE_REACTION_PROBABILITY 1e-6f  // Adjusted for microsecond timescale
#define ENZYME_CATALYSIS_FACTOR 100.0f
#define NUM_REACTION_TYPES 10

// Constants for force calculations
#define COULOMB_CONSTANT 138.935458f  // (kJ*nm)/(mol*e^2)
#define CUTOFF_DISTANCE 10.0f  // nm
#define CUTOFF_DISTANCE_SQ (CUTOFF_DISTANCE * CUTOFF_DISTANCE)
#define EPSILON_0 8.854187817e-12f  // F/m (Vacuum permittivity)
#define K_BOLTZMANN 0.0083144621f  // kJ/(mol*K)
#define TEMPERATURE 310.15f  // K (37°C)
#define SOLVENT_DIELECTRIC 78.5f  // Dimensionless (for water at 37°C)

#define CELL_SIZE 10.0f  // nm

#define VISCOSITY 6.91e-4f  // kJ*s/(nm^3*mol) (viscosity of water at 37°C)

#define REPULSION_COEFFICIENT 1.0f  // Adjust as needed

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

    __shared__ int s_moleculeIndices[MAX_MOLECULES_PER_CELL];
    __shared__ int s_count;

    if (threadIdx.x == 0) {
        s_count = cells[cellIndex].count;
        for (int i = 0; i < s_count; ++i) {
            s_moleculeIndices[i] = cells[cellIndex].moleculeIndices[i];
        }
        s_count = min(s_count, MAX_MOLECULES_PER_CELL);
    }
    __syncthreads();

    // Loop over neighboring cells
    for (int offsetX = -1; offsetX <= 1; ++offsetX) {
        for (int offsetY = -1; offsetY <= 1; ++offsetY) {
            for (int offsetZ = -1; offsetZ <= 1; ++offsetZ) {
                int neighborX = cellX + offsetX;
                int neighborY = cellY + offsetY;
                int neighborZ = cellZ + offsetZ;

                // Check bounds without wrapping
                if (neighborX >= 0 && neighborX < grid.sizeX &&
                    neighborY >= 0 && neighborY < grid.sizeY &&
                    neighborZ >= 0 && neighborZ < grid.sizeZ) {
                    int neighborIndex = neighborX + neighborY * grid.sizeX + neighborZ * grid.sizeX * grid.sizeY;

                    int neighborCount = cells[neighborIndex].count;

                    for (int i = threadIdx.x; i < s_count; i += blockDim.x) {
                        int molIdx = s_moleculeIndices[i];
                        Molecule& mol_i = molecules[molIdx];
                        float3 totalForce = make_float3(0.0f, 0.0f, 0.0f);

                        for (int j = 0; j < neighborCount; ++j) {
                            int molJIdx = cells[neighborIndex].moleculeIndices[j];
                            if (molIdx != molJIdx) {
                                Molecule& mol_j = molecules[molJIdx];

                                float3 r;
                                r.x = mol_j.centerOfMass.x - mol_i.centerOfMass.x;
                                r.y = mol_j.centerOfMass.y - mol_i.centerOfMass.y;
                                r.z = mol_j.centerOfMass.z - mol_i.centerOfMass.z;

                                float distSq = r.x * r.x + r.y * r.y + r.z * r.z;
                                float minDist = mol_i.radius + mol_j.radius;

                                if (distSq < minDist * minDist && distSq > 0.0f) {
                                    float dist = sqrtf(distSq);
                                    float overlap = minDist - dist;
                                    // Simple linear repulsion
                                    float forceMag = overlap * REPULSION_COEFFICIENT; // Define repulsionCoefficient appropriately
                                    float invDist = 1.0f / dist;
                                    totalForce.x += r.x * forceMag * invDist;
                                    totalForce.y += r.y * forceMag * invDist;
                                    totalForce.z += r.z * forceMag * invDist;
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

    Molecule& mol1 = molecules[idx];
    curandState localState = states[idx];

    for (int j = idx + 1; j < *num_molecules; j++) {
        Molecule& mol2 = molecules[j];

        if (distanceSquared(mol1, mol2) <= INTERACTION_RADIUS_SQ) {
            // Existing reaction: Glucose + ATP -> Glucose-6-Phosphate + ADP (Hexokinase)
            if ((mol1.type == GLUCOSE && mol2.type == ATP) || (mol2.type == GLUCOSE && mol1.type == ATP)) {
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

            // Glucose-6-Phosphate -> Fructose-6-Phosphate (Glucose-6-Phosphate Isomerase)
            else if (mol1.type == GLUCOSE_6_PHOSPHATE || mol2.type == GLUCOSE_6_PHOSPHATE) {
                bool enzymePresent = checkEnzymePresence(molecules, *num_molecules, mol1, GLUCOSE_6_PHOSPHATE_ISOMERASE);
                if (shouldReact(&localState, BASE_REACTION_PROBABILITY, enzymePresent)) {
                    int delIdx = atomicAdd(numDeletions, 1);
                    deletionBuffer[delIdx] = (mol1.type == GLUCOSE_6_PHOSPHATE) ? idx : j;

                    int createIdx = atomicAdd(numCreations, 1);
                    creationBuffer[createIdx] = {FRUCTOSE_6_PHOSPHATE, mol1.centerOfMass.x, mol1.centerOfMass.y, mol1.centerOfMass.z};

                    atomicAdd(&reactionCounts[1], 1);
                    return;
                }
            }

            // Fructose-6-Phosphate + ATP -> Fructose-1,6-Bisphosphate + ADP (Phosphofructokinase-1)
            else if ((mol1.type == FRUCTOSE_6_PHOSPHATE && mol2.type == ATP) || (mol2.type == FRUCTOSE_6_PHOSPHATE && mol1.type == ATP)) {
                bool enzymePresent = checkEnzymePresence(molecules, *num_molecules, mol1, PHOSPHOFRUCTOKINASE_1);
                if (shouldReact(&localState, BASE_REACTION_PROBABILITY, enzymePresent)) {
                    int delIdx = atomicAdd(numDeletions, 2);
                    deletionBuffer[delIdx] = idx;
                    deletionBuffer[delIdx + 1] = j;

                    int createIdx = atomicAdd(numCreations, 2);
                    creationBuffer[createIdx] = {FRUCTOSE_1_6_BISPHOSPHATE, mol1.centerOfMass.x, mol1.centerOfMass.y, mol1.centerOfMass.z};
                    creationBuffer[createIdx + 1] = {ADP, mol2.centerOfMass.x, mol2.centerOfMass.y, mol2.centerOfMass.z};

                    atomicAdd(&reactionCounts[2], 1);
                    return;
                }
            }

            // Fructose-1,6-Bisphosphate -> Dihydroxyacetone Phosphate + Glyceraldehyde-3-Phosphate (Aldolase)
            else if (mol1.type == FRUCTOSE_1_6_BISPHOSPHATE || mol2.type == FRUCTOSE_1_6_BISPHOSPHATE) {
                bool enzymePresent = checkEnzymePresence(molecules, *num_molecules, mol1, ALDOLASE);
                if (shouldReact(&localState, BASE_REACTION_PROBABILITY, enzymePresent)) {
                    int delIdx = atomicAdd(numDeletions, 1);
                    deletionBuffer[delIdx] = (mol1.type == FRUCTOSE_1_6_BISPHOSPHATE) ? idx : j;

                    int createIdx = atomicAdd(numCreations, 2);
                    creationBuffer[createIdx] = {DIHYDROXYACETONE_PHOSPHATE, mol1.centerOfMass.x, mol1.centerOfMass.y, mol1.centerOfMass.z};
                    creationBuffer[createIdx + 1] = {GLYCERALDEHYDE_3_PHOSPHATE, mol1.centerOfMass.x, mol1.centerOfMass.y, mol1.centerOfMass.z};

                    atomicAdd(&reactionCounts[3], 1);
                    return;
                }
            }

            // Dihydroxyacetone Phosphate -> Glyceraldehyde-3-Phosphate (Triosephosphate Isomerase)
            else if (mol1.type == DIHYDROXYACETONE_PHOSPHATE || mol2.type == DIHYDROXYACETONE_PHOSPHATE) {
                bool enzymePresent = checkEnzymePresence(molecules, *num_molecules, mol1, TRIOSEPHOSPHATE_ISOMERASE);
                if (shouldReact(&localState, BASE_REACTION_PROBABILITY, enzymePresent)) {
                    int delIdx = atomicAdd(numDeletions, 1);
                    deletionBuffer[delIdx] = (mol1.type == DIHYDROXYACETONE_PHOSPHATE) ? idx : j;

                    int createIdx = atomicAdd(numCreations, 1);
                    creationBuffer[createIdx] = {GLYCERALDEHYDE_3_PHOSPHATE, mol1.centerOfMass.x, mol1.centerOfMass.y, mol1.centerOfMass.z};

                    atomicAdd(&reactionCounts[4], 1);
                    return;
                }
            }

            // Glyceraldehyde-3-Phosphate + NAD+ + Pi -> 1,3-Bisphosphoglycerate + NADH + H+ (Glyceraldehyde-3-Phosphate Dehydrogenase)
            else if ((mol1.type == GLYCERALDEHYDE_3_PHOSPHATE && mol2.type == NAD_PLUS) || (mol2.type == GLYCERALDEHYDE_3_PHOSPHATE && mol1.type == NAD_PLUS)) {
                bool enzymePresent = checkEnzymePresence(molecules, *num_molecules, mol1, GLYCERALDEHYDE_3_PHOSPHATE_DEHYDROGENASE);
                bool phosphatePresent = checkEnzymePresence(molecules, *num_molecules, mol1, INORGANIC_PHOSPHATE);
                if (shouldReact(&localState, BASE_REACTION_PROBABILITY, enzymePresent) && phosphatePresent) {
                    int delIdx = atomicAdd(numDeletions, 3);
                    deletionBuffer[delIdx] = idx;
                    deletionBuffer[delIdx + 1] = j;
                    // Find and delete an inorganic phosphate molecule

                    int createIdx = atomicAdd(numCreations, 3);
                    creationBuffer[createIdx] = {_1_3_BISPHOSPHOGLYCERATE, mol1.centerOfMass.x, mol1.centerOfMass.y, mol1.centerOfMass.z};
                    creationBuffer[createIdx + 1] = {NADH, mol2.centerOfMass.x, mol2.centerOfMass.y, mol2.centerOfMass.z};
                    creationBuffer[createIdx + 2] = {PROTON, mol1.centerOfMass.x, mol1.centerOfMass.y, mol1.centerOfMass.z};

                    atomicAdd(&reactionCounts[5], 1);
                    return;
                }
            }

            // 1,3-Bisphosphoglycerate + ADP -> 3-Phosphoglycerate + ATP (Phosphoglycerate Kinase)
            else if ((mol1.type == _1_3_BISPHOSPHOGLYCERATE && mol2.type == ADP) || (mol2.type == _1_3_BISPHOSPHOGLYCERATE && mol1.type == ADP)) {
                bool enzymePresent = checkEnzymePresence(molecules, *num_molecules, mol1, PHOSPHOGLYCERATE_KINASE);
                if (shouldReact(&localState, BASE_REACTION_PROBABILITY, enzymePresent)) {
                    int delIdx = atomicAdd(numDeletions, 2);
                    deletionBuffer[delIdx] = idx;
                    deletionBuffer[delIdx + 1] = j;

                    int createIdx = atomicAdd(numCreations, 2);
                    creationBuffer[createIdx] = {_3_PHOSPHOGLYCERATE, mol1.centerOfMass.x, mol1.centerOfMass.y, mol1.centerOfMass.z};
                    creationBuffer[createIdx + 1] = {ATP, mol2.centerOfMass.x, mol2.centerOfMass.y, mol2.centerOfMass.z};

                    atomicAdd(&reactionCounts[6], 1);
                    return;
                }
            }

            // 3-Phosphoglycerate -> 2-Phosphoglycerate (Phosphoglycerate Mutase)
            else if (mol1.type == _3_PHOSPHOGLYCERATE || mol2.type == _3_PHOSPHOGLYCERATE) {
                bool enzymePresent = checkEnzymePresence(molecules, *num_molecules, mol1, PHOSPHOGLYCERATE_MUTASE);
                if (shouldReact(&localState, BASE_REACTION_PROBABILITY, enzymePresent)) {
                    int delIdx = atomicAdd(numDeletions, 1);
                    deletionBuffer[delIdx] = (mol1.type == _3_PHOSPHOGLYCERATE) ? idx : j;

                    int createIdx = atomicAdd(numCreations, 1);
                    creationBuffer[createIdx] = {_2_PHOSPHOGLYCERATE, mol1.centerOfMass.x, mol1.centerOfMass.y, mol1.centerOfMass.z};

                    atomicAdd(&reactionCounts[7], 1);
                    return;
                }
            }

            // 2-Phosphoglycerate -> Phosphoenolpyruvate + H2O (Enolase)
            else if (mol1.type == _2_PHOSPHOGLYCERATE || mol2.type == _2_PHOSPHOGLYCERATE) {
                bool enzymePresent = checkEnzymePresence(molecules, *num_molecules, mol1, ENOLASE);
                if (shouldReact(&localState, BASE_REACTION_PROBABILITY, enzymePresent)) {
                    int delIdx = atomicAdd(numDeletions, 1);
                    deletionBuffer[delIdx] = (mol1.type == _2_PHOSPHOGLYCERATE) ? idx : j;

                    int createIdx = atomicAdd(numCreations, 2);
                    creationBuffer[createIdx] = {PHOSPHOENOLPYRUVATE, mol1.centerOfMass.x, mol1.centerOfMass.y, mol1.centerOfMass.z};
                    creationBuffer[createIdx + 1] = {WATER, mol1.centerOfMass.x, mol1.centerOfMass.y, mol1.centerOfMass.z};

                    atomicAdd(&reactionCounts[8], 1);
                    return;
                }
            }

            // Phosphoenolpyruvate + ADP -> Pyruvate + ATP (Pyruvate Kinase)
            else if ((mol1.type == PHOSPHOENOLPYRUVATE && mol2.type == ADP) || (mol2.type == PHOSPHOENOLPYRUVATE && mol1.type == ADP)) {
                bool enzymePresent = checkEnzymePresence(molecules, *num_molecules, mol1, PYRUVATE_KINASE);
                if (shouldReact(&localState, BASE_REACTION_PROBABILITY, enzymePresent)) {
                    int delIdx = atomicAdd(numDeletions, 2);
                    deletionBuffer[delIdx] = idx;
                    deletionBuffer[delIdx + 1] = j;

                    int createIdx = atomicAdd(numCreations, 2);
                    creationBuffer[createIdx] = {PYRUVATE, mol1.centerOfMass.x, mol1.centerOfMass.y, mol1.centerOfMass.z};
                    creationBuffer[createIdx + 1] = {ATP, mol2.centerOfMass.x, mol2.centerOfMass.y, mol2.centerOfMass.z};

                    atomicAdd(&reactionCounts[9], 1);
                    return;
                }
            }
        }
    }

    states[idx] = localState;
}

// Kernel to apply forces and update positions
__global__ void applyForcesAndUpdatePositions(Molecule* molecules, int num_molecules, SimulationSpace space, float dt, curandState* randStates) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_molecules) {
        Molecule& mol = molecules[idx];

        // Calculate diffusion coefficient (D)
        float gamma = 6.0f * 3.14159265358979323846f * mol.radius * VISCOSITY;
        float D = K_BOLTZMANN * TEMPERATURE / gamma; // nm^2/s
        D *= 1e-6f; // Convert D to nm^2/μs

        // Random displacement due to Brownian motion
        curandState localState = randStates[idx];
        float sqrtTerm = sqrtf(2.0f * D * dt);

        float3 randomDisplacement;
        randomDisplacement.x = curand_normal(&localState) * sqrtTerm;
        randomDisplacement.y = curand_normal(&localState) * sqrtTerm;
        randomDisplacement.z = curand_normal(&localState) * sqrtTerm;

        // Update position
        mol.centerOfMass.x += randomDisplacement.x;
        mol.centerOfMass.y += randomDisplacement.y;
        mol.centerOfMass.z += randomDisplacement.z;

        // Handle boundary conditions
        // Bounce off walls
        if (mol.centerOfMass.x < 0 || mol.centerOfMass.x > space.width) {
            mol.vx = -mol.vx;
            mol.centerOfMass.x = fmaxf(0.0f, fminf(mol.centerOfMass.x, space.width));
        }
        if (mol.centerOfMass.y < 0 || mol.centerOfMass.y > space.height) {
            mol.vy = -mol.vy;
            mol.centerOfMass.y = fmaxf(0.0f, fminf(mol.centerOfMass.y, space.height));
        }
        if (mol.centerOfMass.z < 0 || mol.centerOfMass.z > space.depth) {
            mol.vz = -mol.vz;
            mol.centerOfMass.z = fmaxf(0.0f, fminf(mol.centerOfMass.z, space.depth));
        }

        // Update the random state
        randStates[idx] = localState;
    }
}
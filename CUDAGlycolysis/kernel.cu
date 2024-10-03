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
#define INTERACTION_RADIUS 1.0       // nm
#define INTERACTION_RADIUS_SQ (INTERACTION_RADIUS * INTERACTION_RADIUS)
#define BASE_REACTION_PROBABILITY 1e-6  // Adjusted for microsecond timescale
#define ENZYME_CATALYSIS_FACTOR 1e8
#define NUM_REACTION_TYPES 10

#define DISSOCIATION_PROBABILITY 0.0001
#define REACTION_PROBABILITY 0.1
// Constants for force calculations
#define COULOMB_CONSTANT 138.935458      // Keep as double by default
#define CUTOFF_DISTANCE 10.0f  // nm
#define CUTOFF_DISTANCE_SQ (CUTOFF_DISTANCE * CUTOFF_DISTANCE)
#define EPSILON_0 8.854187817e-12        // F/m (Vacuum permittivity)
#define K_BOLTZMANN 1.380649e-23         // J/K
#define TEMPERATURE 310.15               // K (37°C)
#define SOLVENT_DIELECTRIC 78.5f  // Dimensionless (for water at 37°C)

#define CELL_SIZE 10.0f  // nm

#define VISCOSITY 6.91e-13               // kg/(nm·s)

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
            //printf("Overflow detected in cell %d: offset=%d exceeds MAX_MOLECULES_PER_CELL=%d\n", cellIndex, offset, MAX_MOLECULES_PER_CELL);
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

// Kernel to initialize curand states
__global__ void initCurand(unsigned long long seed, curandState *state, int num_molecules) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_molecules) {
        curand_init(seed, idx, 0, &state[idx]);
    }
}

// Helper function to handle reactions and dissociations
__device__ void handleComplexReaction(
    Molecule* molecules, int idx, curandState* states,
    MoleculeCreationInfo* creationBuffer, int* numCreations,
    int* deletionBuffer, int* numDeletions,
    MoleculeType complexType,
    MoleculeType dissociation1, MoleculeType dissociation2, MoleculeType dissociation3,
    MoleculeType reaction1, MoleculeType reaction2, MoleculeType reaction3,
    float reactionProbability, float dissociationProbability
) {
    if (curand_uniform(&states[idx]) < reactionProbability) {
        int delIdx = atomicAdd(numDeletions, 1);
        deletionBuffer[delIdx] = idx;
        
        int createIdx = atomicAdd(numCreations, 1);
        creationBuffer[createIdx] = {reaction1, molecules[idx].centerOfMass.x, molecules[idx].centerOfMass.y, molecules[idx].centerOfMass.z};
        
        if (reaction2 != NONE) {
            createIdx = atomicAdd(numCreations, 1);
            creationBuffer[createIdx] = {reaction2, molecules[idx].centerOfMass.x, molecules[idx].centerOfMass.y, molecules[idx].centerOfMass.z};
        }
        
        if (reaction3 != NONE) {
            createIdx = atomicAdd(numCreations, 1);
            creationBuffer[createIdx] = {reaction3, molecules[idx].centerOfMass.x, molecules[idx].centerOfMass.y, molecules[idx].centerOfMass.z};
        }
    }
    else if (curand_uniform(&states[idx]) < dissociationProbability) {
        int delIdx = atomicAdd(numDeletions, 1);
        deletionBuffer[delIdx] = idx;
        
        int createIdx = atomicAdd(numCreations, 1);
        creationBuffer[createIdx] = {dissociation1, molecules[idx].centerOfMass.x, molecules[idx].centerOfMass.y, molecules[idx].centerOfMass.z};
        
        if (dissociation2 != NONE) {
            createIdx = atomicAdd(numCreations, 1);
            creationBuffer[createIdx] = {dissociation2, molecules[idx].centerOfMass.x, molecules[idx].centerOfMass.y, molecules[idx].centerOfMass.z};
        }
        
        if (dissociation3 != NONE) {
            createIdx = atomicAdd(numCreations, 1);
            creationBuffer[createIdx] = {dissociation3, molecules[idx].centerOfMass.x, molecules[idx].centerOfMass.y, molecules[idx].centerOfMass.z};
        }
    }
}

// Main kernel function
__global__ void handleReactionsAndDissociations(Molecule* molecules, int* num_molecules, int max_molecules, curandState* states,
                                   MoleculeCreationInfo* creationBuffer, int* numCreations,
                                   int* deletionBuffer, int* numDeletions) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= *num_molecules) return;

    switch(molecules[idx].type) {
        case HEXOKINASE_GLUCOSE_COMPLEX:
            handleComplexReaction(molecules, idx, states, creationBuffer, numCreations, deletionBuffer, numDeletions,
                HEXOKINASE_GLUCOSE_COMPLEX,
                GLUCOSE, HEXOKINASE, NONE,
                NONE, NONE, NONE, // No reaction products
                0.0f, DISSOCIATION_PROBABILITY);
            break;

        case HEXOKINASE_GLUCOSE_ATP_COMPLEX:
            handleComplexReaction(molecules, idx, states, creationBuffer, numCreations, deletionBuffer, numDeletions,
                HEXOKINASE_GLUCOSE_ATP_COMPLEX,
                GLUCOSE, HEXOKINASE, ATP,
                GLUCOSE_6_PHOSPHATE, HEXOKINASE, ADP,
                REACTION_PROBABILITY, DISSOCIATION_PROBABILITY);
            break;

        case GLUCOSE_6_PHOSPHATE_ISOMERASE_COMPLEX:
            handleComplexReaction(molecules, idx, states, creationBuffer, numCreations, deletionBuffer, numDeletions,
                GLUCOSE_6_PHOSPHATE_ISOMERASE_COMPLEX, 
                GLUCOSE_6_PHOSPHATE, PHOSPHOGLUCOSE_ISOMERASE, NONE,
                FRUCTOSE_6_PHOSPHATE, PHOSPHOGLUCOSE_ISOMERASE, NONE,
                REACTION_PROBABILITY, DISSOCIATION_PROBABILITY);
            break;

        case FRUCTOSE_6_PHOSPHATE_ISOMERASE_COMPLEX:
            handleComplexReaction(molecules, idx, states, creationBuffer, numCreations, deletionBuffer, numDeletions,
                FRUCTOSE_6_PHOSPHATE_ISOMERASE_COMPLEX,
                FRUCTOSE_6_PHOSPHATE, PHOSPHOGLUCOSE_ISOMERASE, NONE,
                GLUCOSE_6_PHOSPHATE, PHOSPHOGLUCOSE_ISOMERASE, NONE,
                REACTION_PROBABILITY, DISSOCIATION_PROBABILITY);
            break;

        case PHOSPHOFRUCTOKINASE_1_COMPLEX:
            handleComplexReaction(molecules, idx, states, creationBuffer, numCreations, deletionBuffer, numDeletions,
                PHOSPHOFRUCTOKINASE_1_COMPLEX,
                FRUCTOSE_6_PHOSPHATE, PHOSPHOFRUCTOKINASE_1, NONE, 
                NONE, NONE, NONE, // No reaction products
                0.0f, DISSOCIATION_PROBABILITY);
            break;

        case PHOSPHOFRUCTOKINASE_1_ATP_COMPLEX:
            handleComplexReaction(molecules, idx, states, creationBuffer, numCreations, deletionBuffer, numDeletions,
                PHOSPHOFRUCTOKINASE_1_ATP_COMPLEX,
                FRUCTOSE_6_PHOSPHATE, PHOSPHOFRUCTOKINASE_1, NONE,
                FRUCTOSE_1_6_BISPHOSPHATE, PHOSPHOFRUCTOKINASE_1, ADP,
                REACTION_PROBABILITY, DISSOCIATION_PROBABILITY);
            break;

        case FRUCTOSE_1_6_BISPHOSPHATE_ALDOLASE_COMPLEX:
            handleComplexReaction(molecules, idx, states, creationBuffer, numCreations, deletionBuffer, numDeletions,
                FRUCTOSE_1_6_BISPHOSPHATE_ALDOLASE_COMPLEX,
                FRUCTOSE_1_6_BISPHOSPHATE, ALDOLASE, NONE,
                DIHYDROXYACETONE_PHOSPHATE, GLYCERALDEHYDE_3_PHOSPHATE, ALDOLASE,
                REACTION_PROBABILITY, DISSOCIATION_PROBABILITY);
            break;

        case GLYCERALDEHYDE_3_PHOSPHATE_ALDOLASE_COMPLEX:
            handleComplexReaction(molecules, idx, states, creationBuffer, numCreations, deletionBuffer, numDeletions,
                GLYCERALDEHYDE_3_PHOSPHATE_ALDOLASE_COMPLEX,
                GLYCERALDEHYDE_3_PHOSPHATE, ALDOLASE, NONE,
                NONE, NONE, NONE, // No reaction products
                0.0f, DISSOCIATION_PROBABILITY);
            break;

        case GLYCERALDEHYDE_3_PHOSPHATE_ALDOLASE_DHAP_COMPLEX:
            handleComplexReaction(molecules, idx, states, creationBuffer, numCreations, deletionBuffer, numDeletions,
                GLYCERALDEHYDE_3_PHOSPHATE_ALDOLASE_DHAP_COMPLEX,
                GLYCERALDEHYDE_3_PHOSPHATE, ALDOLASE, NONE,
                FRUCTOSE_1_6_BISPHOSPHATE, ALDOLASE, DIHYDROXYACETONE_PHOSPHATE,
                REACTION_PROBABILITY, DISSOCIATION_PROBABILITY);
            break;

        case DHAP_TRIOSEPHOSPHATE_ISOMERASE_COMPLEX:
            handleComplexReaction(molecules, idx, states, creationBuffer, numCreations, deletionBuffer, numDeletions,
                DHAP_TRIOSEPHOSPHATE_ISOMERASE_COMPLEX,
                DIHYDROXYACETONE_PHOSPHATE, TRIOSEPHOSPHATE_ISOMERASE, NONE,
                GLYCERALDEHYDE_3_PHOSPHATE, TRIOSEPHOSPHATE_ISOMERASE, NONE,
                REACTION_PROBABILITY, DISSOCIATION_PROBABILITY);
            break;

        case GLYCERALDEHYDE_3_PHOSPHATE_TRIOSEPHOSPHATE_ISOMERASE_COMPLEX:
            handleComplexReaction(molecules, idx, states, creationBuffer, numCreations, deletionBuffer, numDeletions,
                GLYCERALDEHYDE_3_PHOSPHATE_TRIOSEPHOSPHATE_ISOMERASE_COMPLEX,
                GLYCERALDEHYDE_3_PHOSPHATE, TRIOSEPHOSPHATE_ISOMERASE, NONE,
                DIHYDROXYACETONE_PHOSPHATE, TRIOSEPHOSPHATE_ISOMERASE, NONE,
                REACTION_PROBABILITY, DISSOCIATION_PROBABILITY);
            break;

        case GLYCERALDEHYDE_3_PHOSPHATE_DEHYDROGENASE_COMPLEX:
            handleComplexReaction(molecules, idx, states, creationBuffer, numCreations, deletionBuffer, numDeletions,
                GLYCERALDEHYDE_3_PHOSPHATE_DEHYDROGENASE_COMPLEX,
                GLYCERALDEHYDE_3_PHOSPHATE, GLYCERALDEHYDE_3_PHOSPHATE_DEHYDROGENASE, NONE,
                NONE, NONE, NONE, // No reaction products
                0.0f, DISSOCIATION_PROBABILITY);
            break;

        case GLYCERALDEHYDE_3_PHOSPHATE_DEHYDROGENASE_NAD_PLUS_COMPLEX:
            handleComplexReaction(molecules, idx, states, creationBuffer, numCreations, deletionBuffer, numDeletions,
                GLYCERALDEHYDE_3_PHOSPHATE_DEHYDROGENASE_NAD_PLUS_COMPLEX,
                GLYCERALDEHYDE_3_PHOSPHATE_DEHYDROGENASE_COMPLEX, NAD_PLUS, NONE,
                NONE, NONE, NONE, // No reaction products
                0.0f, DISSOCIATION_PROBABILITY);
            break;

        case GLYCERALDEHYDE_3_PHOSPHATE_DEHYDROGENASE_NAD_PLUS_INORGANIC_PHOSPHATE_COMPLEX:
            handleComplexReaction(molecules, idx, states, creationBuffer, numCreations, deletionBuffer, numDeletions,
                GLYCERALDEHYDE_3_PHOSPHATE_DEHYDROGENASE_NAD_PLUS_INORGANIC_PHOSPHATE_COMPLEX, 
                GLYCERALDEHYDE_3_PHOSPHATE_DEHYDROGENASE_COMPLEX, NAD_PLUS, INORGANIC_PHOSPHATE,
                _1_3_BISPHOSPHOGLYCERATE, NADH, GLYCERALDEHYDE_3_PHOSPHATE_DEHYDROGENASE,
                REACTION_PROBABILITY, DISSOCIATION_PROBABILITY);
            break;

        case PHOSPHOGLYCERATE_KINASE_COMPLEX:
            handleComplexReaction(molecules, idx, states, creationBuffer, numCreations, deletionBuffer, numDeletions,
                PHOSPHOGLYCERATE_KINASE_COMPLEX,
                _1_3_BISPHOSPHOGLYCERATE, PHOSPHOGLYCERATE_KINASE, NONE, 
                NONE, NONE, NONE, // No reaction products
                0.0f, DISSOCIATION_PROBABILITY);
            break;

        case PHOSPHOGLYCERATE_KINASE_ADP_COMPLEX:
            handleComplexReaction(molecules, idx, states, creationBuffer, numCreations, deletionBuffer, numDeletions,
                PHOSPHOGLYCERATE_KINASE_ADP_COMPLEX,
                _1_3_BISPHOSPHOGLYCERATE, PHOSPHOGLYCERATE_KINASE, NONE,
                _3_PHOSPHOGLYCERATE, ATP, PHOSPHOGLYCERATE_KINASE,
                REACTION_PROBABILITY, DISSOCIATION_PROBABILITY);
            break;

        case PHOSPHOGLYCERATE_MUTASE_COMPLEX:
            handleComplexReaction(molecules, idx, states, creationBuffer, numCreations, deletionBuffer, numDeletions,
                PHOSPHOGLYCERATE_MUTASE_COMPLEX,
                _3_PHOSPHOGLYCERATE, PHOSPHOGLYCERATE_MUTASE, NONE,
                _2_PHOSPHOGLYCERATE, PHOSPHOGLYCERATE_MUTASE, NONE,
                REACTION_PROBABILITY, DISSOCIATION_PROBABILITY);
            break;

        case ENOLASE_COMPLEX:
            handleComplexReaction(molecules, idx, states, creationBuffer, numCreations, deletionBuffer, numDeletions,
                ENOLASE_COMPLEX,
                _2_PHOSPHOGLYCERATE, ENOLASE, NONE,
                PHOSPHOENOLPYRUVATE, ENOLASE, NONE,
                REACTION_PROBABILITY, DISSOCIATION_PROBABILITY);
            break;

        case PYRUVATE_KINASE_COMPLEX:
            handleComplexReaction(molecules, idx, states, creationBuffer, numCreations, deletionBuffer, numDeletions,
                PYRUVATE_KINASE_COMPLEX,
                PHOSPHOENOLPYRUVATE, PYRUVATE_KINASE, NONE,
                NONE, NONE, NONE, // No reaction products
                0.0f, DISSOCIATION_PROBABILITY);
            break;

        case PYRUVATE_KINASE_ADP_COMPLEX:
            handleComplexReaction(molecules, idx, states, creationBuffer, numCreations, deletionBuffer, numDeletions,
                PYRUVATE_KINASE_ADP_COMPLEX,
                PYRUVATE_KINASE_COMPLEX, ADP, NONE,
                PYRUVATE, ATP, PYRUVATE_KINASE,
                REACTION_PROBABILITY, DISSOCIATION_PROBABILITY);
            break;
    }
}

__device__ void processBindingReaction(Molecule* molecules, int* numDeletions, int* numCreations, int* deletionBuffer, MoleculeCreationInfo* creationBuffer,
    curandState* states,
    int idx1, int idx2, MoleculeType product1, MoleculeType product2, MoleculeType product3 = NONE,
    float reactionProbability = 1.0f) {

    printf("Processing binding reaction with reactants");

    // Check that the reaction proceeds
    if (curand_uniform(&states[idx1]) < 1- reactionProbability) return;

    printf("Processing binding reaction successfully!\n");

    // Delete reactants
    int delIdx = atomicAdd(numDeletions, 2);
    deletionBuffer[delIdx] = idx1;
    deletionBuffer[delIdx + 1] = idx2;

    // Create products
    int createIdx = atomicAdd(numCreations, 1);
    creationBuffer[createIdx] = {product1, molecules[idx1].centerOfMass.x, molecules[idx1].centerOfMass.y, molecules[idx1].centerOfMass.z};

    if (product2 != NONE) {
        createIdx = atomicAdd(numCreations, 1);
        creationBuffer[createIdx] = {product2, molecules[idx1].centerOfMass.x, molecules[idx1].centerOfMass.y, molecules[idx1].centerOfMass.z};
    }

    if (product3 != NONE) {
        createIdx = atomicAdd(numCreations, 1);
        creationBuffer[createIdx] = {product3, molecules[idx1].centerOfMass.x, molecules[idx1].centerOfMass.y, molecules[idx1].centerOfMass.z};
    }
}

// Main interaction kernel
__global__ void handleBindings(Molecule* molecules, int* num_molecules, int max_molecules, curandState* states,
                               MoleculeCreationInfo* creationBuffer, int* numCreations,
                               int* deletionBuffer, int* numDeletions) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= *num_molecules) return;

    Molecule& mol1 = molecules[idx];
    curandState localState = states[idx];

    for (int j = idx + 1; j < *num_molecules; j++) {
        Molecule& mol2 = molecules[j];

        if (distanceSquared(mol1, mol2) <= INTERACTION_RADIUS_SQ) {
            // Glucose + Hexokinase -> Hexokinase-Glucose Complex
            if (mol1.type == GLUCOSE && mol2.type == HEXOKINASE) {
                processBindingReaction(molecules, numDeletions, numCreations, deletionBuffer, creationBuffer,
                                states, idx, j, HEXOKINASE_GLUCOSE_COMPLEX, NONE, NONE, REACTION_PROBABILITY);
                printf("Creating HEXOKINASE_GLUCOSE_COMPLEX\n");
                return;
            }

            // Hexokinase-Glucose Complex + ATP -> Hexokinase-Glucose-ATP Complex
            else if (mol1.type == HEXOKINASE_GLUCOSE_COMPLEX && mol2.type == ATP) {
                printf("ATP with Hexokinase-Glucose Complex encountered\n");
                processBindingReaction(molecules, numDeletions, numCreations, deletionBuffer, creationBuffer,
                                states, idx, j, HEXOKINASE_GLUCOSE_ATP_COMPLEX, NONE, NONE, REACTION_PROBABILITY);
                return;
            }

            // Glucose-6-Phosphate + Isomerase -> Glucose-6-Phosphate Isomerase Complex
            else if (mol1.type == GLUCOSE_6_PHOSPHATE && mol2.type == PHOSPHOGLUCOSE_ISOMERASE) {
                processBindingReaction(molecules, numDeletions, numCreations, deletionBuffer, creationBuffer,
                                states, idx, j, GLUCOSE_6_PHOSPHATE_ISOMERASE_COMPLEX, NONE, NONE, REACTION_PROBABILITY);
                return;
            }

            // Fructose-6-Phosphate + Isomerase -> Fructose-6-Phosphate Isomerase Complex
            else if (mol1.type == FRUCTOSE_6_PHOSPHATE && mol2.type == PHOSPHOGLUCOSE_ISOMERASE) {
                processBindingReaction(molecules, numDeletions, numCreations, deletionBuffer, creationBuffer,
                                states, idx, j, FRUCTOSE_6_PHOSPHATE_ISOMERASE_COMPLEX, NONE, NONE, REACTION_PROBABILITY);
                return;
            }

            // Fructose-6-Phosphate + Phosphofructokinase-1 -> Phosphofructokinase-1 Complex
            else if (mol1.type == FRUCTOSE_6_PHOSPHATE && mol2.type == PHOSPHOFRUCTOKINASE_1) {
                processBindingReaction(molecules, numDeletions, numCreations, deletionBuffer, creationBuffer,
                                states, idx, j, PHOSPHOFRUCTOKINASE_1_COMPLEX, NONE, NONE, REACTION_PROBABILITY);
                return;
            }

            // Phosphofructokinase-1 Complex + ATP -> Phosphofructokinase-1-ATP Complex
            else if (mol1.type == PHOSPHOFRUCTOKINASE_1_COMPLEX && mol2.type == ATP) {
                processBindingReaction(molecules, numDeletions, numCreations, deletionBuffer, creationBuffer,
                                states, idx, j, PHOSPHOFRUCTOKINASE_1_ATP_COMPLEX, NONE, NONE, REACTION_PROBABILITY);
                return;
            }

            // Fructose-1,6-Bisphosphate + Aldolase -> Fructose-1,6-Bisphosphate-Aldolase Complex
            else if (mol1.type == FRUCTOSE_1_6_BISPHOSPHATE && mol2.type == ALDOLASE) {
                processBindingReaction(molecules, numDeletions, numCreations, deletionBuffer, creationBuffer,
                                states, idx, j, FRUCTOSE_1_6_BISPHOSPHATE_ALDOLASE_COMPLEX, NONE, NONE, REACTION_PROBABILITY);
                return;
            }

            // G3P + Aldolase -> G3P-Aldolase Complex
            else if (mol1.type == GLYCERALDEHYDE_3_PHOSPHATE && mol2.type == ALDOLASE) {
                processBindingReaction(molecules, numDeletions, numCreations, deletionBuffer, creationBuffer,
                                states, idx, j, GLYCERALDEHYDE_3_PHOSPHATE_ALDOLASE_COMPLEX, NONE, NONE, REACTION_PROBABILITY);
                return;
            }

            // G3P-Aldolase Complex + DHAP -> G3P-Aldolase-DHAP Complex
            else if (mol1.type == GLYCERALDEHYDE_3_PHOSPHATE_ALDOLASE_COMPLEX && mol2.type == DIHYDROXYACETONE_PHOSPHATE) {
                processBindingReaction(molecules, numDeletions, numCreations, deletionBuffer, creationBuffer,
                                states, idx, j, GLYCERALDEHYDE_3_PHOSPHATE_ALDOLASE_DHAP_COMPLEX, NONE, NONE, REACTION_PROBABILITY);
                return;
            }

            // Dihydroxyacetone Phosphate + Triosephosphate Isomerase -> Triosephosphate Isomerase Complex
            else if (mol1.type == DIHYDROXYACETONE_PHOSPHATE && mol2.type == TRIOSEPHOSPHATE_ISOMERASE) {
                processBindingReaction(molecules, numDeletions, numCreations, deletionBuffer, creationBuffer,
                                states, idx, j, DHAP_TRIOSEPHOSPHATE_ISOMERASE_COMPLEX, NONE, NONE, REACTION_PROBABILITY);
                return;
            }

            // G3P + Triosephosphate Isomerase -> G3P-Triosephosphate Isomerase Complex
            else if (mol1.type == GLYCERALDEHYDE_3_PHOSPHATE && mol2.type == TRIOSEPHOSPHATE_ISOMERASE) {
                processBindingReaction(molecules, numDeletions, numCreations, deletionBuffer, creationBuffer,
                                states, idx, j, GLYCERALDEHYDE_3_PHOSPHATE_TRIOSEPHOSPHATE_ISOMERASE_COMPLEX, NONE, NONE, REACTION_PROBABILITY);    
                return;
            }

            // Glyceraldehyde-3-Phosphate + Glyceraldehyde-3-Phosphate Dehydrogenase -> Glyceraldehyde-3-Phosphate Dehydrogenase Complex
            else if (mol1.type == GLYCERALDEHYDE_3_PHOSPHATE && mol2.type == GLYCERALDEHYDE_3_PHOSPHATE_DEHYDROGENASE) {
                processBindingReaction(molecules, numDeletions, numCreations, deletionBuffer, creationBuffer,
                                states, idx, j, GLYCERALDEHYDE_3_PHOSPHATE_DEHYDROGENASE_COMPLEX, NONE, NONE, REACTION_PROBABILITY);
                return;
            }

            // Glyceraldehyde-3-Phosphate Dehydrogenase Complex + NAD+ -> Glyceraldehyde-3-Phosphate Dehydrogenase Complex-NAD+
            else if (mol1.type == GLYCERALDEHYDE_3_PHOSPHATE_DEHYDROGENASE_COMPLEX && mol2.type == NAD_PLUS) {
                processBindingReaction(molecules, numDeletions, numCreations, deletionBuffer, creationBuffer,
                                states, idx, j, GLYCERALDEHYDE_3_PHOSPHATE_DEHYDROGENASE_NAD_PLUS_COMPLEX, NONE, NONE, REACTION_PROBABILITY);
                return;
            }

            // Glyceraldehyde-3-Phosphate Dehydrogenase Complex-NAD+ + Pi -> Glyceraldehyde-3-Phosphate Dehydrogenase Complex-NAD+-Pi
            else if (mol1.type == GLYCERALDEHYDE_3_PHOSPHATE_DEHYDROGENASE_NAD_PLUS_COMPLEX && mol2.type == INORGANIC_PHOSPHATE) {
                processBindingReaction(molecules, numDeletions, numCreations, deletionBuffer, creationBuffer,
                                states, idx, j, GLYCERALDEHYDE_3_PHOSPHATE_DEHYDROGENASE_NAD_PLUS_INORGANIC_PHOSPHATE_COMPLEX, NONE, NONE, REACTION_PROBABILITY);
                return;
            }

            // 1,3-Bisphosphoglycerate + Phosphoglycerate Kinase -> Phosphoglycerate Kinase Complex
            else if (mol1.type == _1_3_BISPHOSPHOGLYCERATE && mol2.type == PHOSPHOGLYCERATE_KINASE) {
                processBindingReaction(molecules, numDeletions, numCreations, deletionBuffer, creationBuffer,
                                states, idx, j, PHOSPHOGLYCERATE_KINASE_COMPLEX, NONE, NONE, REACTION_PROBABILITY);
                return;
            }

            // Phosphoglycerate Kinase Complex + ADP -> Phosphoglycerate Kinase ADP Complex
            else if (mol1.type == PHOSPHOGLYCERATE_KINASE_COMPLEX && mol2.type == ADP) {
                processBindingReaction(molecules, numDeletions, numCreations, deletionBuffer, creationBuffer,
                                states, idx, j, PHOSPHOGLYCERATE_KINASE_ADP_COMPLEX, NONE, NONE, REACTION_PROBABILITY);
                return;
            }

            // 3-Phosphoglycerate + Phosphoglycerate Mutase -> Phosphoglycerate Mutase Complex
            else if (mol1.type == _3_PHOSPHOGLYCERATE && mol2.type == PHOSPHOGLYCERATE_MUTASE) {
                processBindingReaction(molecules, numDeletions, numCreations, deletionBuffer, creationBuffer,
                                states, idx, j, PHOSPHOGLYCERATE_MUTASE_COMPLEX, NONE, NONE, REACTION_PROBABILITY);
                return;
            }

            // 2-Phosphoglycerate + Enolase -> Enolase Complex
            else if (mol1.type == _2_PHOSPHOGLYCERATE && mol2.type == ENOLASE) {
                processBindingReaction(molecules, numDeletions, numCreations, deletionBuffer, creationBuffer,
                                states, idx, j, ENOLASE_COMPLEX, NONE, NONE, REACTION_PROBABILITY);
                return;
            }

            // Phosphoenolpyruvate + Pyruvate Kinase -> Pyruvate Kinase Complex
            else if (mol1.type == PHOSPHOENOLPYRUVATE && mol2.type == PYRUVATE_KINASE) {
                processBindingReaction(molecules, numDeletions, numCreations, deletionBuffer, creationBuffer,
                                states, idx, j, PYRUVATE_KINASE_COMPLEX, NONE, NONE, REACTION_PROBABILITY);
                return;
            }

            // Pyruvate Kinase Complex + ADP -> Pyruvate Kinase ADP Complex
            else if (mol1.type == PYRUVATE_KINASE_COMPLEX && mol2.type == ADP) {
                processBindingReaction(molecules, numDeletions, numCreations, deletionBuffer, creationBuffer,
                                states, idx, j, PYRUVATE_KINASE_ADP_COMPLEX, NONE, NONE, REACTION_PROBABILITY);
                return;
            }
        }
    }

    states[idx] = localState;
}

// Kernel to apply forces and update positions
__global__ void applyForcesAndUpdatePositions(Molecule* molecules, int num_molecules, SimulationSpace space, double dt, curandState* randStates) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_molecules) {
        Molecule& mol = molecules[idx];

        // Using double precision for gamma and D
        double gamma = 6.0 * 3.14159265358979323846 * mol.radius * VISCOSITY;

        if (idx == 0) {
            //printf("gamma: %e\n", gamma);
        }

        double D = (K_BOLTZMANN * TEMPERATURE) / gamma; // nm^2/s
        if (idx == 0) {
            //printf("D: %e\n", D);
        }

        // Convert D to nm^2/s accounting for 1 m2 = 1e18 nm2
        D *= 1e18;

        if (idx == 0) {
            //printf("D converted: %e\n", D);
        }

        // Convert D to nm^2/μs accounting for 1 μs = 1e-6 s
        D *= 1e-6;

        if (idx == 0) {
            //printf("D final converted: %e\n", D);
        }

        // Random displacement due to Brownian motion
        curandState localState = randStates[idx];
        double sqrtTerm = sqrt(2.0 * D * 1); // D in nm^2/μs, dt in μs

        if (idx == 0) {
            //printf("sqrtTerm: %e\n", sqrtTerm);
        }

        double3 randomDisplacement;
        randomDisplacement.x = curand_normal(&localState) * sqrtTerm;
        randomDisplacement.y = curand_normal(&localState) * sqrtTerm;
        randomDisplacement.z = curand_normal(&localState) * sqrtTerm;

        if (idx == 0) {
            //printf("randomDisplacement: %e, %e, %e\n", randomDisplacement.x, randomDisplacement.y, randomDisplacement.z);
        }

        // Update position
        mol.centerOfMass.x += randomDisplacement.x;
        mol.centerOfMass.y += randomDisplacement.y;
        mol.centerOfMass.z += randomDisplacement.z;

        // Handle boundary conditions
        // Bounce off walls
        if (mol.centerOfMass.x < 0 || mol.centerOfMass.x > space.width) {
            mol.centerOfMass.x = fmaxf(0.0f, fminf(mol.centerOfMass.x, space.width));
        }
        if (mol.centerOfMass.y < 0 || mol.centerOfMass.y > space.height) {
            mol.centerOfMass.y = fmaxf(0.0f, fminf(mol.centerOfMass.y, space.height));
        }
        if (mol.centerOfMass.z < 0 || mol.centerOfMass.z > space.depth) {
            mol.centerOfMass.z = fmaxf(0.0f, fminf(mol.centerOfMass.z, space.depth));
        }

        // Update the random state
        randStates[idx] = localState;

        if (idx == 0) {
            //printf("gamma: %e\n", gamma);
            //printf("D: %e\n", D);
        }
    }
}
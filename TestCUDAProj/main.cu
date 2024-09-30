﻿#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <curand.h>
#include <curand_kernel.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include "SimulationSpace.h"
#include "Molecule.cuh"
#include "Atom.cuh"
#include "kernel.cuh"

// Define constants
#define MAX_MOLECULES 1000000
#define MAX_MOLECULE_TYPES 33
#define NUM_REACTION_TYPES 10 // Update this as you add more reaction types

// Constants for force calculations
#define COULOMB_CONSTANT 8.99e9f  // N*m^2/C^2
#define CUTOFF_DISTANCE 2.0f      // nm
#define CUTOFF_DISTANCE_SQ (CUTOFF_DISTANCE * CUTOFF_DISTANCE)
#define EPSILON_0 8.854187817e-12f // Vacuum permittivity
#define K_BOLTZMANN 1.380649e-23f  // Boltzmann constant
#define TEMPERATURE 310.15f        // Temperature in Kelvin (37°C)
#define SOLVENT_DIELECTRIC 78.5f   // Dielectric constant of water at 37°C

// Update these constants at the top of the file
#define MAX_THREADS_PER_BLOCK 1024
#define MAX_BLOCKS 65535

// Function prototypes
__device__ float3 calculatePairwiseForce(const Atom& atom1, const Atom& atom2, float invDist, float distSq);
__global__ void calculateForces(Molecule* molecules, int num_molecules, float3* forces);
__global__ void applyForcesAndUpdatePositions(Molecule* molecules, float3* forces, int num_molecules, SimulationSpace space, float dt);
cudaError_t runSimulation(SimulationSpace* space, Molecule* molecules, int num_ticks);

// Add this function before the main() function

const char* getMoleculeTypeName(MoleculeType type) {
    switch(type) {
        case GLUCOSE: return "GLUCOSE";
        case ATP: return "ATP";
        case ADP: return "ADP";
        case GLUCOSE_6_PHOSPHATE: return "GLUCOSE_6_PHOSPHATE";
        case FRUCTOSE_6_PHOSPHATE: return "FRUCTOSE_6_PHOSPHATE";
        case FRUCTOSE_1_6_BISPHOSPHATE: return "FRUCTOSE_1_6_BISPHOSPHATE";
        case DIHYDROXYACETONE_PHOSPHATE: return "DIHYDROXYACETONE_PHOSPHATE";
        case GLYCERALDEHYDE_3_PHOSPHATE: return "GLYCERALDEHYDE_3_PHOSPHATE";
        case _1_3_BISPHOSPHOGLYCERATE: return "1_3_BISPHOSPHOGLYCERATE";
        case _3_PHOSPHOGLYCERATE: return "3_PHOSPHOGLYCERATE";
        case _2_PHOSPHOGLYCERATE: return "2_PHOSPHOGLYCERATE";
        case PHOSPHOENOLPYRUVATE: return "PHOSPHOENOLPYRUVATE";
        case PYRUVATE: return "PYRUVATE";
        case NAD_PLUS: return "NAD_PLUS";
        case NADH: return "NADH";
        case PROTON: return "PROTON";
        case INORGANIC_PHOSPHATE: return "INORGANIC_PHOSPHATE";
        case WATER: return "WATER";
        case HEXOKINASE: return "HEXOKINASE";
        case GLUCOSE_6_PHOSPHATE_ISOMERASE: return "GLUCOSE_6_PHOSPHATE_ISOMERASE";
        case PHOSPHOFRUCTOKINASE_1: return "PHOSPHOFRUCTOKINASE_1";
        case ALDOLASE: return "ALDOLASE";
        case TRIOSEPHOSPHATE_ISOMERASE: return "TRIOSEPHOSPHATE_ISOMERASE";
        case GLYCERALDEHYDE_3_PHOSPHATE_DEHYDROGENASE: return "GLYCERALDEHYDE_3_PHOSPHATE_DEHYDROGENASE";
        case PHOSPHOGLYCERATE_KINASE: return "PHOSPHOGLYCERATE_KINASE";
        case PHOSPHOGLYCERATE_MUTASE: return "PHOSPHOGLYCERATE_MUTASE";
        case ENOLASE: return "ENOLASE";
        case PYRUVATE_KINASE: return "PYRUVATE_KINASE";
        case AMP: return "AMP";
        case CITRATE: return "CITRATE";
        case FRUCTOSE_2_6_BISPHOSPHATE: return "FRUCTOSE_2_6_BISPHOSPHATE";
        default: return "UNKNOWN";
    }
}

// Host function to create a molecule
Molecule createMolecule(MoleculeType type) {
    switch (type) {
        // Substrates and products
        case GLUCOSE:
            return Molecule::createGlucose();
        case ATP:
            return Molecule::createATP();
        case ADP:
            return Molecule::createADP();
        case GLUCOSE_6_PHOSPHATE:
            return Molecule::createGlucose6Phosphate();
        case FRUCTOSE_6_PHOSPHATE:
            return Molecule::createFructose6Phosphate();
        case FRUCTOSE_1_6_BISPHOSPHATE:
            return Molecule::createFructose16Bisphosphate();
        case DIHYDROXYACETONE_PHOSPHATE:
            return Molecule::createDihydroxyacetonePhosphate();
        case GLYCERALDEHYDE_3_PHOSPHATE:
            return Molecule::createGlyceraldehyde3Phosphate();
        case _1_3_BISPHOSPHOGLYCERATE:
            return Molecule::create13Bisphosphoglycerate();
        case _3_PHOSPHOGLYCERATE:
            return Molecule::create3Phosphoglycerate();
        case _2_PHOSPHOGLYCERATE:
            return Molecule::create2Phosphoglycerate();
        case PHOSPHOENOLPYRUVATE:
            return Molecule::createPhosphoenolpyruvate();
        case PYRUVATE:
            return Molecule::createPyruvate();
        case NAD_PLUS:
            return Molecule::createNADPlus();
        case NADH:
            return Molecule::createNADH();
        case PROTON:
            return Molecule::createProton();
        case INORGANIC_PHOSPHATE:
            return Molecule::createInorganicPhosphate();
        case WATER:
            return Molecule::createWater();

        // Enzymes
        case HEXOKINASE:
            return Molecule::createHexokinase();
        case GLUCOSE_6_PHOSPHATE_ISOMERASE:
            return Molecule::createGlucose6PhosphateIsomerase();
        case PHOSPHOFRUCTOKINASE_1:
            return Molecule::createPhosphofructokinase1();
        case ALDOLASE:
            return Molecule::createAldolase();
        case TRIOSEPHOSPHATE_ISOMERASE:
            return Molecule::createTriosephosphateIsomerase();
        case GLYCERALDEHYDE_3_PHOSPHATE_DEHYDROGENASE:
            return Molecule::createGlyceraldehyde3PhosphateDehydrogenase();
        case PHOSPHOGLYCERATE_KINASE:
            return Molecule::createPhosphoglycerateKinase();
        case PHOSPHOGLYCERATE_MUTASE:
            return Molecule::createPhosphoglycerateMutase();
        case ENOLASE:
            return Molecule::createEnolase();
        case PYRUVATE_KINASE:
            return Molecule::createPyruvateKinase();

        // Regulatory molecules
        case AMP:
            return Molecule::createAMP();
        case CITRATE:
            return Molecule::createCitrate();
        case FRUCTOSE_2_6_BISPHOSPHATE:
            return Molecule::createFructose26Bisphosphate();

        default:
            fprintf(stderr, "Unknown molecule type\n");
            return Molecule::createWater();  // Default case, could also throw an exception
    }
}

// Host function to process creation and deletion flags
void processCreationDeletionFlags(Molecule* molecules, int* num_molecules, int max_molecules, 
                                  const MoleculeCreationInfo* creationBuffer, int numCreations,
                                  const int* deletionBuffer, int numDeletions) {
    // Process deletions
    for (int i = 0; i < numDeletions; i++) {
        int indexToDelete = deletionBuffer[i];
        if (indexToDelete < *num_molecules - 1) {
            molecules[indexToDelete] = molecules[*num_molecules - 1];
        }
        (*num_molecules)--;
    }

    // Process creations
    for (int i = 0; i < numCreations; i++) {
        if (*num_molecules < max_molecules) {
            Molecule newMolecule = createMolecule(creationBuffer[i].type);
            newMolecule.setPosition(creationBuffer[i].x, creationBuffer[i].y, creationBuffer[i].z);
            molecules[*num_molecules] = newMolecule;
            (*num_molecules)++;
        }
    }
}

// Function to run the simulation
cudaError_t runSimulation(SimulationSpace* space, Molecule* molecules, int num_ticks) {
    printf("Starting simulation with %d molecules for %d ticks\n", space->num_molecules, num_ticks);

    Molecule* dev_molecules = nullptr;
    float3* dev_forces = nullptr;
    cudaError_t cudaStatus;
    int threadsPerBlock = 256;
    int blocksPerGrid = min((space->num_molecules + threadsPerBlock - 1) / threadsPerBlock, MAX_BLOCKS);

    printf("Using %d threads per block and %d blocks per grid\n", threadsPerBlock, blocksPerGrid);

    // Declare these vectors at the beginning of the function
    std::vector<MoleculeCreationInfo> h_creationBuffer;
    std::vector<int> h_deletionBuffer;

    // Choose which GPU to run on, change this on a multi-GPU system.
    cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaSetDevice failed! Do you have a CUDA-capable GPU installed?\n");
        goto Error;
    }
    printf("CUDA device set successfully\n");

    // Allocate GPU buffers
    cudaStatus = cudaMalloc((void**)&dev_molecules, space->num_molecules * sizeof(Molecule));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed for dev_molecules!\n");
        goto Error;
    }
    printf("Allocated GPU memory for molecules\n");

    cudaStatus = cudaMalloc((void**)&dev_forces, space->num_molecules * sizeof(float3));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed for dev_forces!\n");
        goto Error;
    }
    printf("Allocated GPU memory for forces\n");

    // Copy molecules to GPU
    cudaStatus = cudaMemcpy(dev_molecules, molecules, space->num_molecules * sizeof(Molecule), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed for molecules!\n");
        goto Error;
    }
    printf("Copied molecules to GPU\n");

    // Allocate and initialize curandState
    curandState* dev_states;
    cudaMalloc(&dev_states, space->num_molecules * sizeof(curandState));
    initCurand<<<blocksPerGrid, threadsPerBlock>>>(time(NULL), dev_states, space->num_molecules);
    printf("Initialized curandState\n");

    // Allocate memory for reaction counts
    int* dev_reactionCounts;
    cudaMalloc(&dev_reactionCounts, NUM_REACTION_TYPES * sizeof(int));
    cudaMemset(dev_reactionCounts, 0, NUM_REACTION_TYPES * sizeof(int));
    printf("Allocated and initialized reaction counts\n");

    // Allocate GPU buffers for num_molecules
    int* dev_num_molecules;
    cudaStatus = cudaMalloc((void**)&dev_num_molecules, sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed for dev_num_molecules!\n");
        goto Error;
    }

    // Copy num_molecules to GPU
    cudaStatus = cudaMemcpy(dev_num_molecules, &space->num_molecules, sizeof(int), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed for dev_num_molecules!\n");
        goto Error;
    }
    printf("Allocated and copied num_molecules to GPU\n");

    // Allocate GPU buffers for creation and deletion
    MoleculeCreationInfo* dev_creationBuffer;
    int* dev_numCreations;
    int* dev_deletionBuffer;
    int* dev_numDeletions;

    cudaStatus = cudaMalloc((void**)&dev_creationBuffer, MAX_MOLECULES * sizeof(MoleculeCreationInfo));
    cudaStatus = cudaMalloc((void**)&dev_numCreations, sizeof(int));
    cudaStatus = cudaMalloc((void**)&dev_deletionBuffer, MAX_MOLECULES * sizeof(int));
    cudaStatus = cudaMalloc((void**)&dev_numDeletions, sizeof(int));

    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed for creation/deletion buffers!\n");
        goto Error;
    }
    printf("Allocated GPU memory for creation and deletion buffers\n");

    // Initialize numCreations and numDeletions to 0
    cudaMemset(dev_numCreations, 0, sizeof(int));
    cudaMemset(dev_numDeletions, 0, sizeof(int));
    printf("Initialized creation and deletion counters\n");

    // Main simulation loop
    for (int tick = 0; tick < num_ticks; tick++) {
        printf("Starting tick %d\n", tick);

        // Calculate forces
        calculateForces<<<blocksPerGrid, threadsPerBlock>>>(dev_molecules, space->num_molecules, dev_forces);
        printf("  Calculated forces\n");

        // Apply forces and update positions
        applyForcesAndUpdatePositions<<<blocksPerGrid, threadsPerBlock>>>(dev_molecules, dev_forces, space->num_molecules, *space, 0.01f);
        printf("  Applied forces and updated positions\n");

        handleInteractions<<<blocksPerGrid, threadsPerBlock>>>(dev_molecules, dev_num_molecules, MAX_MOLECULES, dev_states, dev_reactionCounts, dev_creationBuffer, dev_numCreations, dev_deletionBuffer, dev_numDeletions);
        printf("  Handled interactions\n");

        // Copy creation and deletion buffers from device to host
        int h_numCreations, h_numDeletions;
        cudaMemcpy(&h_numCreations, dev_numCreations, sizeof(int), cudaMemcpyDeviceToHost);
        cudaMemcpy(&h_numDeletions, dev_numDeletions, sizeof(int), cudaMemcpyDeviceToHost);

        h_creationBuffer.resize(h_numCreations);
        h_deletionBuffer.resize(h_numDeletions);
        cudaMemcpy(h_creationBuffer.data(), dev_creationBuffer, h_numCreations * sizeof(MoleculeCreationInfo), cudaMemcpyDeviceToHost);
        cudaMemcpy(h_deletionBuffer.data(), dev_deletionBuffer, h_numDeletions * sizeof(int), cudaMemcpyDeviceToHost);

        printf("  Copied creation and deletion data to host. Creations: %d, Deletions: %d\n", h_numCreations, h_numDeletions);

        // Process creation and deletion flags on the host
        processCreationDeletionFlags(molecules, &space->num_molecules, MAX_MOLECULES, h_creationBuffer.data(), h_numCreations, h_deletionBuffer.data(), h_numDeletions);
        printf("  Processed creation and deletion flags. New molecule count: %d\n", space->num_molecules);

        // Update device memory with new molecule data
        cudaMemcpy(dev_molecules, molecules, space->num_molecules * sizeof(Molecule), cudaMemcpyHostToDevice);
        cudaMemcpy(dev_num_molecules, &space->num_molecules, sizeof(int), cudaMemcpyHostToDevice);
        printf("  Updated GPU with new molecule data\n");

        // Reset numCreations and numDeletions to 0 for the next tick
        cudaMemset(dev_numCreations, 0, sizeof(int));
        cudaMemset(dev_numDeletions, 0, sizeof(int));

        // Check for errors after each kernel launch
        cudaStatus = cudaGetLastError();
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "Kernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
            goto Error;
        }

        cudaStatus = cudaDeviceSynchronize();
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching kernels!\n", cudaStatus);
            goto Error;
        }

        printf("Completed tick %d\n", tick);
    }

    // Copy results back to host
    cudaStatus = cudaMemcpy(molecules, dev_molecules, space->num_molecules * sizeof(Molecule), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "Final cudaMemcpy failed!\n");
        goto Error;
    }
    printf("Copied final results back to host\n");

    printf("Simulation completed successfully\n");

Error:
    printf("Freeing GPU memory\n");
    cudaFree(dev_molecules);
    cudaFree(dev_forces);
    cudaFree(dev_states);
    cudaFree(dev_reactionCounts);
    cudaFree(dev_num_molecules);
    cudaFree(dev_creationBuffer);
    cudaFree(dev_numCreations);
    cudaFree(dev_deletionBuffer);
    cudaFree(dev_numDeletions);
    
    return cudaStatus;
}

// Main function
int main() {
    // Get and print CUDA device properties
    cudaDeviceProp deviceProp;
    cudaError_t cudaStatus = cudaGetDeviceProperties(&deviceProp, 0);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaGetDeviceProperties failed! Error: %s\n", cudaGetErrorString(cudaStatus));
        return 1;
    }

    printf("CUDA Device Properties:\n");
    printf("  Device name: %s\n", deviceProp.name);
    printf("  Compute capability: %d.%d\n", deviceProp.major, deviceProp.minor);
    printf("  Total global memory: %zu bytes\n", deviceProp.totalGlobalMem);
    printf("  Max threads per block: %d\n", deviceProp.maxThreadsPerBlock);
    printf("  Max threads dim: (%d, %d, %d)\n", deviceProp.maxThreadsDim[0], deviceProp.maxThreadsDim[1], deviceProp.maxThreadsDim[2]);
    printf("  Max grid size: (%d, %d, %d)\n", deviceProp.maxGridSize[0], deviceProp.maxGridSize[1], deviceProp.maxGridSize[2]);
    printf("  Warp size: %d\n", deviceProp.warpSize);
    printf("  Memory clock rate: %d kHz\n", deviceProp.memoryClockRate);
    printf("  Memory bus width: %d bits\n", deviceProp.memoryBusWidth);
    printf("\n");

    SimulationSpace space;
    Molecule* molecules = nullptr;
    
    // Read input from file
    FILE* input_file = fopen("input.txt", "r");
    if (input_file == NULL) {
        fprintf(stderr, "Failed to open input.txt\n");
        return 1;
    }
    
    char line[256];
    
    // Read simulation space dimensions
    while (fgets(line, sizeof(line), input_file)) {
        if (line[0] != '#') {
            if (sscanf(line, "%d %d %d", &space.width, &space.height, &space.depth) != 3) {
                fprintf(stderr, "Failed to read simulation space dimensions\n");
                fclose(input_file);
                return 1;
            }
            printf("Simulation space dimensions: %d x %d x %d\n", space.width, space.height, space.depth);
            break;
        }
    }
    
    // Read number of molecule types
    while (fgets(line, sizeof(line), input_file)) {
        if (line[0] != '#') {
            if (sscanf(line, "%d", &space.num_molecule_types) != 1) {
                fprintf(stderr, "Failed to read number of molecule types\n");
                fclose(input_file);
                return 1;
            }
            printf("Number of molecule types: %d\n", space.num_molecule_types);
            break;
        }
    }
    
    // Read molecule counts
    space.num_molecules = 0;
    int molecule_type = 0;
    while (fgets(line, sizeof(line), input_file) && molecule_type < space.num_molecule_types) {
        if (line[0] != '#') {
            char molecule_name[50];
            int count;
            if (sscanf(line, "%[^:]: %d", molecule_name, &count) == 2) {
                space.molecule_counts[molecule_type] = count;
                space.num_molecules += count;
                printf("Molecule type %d (%s): %d\n", molecule_type, molecule_name, count);
                molecule_type++;
            }
        }
    }
    
    fclose(input_file);
    
    printf("Total number of molecules: %d\n", space.num_molecules);
    
    // Before allocating memory for molecules
    if (space.num_molecules > MAX_MOLECULES) {
        fprintf(stderr, "Error: Number of molecules (%d) exceeds maximum allowed (%d)\n", space.num_molecules, MAX_MOLECULES);
        return 1;
    }

    // Allocate memory for molecules
    molecules = (Molecule*)malloc(MAX_MOLECULES * sizeof(Molecule));
    if (molecules == nullptr) {
        fprintf(stderr, "Failed to allocate memory for molecules\n");
        return 1;
    }

    printf("Molecules allocated successfully\n");
    
    // Initialize molecules
    int molecule_index = 0;
    int type_counts[MAX_MOLECULE_TYPES] = {0};  // Array to keep count of each molecule type

    for (int i = 0; i < space.num_molecule_types; i++) {
        MoleculeType currentType = static_cast<MoleculeType>(i);
        int count = space.molecule_counts[i];
        
        printf("Creating %d molecules of type %d (%s)\n", count, i, getMoleculeTypeName(currentType));

        for (int j = 0; j < count; j++) {
            if (molecule_index >= MAX_MOLECULES) {
                fprintf(stderr, "Error: Exceeded maximum number of molecules (%d)\n", MAX_MOLECULES);
                free(molecules);
                return 1;
            }

            molecules[molecule_index] = createMolecule(currentType);

            // Initialize position randomly
            float x = static_cast<float>(rand()) / RAND_MAX * space.width;
            float y = static_cast<float>(rand()) / RAND_MAX * space.height;
            float z = static_cast<float>(rand()) / RAND_MAX * space.depth;
            molecules[molecule_index].setPosition(x, y, z);

            type_counts[i]++;
            molecule_index++;
        }
        printf("Created molecule type %d (%s): %d\n", i, getMoleculeTypeName(currentType), type_counts[i]);
    }

    // Print summary of all molecule types created
    printf("\nSummary of molecules created:\n");
    for (int i = 0; i < space.num_molecule_types; i++) {
        if (type_counts[i] > 0) {
            printf("%s: %d\n", getMoleculeTypeName(static_cast<MoleculeType>(i)), type_counts[i]);
        }
    }

    // Check if we've initialized the correct number of molecules
    if (molecule_index != space.num_molecules) {
        fprintf(stderr, "Error: Initialized %d molecules, expected %d\n", molecule_index, space.num_molecules);
        free(molecules);
        return 1;
    }

    printf("\nTotal molecules initialized successfully: %d\n", molecule_index);

    // Run simulation
    int num_ticks = 1000; // Adjust as needed
    cudaStatus = runSimulation(&space, molecules, num_ticks);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "Simulation failed! Error: %s\n", cudaGetErrorString(cudaStatus));
        free(molecules);
        return 1;
    }

    // Free memory
    free(molecules);

    return 0;
}
// CUDA kernels
__device__ float3 calculatePairwiseForce(const Atom& atom1, const Atom& atom2, float invDist, float distSq) {
    float3 force = make_float3(0.0f, 0.0f, 0.0f);

    // Lennard-Jones potential
    float sigma = 0.5f * (atom1.getVanDerWaalsRadius() + atom2.getVanDerWaalsRadius());
    float epsilon = sqrtf(atom1.getEpsilon() * atom2.getEpsilon()); // Combining rule
    float sigmaOverDist = sigma * invDist;
    float sigmaOverDist6 = sigmaOverDist * sigmaOverDist * sigmaOverDist;
    sigmaOverDist6 = sigmaOverDist6 * sigmaOverDist6;
    float forceMultiplierLJ = 24.0f * epsilon * invDist * sigmaOverDist6 * (1.0f - 2.0f * sigmaOverDist6);

    // Coulomb force with Generalized Born model
    float fGB = sqrtf(distSq + atom1.getBornRadius() * atom2.getBornRadius() * expf(-distSq / (4.0f * atom1.getBornRadius() * atom2.getBornRadius())));
    float dielectric = 1.0f / (1.0f / SOLVENT_DIELECTRIC - 1.0f / 1.0f) * expf(-distSq / (4.0f * atom1.getBornRadius() * atom2.getBornRadius()));
    float forceMultiplierCoulomb = COULOMB_CONSTANT * atom1.getCharge() * atom2.getCharge() * (1.0f / distSq - 1.0f / (fGB * fGB)) / dielectric;

    float totalForceMultiplier = forceMultiplierLJ + forceMultiplierCoulomb;

    force.x = (atom2.getX() - atom1.getX()) * totalForceMultiplier * invDist;
    force.y = (atom2.getY() - atom1.getY()) * totalForceMultiplier * invDist;
    force.z = (atom2.getZ() - atom1.getZ()) * totalForceMultiplier * invDist;

    return force;
}
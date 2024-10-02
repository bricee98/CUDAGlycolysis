#include <GL/glew.h>
#include <GL/gl.h>
#include <GLFW/glfw3.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <curand.h>
#include <curand_kernel.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include "SimulationSpace.h"
#include "Molecule.cuh"
#include "kernel.cuh"
#include "visualization.h"
#include "Cell.cuh"
#include "SimulationData.h"



// Define constants
#define MAX_MOLECULES 2000
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

// Add these declarations
extern int h_GRID_SIZE_X;
extern int h_GRID_SIZE_Y;
extern int h_GRID_SIZE_Z;

// Define window and isPaused
GLFWwindow* window;
bool isPaused = false;

// Function prototypes
cudaError_t runSimulation(SimulationSpace* space, Molecule* molecules, int num_ticks);
cudaError_t runSimulationStep(SimulationSpace* space, Molecule* molecules);

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
    if (deletionBuffer && numDeletions > 0) {
        for (int i = 0; i < numDeletions; i++) {
            int indexToDelete = deletionBuffer[i];
            if (indexToDelete < *num_molecules - 1) {
                molecules[indexToDelete] = molecules[*num_molecules - 1];
            }
            (*num_molecules)--;
        }
    }

    // Process creations
    if (creationBuffer && numCreations > 0) {
        for (int i = 0; i < numCreations; i++) {
            if (*num_molecules < max_molecules) {
                Molecule newMolecule = createMolecule(creationBuffer[i].type);
                newMolecule.centerOfMass = make_float3(creationBuffer[i].x, creationBuffer[i].y, creationBuffer[i].z);
                molecules[*num_molecules] = newMolecule;
                (*num_molecules)++;
            }
        }
    }
}

// Modify the runSimulation function to use runSimulationStep
cudaError_t runSimulation(SimulationSpace* space, Molecule* molecules, int num_ticks) {
    printf("Starting simulation with %d molecules for %d ticks\n", space->num_molecules, num_ticks);

    cudaError_t cudaStatus;

    // Choose which GPU to run on, change this on a multi-GPU system.
    cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaSetDevice failed! Do you have a CUDA-capable GPU installed?\n");
        return cudaStatus;
    }
    printf("CUDA device set successfully\n");

    // Main simulation loop
    for (int tick = 0; tick < num_ticks; tick++) {
        printf("Starting tick %d\n", tick);

        cudaStatus = runSimulationStep(space, molecules);
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "Simulation step failed! Error: %s\n", cudaGetErrorString(cudaStatus));
            break;
        }

        printf("Completed tick %d\n", tick);
    }

    printf("Simulation completed\n");

    return cudaStatus;
}

cudaError_t runSimulationStep(SimulationSpace* space, Molecule* molecules) {
    static Molecule* dev_molecules = nullptr;
    static float3* dev_forces = nullptr;
    static curandState* dev_states = nullptr;
    static int* dev_reactionCounts = nullptr;
    static int* dev_num_molecules = nullptr;
    static MoleculeCreationInfo* dev_creationBuffer = nullptr;
    static int* dev_numCreations = nullptr;
    static int* dev_deletionBuffer = nullptr;
    static int* dev_numDeletions = nullptr;
    static Cell* dev_cells = nullptr;

    // Remove the extern variables and define grid sizes locally
    int gridSizeX = static_cast<int>(space->width / CELL_SIZE);
    int gridSizeY = static_cast<int>(space->height / CELL_SIZE);
    int gridSizeZ = static_cast<int>(space->depth / CELL_SIZE);

    Grid grid;
    grid.sizeX = gridSizeX;
    grid.sizeY = gridSizeY;
    grid.sizeZ = gridSizeZ;

    printf("Grid Sizes - X: %d, Y: %d, Z: %d\n", grid.sizeX, grid.sizeY, grid.sizeZ);

    cudaError_t cudaStatus;
    int threadsPerBlock = 256;
    int blocksPerGrid = min((space->num_molecules + threadsPerBlock - 1) / threadsPerBlock, MAX_BLOCKS);

    // Ensure blocksPerGrid is at least 1
    if (blocksPerGrid < 1) blocksPerGrid = 1;

    // Calculate total cells
    int totalCells = grid.sizeX * grid.sizeY * grid.sizeZ;
    printf("Total cells: %d\n", totalCells);

    if (dev_cells == nullptr) {
        cudaStatus = cudaMalloc(&dev_cells, totalCells * sizeof(Cell));
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "cudaMalloc failed for dev_cells!\n");
            return cudaStatus;
        }
    }

    // Reset cells
    cudaMemset(dev_cells, 0, totalCells * sizeof(Cell));

    // Initialize device memory if it's the first run
    if (dev_molecules == nullptr) {
        // Allocate GPU buffers and copy data
        cudaStatus = cudaMalloc((void**)&dev_molecules, MAX_MOLECULES * sizeof(Molecule));
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "cudaMalloc failed for dev_molecules! Error: %s\n", cudaGetErrorString(cudaStatus));
            return cudaStatus;
        }
        cudaStatus = cudaMalloc((void**)&dev_forces, MAX_MOLECULES * sizeof(float3));
        if (cudaStatus != cudaSuccess) { fprintf(stderr, "cudaMalloc failed for dev_forces!\n"); return cudaStatus; }
        cudaStatus = cudaMalloc((void**)&dev_states, MAX_MOLECULES * sizeof(curandState));
        if (cudaStatus != cudaSuccess) { fprintf(stderr, "cudaMalloc failed for dev_states!\n"); return cudaStatus; }
        cudaStatus = cudaMalloc(&dev_reactionCounts, NUM_REACTION_TYPES * sizeof(int));
        if (cudaStatus != cudaSuccess) { fprintf(stderr, "cudaMalloc failed for dev_reactionCounts!\n"); return cudaStatus; }
        cudaStatus = cudaMalloc((void**)&dev_num_molecules, sizeof(int));
        if (cudaStatus != cudaSuccess) { fprintf(stderr, "cudaMalloc failed for dev_num_molecules!\n"); return cudaStatus; }
        cudaStatus = cudaMalloc((void**)&dev_creationBuffer, MAX_MOLECULES * sizeof(MoleculeCreationInfo));
        if (cudaStatus != cudaSuccess) { fprintf(stderr, "cudaMalloc failed for dev_creationBuffer!\n"); return cudaStatus; }
        cudaStatus = cudaMalloc((void**)&dev_numCreations, sizeof(int));
        if (cudaStatus != cudaSuccess) { fprintf(stderr, "cudaMalloc failed for dev_numCreations!\n"); return cudaStatus; }
        cudaStatus = cudaMalloc((void**)&dev_deletionBuffer, MAX_MOLECULES * sizeof(int));
        if (cudaStatus != cudaSuccess) { fprintf(stderr, "cudaMalloc failed for dev_deletionBuffer!\n"); return cudaStatus; }
        cudaStatus = cudaMalloc((void**)&dev_numDeletions, sizeof(int));
        if (cudaStatus != cudaSuccess) { fprintf(stderr, "cudaMalloc failed for dev_numDeletions!\n"); return cudaStatus; }

        // Initialize curandState
        initCurand<<<blocksPerGrid, threadsPerBlock>>>(time(NULL), dev_states, space->num_molecules);
    }


    // available memory
    size_t availableMemory;
    cudaStatus = cudaMemGetInfo(&availableMemory, NULL);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemGetInfo failed! Error: %s\n", cudaGetErrorString(cudaStatus));
        return cudaStatus;
    }
    printf("Available memory: %zu bytes\n", availableMemory);

    // Copy current state to device
    cudaStatus = cudaMemcpy(dev_molecules, molecules, MAX_MOLECULES * sizeof(Molecule), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed for molecules to dev_molecules! Error: %s\n", cudaGetErrorString(cudaStatus));
        return cudaStatus;
    }

    printf("Copied molecules to device\n");

    cudaStatus = cudaMemcpy(dev_num_molecules, &space->num_molecules, sizeof(int), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) { fprintf(stderr, "cudaMemcpy failed for num_molecules to dev_num_molecules!\n"); return cudaStatus; }
    printf("Copied num_molecules to device\n");

    cudaMemset(dev_reactionCounts, 0, NUM_REACTION_TYPES * sizeof(int));
    cudaMemset(dev_numCreations, 0, sizeof(int));
    cudaMemset(dev_numDeletions, 0, sizeof(int));
    printf("Reset reaction counts, numCreations, and numDeletions\n");
    // Assign molecules to cells
    dim3 gridAssign((space->num_molecules + threadsPerBlock - 1) / threadsPerBlock);
    assignMoleculesToCells<<<gridAssign, threadsPerBlock>>>(dev_molecules, space->num_molecules, dev_cells, *space, grid);
    printf("Assigned molecules to cells\n");
    // Synchronize to ensure kernel execution completion and flush printf output
    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching assignMoleculesToCells kernel!\n", cudaStatus);
        return cudaStatus;
    }


    // Reset forces
    cudaMemset(dev_forces, 0, space->num_molecules * sizeof(float3));
    printf("Reset forces\n");
    // Adjust the timestep for microsecond timescales
    float dt = 1e-6f; // Timestep of 1 microsecond

    // Call applyForcesAndUpdatePositions with the new dt
    applyForcesAndUpdatePositions<<<blocksPerGrid, threadsPerBlock>>>(
        dev_molecules, space->num_molecules, *space, dt, dev_states);

    // Synchronize to ensure kernel execution completion and flush printf output
    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching applyForcesAndUpdatePositions kernel!\n", cudaStatus);
        return cudaStatus;
    }
    printf("Applied forces and updated positions\n");
    // Handle interactions
    handleInteractions<<<blocksPerGrid, threadsPerBlock>>>(dev_molecules, dev_num_molecules, MAX_MOLECULES, dev_states,
                                                           dev_reactionCounts, dev_creationBuffer, dev_numCreations,
                                                           dev_deletionBuffer, dev_numDeletions);

    // Check for errors during kernel launch
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "Error after handleInteractions kernel launch: %s\n", cudaGetErrorString(err));
        return err;
    }

    // Synchronize to ensure kernel execution completion and flush printf output
    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching handleInteractions kernel!\n", cudaStatus);
        return cudaStatus;
    }

    // Copy the updated molecules back to the host
    cudaStatus = cudaMemcpy(molecules, dev_molecules, space->num_molecules * sizeof(Molecule), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed for dev_molecules to molecules!\n");
        return cudaStatus;
    }

    printf("Copied updated molecules to host\n");

    printf("Handled interactions\n");
    // Copy the number of creations and deletions back to the host
    int h_numCreations, h_numDeletions;
    cudaMemcpy(&h_numCreations, dev_numCreations, sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(&h_numDeletions, dev_numDeletions, sizeof(int), cudaMemcpyDeviceToHost);
    printf("Copied numCreations (%d) and numDeletions (%d) to host\n", h_numCreations, h_numDeletions);

    // Allocate host buffers for creations and deletions
    MoleculeCreationInfo* h_creationBuffer = nullptr;
    int* h_deletionBuffer = nullptr;

    if (h_numCreations > 0) {
        h_creationBuffer = new MoleculeCreationInfo[h_numCreations];
        // Copy creation buffer back to the host
        cudaMemcpy(h_creationBuffer, dev_creationBuffer, h_numCreations * sizeof(MoleculeCreationInfo), cudaMemcpyDeviceToHost);
        printf("Allocated and copied creation buffer to host\n");
    }

    if (h_numDeletions > 0) {
        h_deletionBuffer = new int[h_numDeletions];
        // Copy deletion buffer back to the host
        cudaMemcpy(h_deletionBuffer, dev_deletionBuffer, h_numDeletions * sizeof(int), cudaMemcpyDeviceToHost);
        printf("Allocated and copied deletion buffer to host\n");
    }

    // Process creation and deletion flags
    processCreationDeletionFlags(molecules, &space->num_molecules, MAX_MOLECULES,
                                 h_creationBuffer, h_numCreations,
                                 h_deletionBuffer, h_numDeletions);
    printf("Processed creation and deletion flags\n");

    // Free host buffers
    if (h_creationBuffer) {
        delete[] h_creationBuffer;
    }
    if (h_deletionBuffer) {
        delete[] h_deletionBuffer;
    }
    printf("Freed host buffers\n");
    // Copy updated molecules back to the device
    cudaStatus = cudaMemcpy(dev_molecules, molecules, space->num_molecules * sizeof(Molecule), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) { fprintf(stderr, "cudaMemcpy failed for molecules to dev_molecules!\n"); return cudaStatus; }
    printf("Copied updated molecules to device\n");

    // Print the position of molecule 500
    printf("Host Molecule 500: Position (%f, %f, %f)\n", molecules[500].centerOfMass.x, molecules[500].centerOfMass.y, molecules[500].centerOfMass.z);
    printf("Host Molecule 500: Velocity (%f, %f, %f)\n", molecules[500].vx, molecules[500].vy, molecules[500].vz);

    cudaStatus = cudaMemcpy(dev_num_molecules, &space->num_molecules, sizeof(int), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) { fprintf(stderr, "cudaMemcpy failed for num_molecules to dev_num_molecules!\n"); return cudaStatus; }
    printf("Copied updated num_molecules to device\n");
    // Check for errors
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "Kernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
        return cudaStatus;
    }
    printf("Checked for errors\n");
    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching kernels!\n", cudaStatus);
        return cudaStatus;
    }
    printf("Synchronized device\n");
    // Copy updated molecules back to host
    cudaStatus = cudaMemcpy(molecules, dev_molecules, space->num_molecules * sizeof(Molecule), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) { fprintf(stderr, "cudaMemcpy failed for dev_molecules to molecules!\n"); return cudaStatus; }
    printf("Copied updated molecules to host\n");
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
            int read_width, read_height, read_depth;
            if (sscanf(line, "%d %d %d", &read_width, &read_height, &read_depth) == 3) {
                space.width = (float)read_width;
                space.height = (float)read_height;
                space.depth = (float)read_depth;

                printf("Read from file - Space width: %.2f\n", space.width);
                printf("Read from file - Space height: %.2f\n", space.height);
                printf("Read from file - Space depth: %.2f\n", space.depth);
            } else {
                fprintf(stderr, "Failed to read simulation space dimensions\n");
                // Set default values
                space.width = 100.0f;
                space.height = 100.0f;
                space.depth = 100.0f;
            }
            
            printf("Simulation space dimensions: %.2f x %.2f x %.2f\n", space.width, space.height, space.depth);
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

            Molecule newMol = createMolecule(currentType);

            // Initialize position randomly
            float x = static_cast<float>(rand()) / RAND_MAX * space.width;
            float y = static_cast<float>(rand()) / RAND_MAX * space.height;
            float z = static_cast<float>(rand()) / RAND_MAX * space.depth;
            newMol.centerOfMass = make_float3(x, y, z);

            molecules[molecule_index++] = newMol;

            type_counts[i]++;
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

    // Initialize visualization
    initVisualization();

    // Main simulation loop
    while (!glfwWindowShouldClose(window)) {
        if (!isPaused) {
            // Run a single step of the simulation
            cudaStatus = runSimulationStep(&space, molecules);
            if (cudaStatus != cudaSuccess) {
                fprintf(stderr, "Simulation step failed! Error: %s\n", cudaGetErrorString(cudaStatus));
                break;
            }
        }

        // Render the current state of the simulation
        renderSimulation(space, std::vector<Molecule>(molecules, molecules + space.num_molecules));
    }

    // Cleanup
    cleanupVisualization();
    free(molecules);

    return 0;
}
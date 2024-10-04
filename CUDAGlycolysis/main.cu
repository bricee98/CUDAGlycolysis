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
#include <chrono>
#include <numeric>
#include <fstream>
#include <algorithm>
#include <vector>
#include <cassert>
#include <conio.h>
#include <cmath>

std::ofstream logFile("performance_log.txt");

// Define constants
#define MAX_MOLECULE_TYPES 33

// Constants for force calculations
#define COULOMB_CONSTANT 8.99e9f  // N*m^2/C^2
#define CUTOFF_DISTANCE 2.0f      // nm
#define CUTOFF_DISTANCE_SQ (CUTOFF_DISTANCE * CUTOFF_DISTANCE)
#define EPSILON_0 8.854187817e-12f // Vacuum permittivity
#define K_BOLTZMANN 1.380649e-23f  // Boltzmann constant
#define TEMPERATURE 310.15f        // Temperature in Kelvin (37°C)
#define SOLVENT_DIELECTRIC 78.5f   // Dielectric constant of water at 37°C

#define MAX_MOLECULES 6000

int g_threadsToUse;
int g_blocksToUse;

int recommended_max_molecules;
int recommended_max_molecules_per_cell;

// Add these declarations
extern int h_GRID_SIZE_X;
extern int h_GRID_SIZE_Y;
extern int h_GRID_SIZE_Z;

// Define window and isPaused
GLFWwindow* window;
bool isPaused = false;

float total_simulated_time = 0.0f;

// Add this global variable at the top of main.cu
bool isRenderingPaused = false;

// Add this global variable
bool printTimings = true;

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
        case PHOSPHOGLUCOSE_ISOMERASE: return "PHOSPHOGLUCOSE_ISOMERASE";
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
        case HEXOKINASE_GLUCOSE_COMPLEX: return "HEXOKINASE_GLUCOSE_COMPLEX";
        case HEXOKINASE_GLUCOSE_ATP_COMPLEX: return "HEXOKINASE_GLUCOSE_ATP_COMPLEX";
        case GLUCOSE_6_PHOSPHATE_ISOMERASE_COMPLEX: return "GLUCOSE_6_PHOSPHATE_ISOMERASE_COMPLEX";
        case FRUCTOSE_6_PHOSPHATE_ISOMERASE_COMPLEX: return "FRUCTOSE_6_PHOSPHATE_ISOMERASE_COMPLEX";
        case PHOSPHOFRUCTOKINASE_1_COMPLEX: return "PHOSPHOFRUCTOKINASE_1_COMPLEX";
        case PHOSPHOFRUCTOKINASE_1_ATP_COMPLEX: return "PHOSPHOFRUCTOKINASE_1_ATP_COMPLEX";
        case FRUCTOSE_1_6_BISPHOSPHATE_ALDOLASE_COMPLEX: return "FRUCTOSE_1_6_BISPHOSPHATE_ALDOLASE_COMPLEX";
        case GLYCERALDEHYDE_3_PHOSPHATE_ALDOLASE_COMPLEX: return "GLYCERALDEHYDE_3_PHOSPHATE_ALDOLASE_COMPLEX";
        case GLYCERALDEHYDE_3_PHOSPHATE_ALDOLASE_DHAP_COMPLEX: return "GLYCERALDEHYDE_3_PHOSPHATE_ALDOLASE_DHAP_COMPLEX";
        case DHAP_TRIOSEPHOSPHATE_ISOMERASE_COMPLEX: return "DHAP_TRIOSEPHOSPHATE_ISOMERASE_COMPLEX";
        case GLYCERALDEHYDE_3_PHOSPHATE_TRIOSEPHOSPHATE_ISOMERASE_COMPLEX: return "GLYCERALDEHYDE_3_PHOSPHATE_TRIOSEPHOSPHATE_ISOMERASE_COMPLEX";
        case GLYCERALDEHYDE_3_PHOSPHATE_DEHYDROGENASE_COMPLEX: return "GLYCERALDEHYDE_3_PHOSPHATE_DEHYDROGENASE_COMPLEX";
        case GLYCERALDEHYDE_3_PHOSPHATE_DEHYDROGENASE_NAD_PLUS_COMPLEX: return "GLYCERALDEHYDE_3_PHOSPHATE_DEHYDROGENASE_NAD_PLUS_COMPLEX";
        case GLYCERALDEHYDE_3_PHOSPHATE_DEHYDROGENASE_NAD_PLUS_INORGANIC_PHOSPHATE_COMPLEX: return "GLYCERALDEHYDE_3_PHOSPHATE_DEHYDROGENASE_NAD_PLUS_INORGANIC_PHOSPHATE_COMPLEX";
        case PHOSPHOGLYCERATE_KINASE_COMPLEX: return "PHOSPHOGLYCERATE_KINASE_COMPLEX";
        case PHOSPHOGLYCERATE_KINASE_ADP_COMPLEX: return "PHOSPHOGLYCERATE_KINASE_ADP_COMPLEX";
        case PHOSPHOGLYCERATE_MUTASE_COMPLEX: return "PHOSPHOGLYCERATE_MUTASE_COMPLEX";
        case ENOLASE_COMPLEX: return "ENOLASE_COMPLEX";
        case PYRUVATE_KINASE_COMPLEX: return "PYRUVATE_KINASE_COMPLEX";
        case PYRUVATE_KINASE_ADP_COMPLEX: return "PYRUVATE_KINASE_ADP_COMPLEX";
        case NONE: return "NONE";
        default: return "UNKNOWN";
    }
}

// Host function to create a molecule
Molecule createMolecule(MoleculeType type) {
    switch (type) {
        // Substrates and products
        case GLUCOSE: return Molecule::createGlucose();
        case ATP: return Molecule::createATP();
        case ADP: return Molecule::createADP();
        case GLUCOSE_6_PHOSPHATE: return Molecule::createGlucose6Phosphate();
        case FRUCTOSE_6_PHOSPHATE: return Molecule::createFructose6Phosphate();
        case FRUCTOSE_1_6_BISPHOSPHATE: return Molecule::createFructose16Bisphosphate();
        case DIHYDROXYACETONE_PHOSPHATE: return Molecule::createDihydroxyacetonePhosphate();
        case GLYCERALDEHYDE_3_PHOSPHATE: return Molecule::createGlyceraldehyde3Phosphate();
        case _1_3_BISPHOSPHOGLYCERATE: return Molecule::create13Bisphosphoglycerate();
        case _3_PHOSPHOGLYCERATE: return Molecule::create3Phosphoglycerate();
        case _2_PHOSPHOGLYCERATE: return Molecule::create2Phosphoglycerate();
        case PHOSPHOENOLPYRUVATE: return Molecule::createPhosphoenolpyruvate();
        case PYRUVATE: return Molecule::createPyruvate();
        case NAD_PLUS: return Molecule::createNADPlus();
        case NADH: return Molecule::createNADH();
        case PROTON: return Molecule::createProton();
        case INORGANIC_PHOSPHATE: return Molecule::createInorganicPhosphate();
        case WATER: return Molecule::createWater();
        case AMP: return Molecule::createAMP();
        case CITRATE: return Molecule::createCitrate();
        case FRUCTOSE_2_6_BISPHOSPHATE: return Molecule::createFructose26Bisphosphate();

        // Enzymes
        case HEXOKINASE: return Molecule::createHexokinase();
        case PHOSPHOGLUCOSE_ISOMERASE: return Molecule::createGlucose6PhosphateIsomerase();
        case PHOSPHOFRUCTOKINASE_1: return Molecule::createPhosphofructokinase1();
        case ALDOLASE: return Molecule::createAldolase();
        case TRIOSEPHOSPHATE_ISOMERASE: return Molecule::createTriosephosphateIsomerase();
        case GLYCERALDEHYDE_3_PHOSPHATE_DEHYDROGENASE: return Molecule::createGlyceraldehyde3PhosphateDehydrogenase();
        case PHOSPHOGLYCERATE_KINASE: return Molecule::createPhosphoglycerateKinase();
        case PHOSPHOGLYCERATE_MUTASE: return Molecule::createPhosphoglycerateMutase();
        case ENOLASE: return Molecule::createEnolase();
        case PYRUVATE_KINASE: return Molecule::createPyruvateKinase();

        // Enzyme complexes
        case HEXOKINASE_GLUCOSE_COMPLEX: return Molecule::createHexokinaseGlucoseComplex();
        case HEXOKINASE_GLUCOSE_ATP_COMPLEX: return Molecule::createHexokinaseGlucoseATPComplex();
        case GLUCOSE_6_PHOSPHATE_ISOMERASE_COMPLEX: return Molecule::createGlucose6PhosphateIsomeraseComplex();
        case FRUCTOSE_6_PHOSPHATE_ISOMERASE_COMPLEX: return Molecule::createFructose6PhosphateIsomeraseComplex();
        case PHOSPHOFRUCTOKINASE_1_COMPLEX: return Molecule::createPhosphofructokinase1Complex();
        case PHOSPHOFRUCTOKINASE_1_ATP_COMPLEX: return Molecule::createPhosphofructokinase1ATPComplex();
        case FRUCTOSE_1_6_BISPHOSPHATE_ALDOLASE_COMPLEX: return Molecule::createFructose16BisphosphateAldolaseComplex();
        case GLYCERALDEHYDE_3_PHOSPHATE_ALDOLASE_COMPLEX: return Molecule::createGlyceraldehyde3PhosphateAldolaseComplex();
        case GLYCERALDEHYDE_3_PHOSPHATE_ALDOLASE_DHAP_COMPLEX: return Molecule::createGlyceraldehyde3PhosphateAldolaseDHAPComplex();
        case DHAP_TRIOSEPHOSPHATE_ISOMERASE_COMPLEX: return Molecule::createDHAPTriosephosphateIsomeraseComplex();
        case GLYCERALDEHYDE_3_PHOSPHATE_TRIOSEPHOSPHATE_ISOMERASE_COMPLEX: return Molecule::createGlyceraldehyde3PhosphateTriosephosphateIsomeraseComplex();
        case GLYCERALDEHYDE_3_PHOSPHATE_DEHYDROGENASE_COMPLEX: return Molecule::createGlyceraldehyde3PhosphateDehydrogenaseComplex();
        case GLYCERALDEHYDE_3_PHOSPHATE_DEHYDROGENASE_NAD_PLUS_COMPLEX: return Molecule::createGlyceraldehyde3PhosphateDehydrogenaseNADPlusComplex();
        case GLYCERALDEHYDE_3_PHOSPHATE_DEHYDROGENASE_NAD_PLUS_INORGANIC_PHOSPHATE_COMPLEX: return Molecule::createGlyceraldehyde3PhosphateDehydrogenaseNADPlusInorganicPhosphateComplex();
        case PHOSPHOGLYCERATE_KINASE_COMPLEX: return Molecule::createPhosphoglycerateKinaseComplex();
        case PHOSPHOGLYCERATE_KINASE_ADP_COMPLEX: return Molecule::createPhosphoglycerateKinaseADPComplex();
        case PHOSPHOGLYCERATE_MUTASE_COMPLEX: return Molecule::createPhosphoglycerateMutaseComplex();
        case ENOLASE_COMPLEX: return Molecule::createEnolaseComplex();
        case PYRUVATE_KINASE_COMPLEX: return Molecule::createPyruvateKinaseComplex();
        case PYRUVATE_KINASE_ADP_COMPLEX: return Molecule::createPyruvateKinaseADPComplex();

        case NONE: return Molecule::createNone();

        default:
            fprintf(stderr, "Unknown molecule type: %d\n", type);
            return Molecule::createWater();  // Default case, could also throw an exception
    }
}

// Host function to process creation and deletion flags
void processCreationDeletionFlags(Molecule* molecules, int* num_molecules, int max_molecules,
                                  const MoleculeCreationInfo* creationBuffer, int numCreations,
                                  const int* deletionBuffer, int numDeletions) {
    // Process deletions
    if (deletionBuffer && numDeletions > 0) {
        std::vector<int> deletionIndices(deletionBuffer, deletionBuffer + numDeletions);
        std::sort(deletionIndices.begin(), deletionIndices.end(), std::greater<int>());
        for (int i = 0; i < numDeletions; i++) {
            int indexToDelete = deletionIndices[i];
            assert(indexToDelete >= 0 && indexToDelete < *num_molecules); // Ensure valid index
            if (indexToDelete != *num_molecules - 1) {
                molecules[indexToDelete] = molecules[*num_molecules - 1];
            }
            (*num_molecules)--;
            //printf("Successfully deleted molecule at index %d\n", indexToDelete);
        }
    }

    // Process creations
    if (creationBuffer && numCreations > 0) {
        for (int i = 0; i < numCreations; i++) {
            assert(*num_molecules < max_molecules); // Ensure we don't exceed max molecules
            Molecule newMolecule = createMolecule(creationBuffer[i].type);
            //printf("Created molecule of type %s\n", getMoleculeTypeName(newMolecule.type));
            newMolecule.centerOfMass = make_float3(creationBuffer[i].x, creationBuffer[i].y, creationBuffer[i].z);
            molecules[*num_molecules] = newMolecule;
            (*num_molecules)++;
            //printf("Successfully added molecule to molecules array\n");
        }
    }
}

// Modify the runSimulation function to use runSimulationStep
cudaError_t runSimulation(SimulationSpace* space, Molecule* molecules, int num_ticks) {
    assert(space != nullptr); // Ensure space is not null
    assert(molecules != nullptr); // Ensure molecules array is not null
    assert(num_ticks > 0); // Ensure there is at least one tick

    cudaError_t cudaStatus;

    // Choose which GPU to run on, change this on a multi-GPU system.
    cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaSetDevice failed! Do you have a CUDA-capable GPU installed?\n");
        return cudaStatus;
    }
    //printf("CUDA device set successfully\n");

    // Main simulation loop
    for (int tick = 0; tick < num_ticks; tick++) {
        //printf("Starting tick %d\n", tick);

        cudaStatus = runSimulationStep(space, molecules);
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "Simulation step failed! Error: %s\n", cudaGetErrorString(cudaStatus));
            break;
        }

        //printf("Completed tick %d\n", tick);
    }

    //printf("Simulation completed\n");

    return cudaStatus;
}

cudaError_t runSimulationStep(SimulationSpace* space, Molecule* molecules) {
    static Molecule* dev_molecules = nullptr;
    static curandState* dev_states = nullptr;
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

    cudaError_t cudaStatus;
    int threadsPerBlock = g_threadsToUse;
    int blocksPerGrid = g_blocksToUse;

    // Calculate total cells
    int totalCells = grid.sizeX * grid.sizeY * grid.sizeZ;

    dim3 gridAssign(blocksPerGrid, 1, 1);

    // CUDA event creation for GPU timing
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    //printf("Starting simulation step\n");

    // CPU timing variables
    std::chrono::high_resolution_clock::time_point t1, t2;

    // Add these new timing variables
    float resetCellsTime = 0.0f;
    float initCurandTime = 0.0f;
    float resetCreationDeletionBuffersTime = 0.0f;

    // Timing: Memory allocation and initialization
    t1 = std::chrono::high_resolution_clock::now();

    if (dev_cells == nullptr) {
        cudaStatus = cudaMalloc(&dev_cells, totalCells * sizeof(Cell));
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "cudaMalloc failed for dev_cells!\n");
            return cudaStatus;
        }
    }

    // Timing: Reset cells
    cudaEventRecord(start);
    cudaMemset(dev_cells, 0, totalCells * sizeof(Cell));
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&resetCellsTime, start, stop);

    // Initialize device memory if it's the first run
    if (dev_molecules == nullptr) {

        printf("Max molecules is %d\n", MAX_MOLECULES);
        printf("About to allocate max_molecules * sizeof(Molecule) = %zu\n", MAX_MOLECULES * sizeof(Molecule));

        // Allocate GPU buffers and copy data
        cudaStatus = cudaMalloc((void**)&dev_molecules, MAX_MOLECULES * sizeof(Molecule));
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "cudaMalloc failed for dev_molecules! Error: %s\n", cudaGetErrorString(cudaStatus));
            return cudaStatus;
        }
        printf("Successfully allocated dev_molecules with size %zu\n", MAX_MOLECULES * sizeof(Molecule));
        cudaStatus = cudaMalloc((void**)&dev_states, blocksPerGrid * threadsPerBlock * sizeof(curandState));
        if (cudaStatus != cudaSuccess) { fprintf(stderr, "cudaMalloc failed for dev_states!\n"); return cudaStatus; }
        printf("Successfully allocated dev_states with size %zu\n", blocksPerGrid * threadsPerBlock * sizeof(curandState));
        cudaStatus = cudaMalloc((void**)&dev_num_molecules, sizeof(int));
        if (cudaStatus != cudaSuccess) { fprintf(stderr, "cudaMalloc failed for dev_num_molecules!\n"); return cudaStatus; }
        printf("Successfully allocated dev_num_molecules with size %zu\n", sizeof(int));
        cudaStatus = cudaMalloc((void**)&dev_creationBuffer, MAX_MOLECULES * sizeof(MoleculeCreationInfo));
        if (cudaStatus != cudaSuccess) { fprintf(stderr, "cudaMalloc failed for dev_creationBuffer!\n"); return cudaStatus; }
        printf("Successfully allocated dev_creationBuffer with size %zu\n", MAX_MOLECULES * sizeof(MoleculeCreationInfo));
        cudaStatus = cudaMalloc((void**)&dev_numCreations, sizeof(int));
        if (cudaStatus != cudaSuccess) { fprintf(stderr, "cudaMalloc failed for dev_numCreations!\n"); return cudaStatus; }
        printf("Successfully allocated dev_numCreations with size %zu\n", sizeof(int));
        cudaStatus = cudaMalloc((void**)&dev_deletionBuffer, MAX_MOLECULES * sizeof(int));
        if (cudaStatus != cudaSuccess) { fprintf(stderr, "cudaMalloc failed for dev_deletionBuffer!\n"); return cudaStatus; }
        printf("Successfully allocated dev_deletionBuffer with size %zu\n", MAX_MOLECULES * sizeof(int));
        cudaStatus = cudaMalloc((void**)&dev_numDeletions, sizeof(int));
        if (cudaStatus != cudaSuccess) { fprintf(stderr, "cudaMalloc failed for dev_numDeletions!\n"); return cudaStatus; }
        printf("Successfully allocated dev_numDeletions with size %zu\n", sizeof(int));

        // Timing: Initialize curandState
        cudaEventRecord(start);
        initCurand<<<blocksPerGrid, threadsPerBlock>>>(time(NULL), dev_states, blocksPerGrid * threadsPerBlock);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&initCurandTime, start, stop);
    }

    t2 = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> memoryAllocationTime = t2 - t1;
    //printf("Successfully allocated memory\n");

    // Timing: Memory copy to device
    cudaEventRecord(start);
    cudaStatus = cudaMemcpy(dev_molecules, molecules, MAX_MOLECULES * sizeof(Molecule), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed for molecules to dev_molecules! Error: %s\n", cudaGetErrorString(cudaStatus));
        return cudaStatus;
    }
    //printf("Successfully copied molecules to device\n");

    cudaStatus = cudaMemcpy(dev_num_molecules, &space->num_molecules, sizeof(int), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed for num_molecules to dev_num_molecules!\n");
        return cudaStatus;
    }
    //printf("Successfully copied num_molecules to device\n");

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float memcpyToDeviceTime;
    cudaEventElapsedTime(&memcpyToDeviceTime, start, stop);

    // Timing: Reset the creation and deletion buffers
    cudaEventRecord(start);
    cudaMemset(dev_numCreations, 0, sizeof(int));
    cudaMemset(dev_numDeletions, 0, sizeof(int));
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&resetCreationDeletionBuffersTime, start, stop);

    //printf("Successfully set numCreations and numDeletions to 0\n");

    // Timing: Assign molecules to cells kernel
    cudaEventRecord(start);
    //dim3 gridAssign((space->num_molecules + threadsPerBlock - 1) / threadsPerBlock);
    assignMoleculesToCells<<<blocksPerGrid, threadsPerBlock>>>(dev_molecules, space->num_molecules, dev_cells, *space, grid);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float assignMoleculesTime;
    cudaEventElapsedTime(&assignMoleculesTime, start, stop);
    //printf("Successfully assigned molecules to cells\n");

    // Timing: Apply forces and update positions kernel
    cudaEventRecord(start);
    float dt = 1e-6f; // Timestep of 1 microsecond
    applyForcesAndUpdatePositions<<<blocksPerGrid, threadsPerBlock>>>(
        dev_molecules, space->num_molecules, *space, dt, dev_states);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float applyForcesTime;
    cudaEventElapsedTime(&applyForcesTime, start, stop);
    //printf("Successfully applied forces and updated positions\n");
    // Timing: handle binding reactions kernel
    cudaEventRecord(start);
    handleBindings<<<blocksPerGrid, threadsPerBlock>>>(dev_molecules, dev_num_molecules, MAX_MOLECULES, dev_states,
                                                      dev_creationBuffer, dev_numCreations,
                                                      dev_deletionBuffer, dev_numDeletions,
                                                      dev_cells, grid);

    // sync to flush the buffer
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float handleBindingsTime;
    cudaEventElapsedTime(&handleBindingsTime, start, stop);
    //printf("Successfully handled bindings\n");
    // Timing: Memory copy from device to host
    cudaEventRecord(start);
    cudaStatus = cudaMemcpy(molecules, dev_molecules, space->num_molecules * sizeof(Molecule), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed for dev_molecules to molecules!\n");
        return cudaStatus;
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float memcpyFromDeviceTime;
    cudaEventElapsedTime(&memcpyFromDeviceTime, start, stop);
    //printf("Successfully copied molecules from device to host\n");
    // Timing: Process creation and deletion flags
    t1 = std::chrono::high_resolution_clock::now();
    int h_numCreations, h_numDeletions;
    cudaMemcpy(&h_numCreations, dev_numCreations, sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(&h_numDeletions, dev_numDeletions, sizeof(int), cudaMemcpyDeviceToHost);

    MoleculeCreationInfo* h_creationBuffer = nullptr;
    int* h_deletionBuffer = nullptr;

    if (h_numCreations > 0) {
        h_creationBuffer = new MoleculeCreationInfo[h_numCreations];
        cudaMemcpy(h_creationBuffer, dev_creationBuffer, h_numCreations * sizeof(MoleculeCreationInfo), cudaMemcpyDeviceToHost);
    }

    if (h_numDeletions > 0) {
        h_deletionBuffer = new int[h_numDeletions];
        cudaMemcpy(h_deletionBuffer, dev_deletionBuffer, h_numDeletions * sizeof(int), cudaMemcpyDeviceToHost);
    }

    //printf("Successfully first copied creation and deletion buffers from device to host\n");

    processCreationDeletionFlags(molecules, &space->num_molecules, MAX_MOLECULES,
                                 h_creationBuffer, h_numCreations,
                                 h_deletionBuffer, h_numDeletions);


    t2 = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> processFlagsTime = t2 - t1;
    //printf("Successfully processed creation and deletion flags\n");
    // Reset the creation and deletion buffers
    cudaMemset(dev_numCreations, 0, sizeof(int));
    cudaMemset(dev_numDeletions, 0, sizeof(int));

    //printf("Successfully reset creation and deletion buffer counters\n");

    // Timing: copy molecules back to device
    cudaEventRecord(start);
    cudaStatus = cudaMemcpy(dev_molecules, molecules, space->num_molecules * sizeof(Molecule), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed for molecules to dev_molecules!\n");
        return cudaStatus;
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float secondMemcpyToDeviceTime;
    cudaEventElapsedTime(&secondMemcpyToDeviceTime, start, stop);
    //printf("Successfully copied molecules back to device\n");
    // Timing: handle reactions and dissociations kernel
    t1 = std::chrono::high_resolution_clock::now();
    handleReactionsAndDissociations<<<blocksPerGrid, threadsPerBlock>>>(dev_molecules, dev_num_molecules, MAX_MOLECULES, dev_states,
                                                                     dev_creationBuffer, dev_numCreations,
                                                                     dev_deletionBuffer, dev_numDeletions);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float handleReactionsTime;
    cudaEventElapsedTime(&handleReactionsTime, start, stop);
    //printf("Successfully handled reactions and dissociations\n");
    // Timing: copy molecules back to host
    cudaEventRecord(start);
    cudaStatus = cudaMemcpy(molecules, dev_molecules, space->num_molecules * sizeof(Molecule), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed for dev_molecules to molecules!\n");
        return cudaStatus;
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float secondMemcpyFromDeviceTime;
    cudaEventElapsedTime(&secondMemcpyFromDeviceTime, start, stop);
    //printf("Successfully copied molecules back to host\n");
    // Timing: process creation and deletion flags again
    t1 = std::chrono::high_resolution_clock::now();
    cudaMemcpy(&h_numCreations, dev_numCreations, sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(&h_numDeletions, dev_numDeletions, sizeof(int), cudaMemcpyDeviceToHost);

    //printf("Successfully copied creation and deletion buffers from device to host\n");

    if (h_numCreations > 0) {
        h_creationBuffer = new MoleculeCreationInfo[h_numCreations];
        cudaMemcpy(h_creationBuffer, dev_creationBuffer, h_numCreations * sizeof(MoleculeCreationInfo), cudaMemcpyDeviceToHost);
    }

    if (h_numDeletions > 0) {
        h_deletionBuffer = new int[h_numDeletions];
        cudaMemcpy(h_deletionBuffer, dev_deletionBuffer, h_numDeletions * sizeof(int), cudaMemcpyDeviceToHost);
    }

    //printf("Successfully second copied creation and deletion buffers from device to host\n");

    processCreationDeletionFlags(molecules, &space->num_molecules, MAX_MOLECULES,
                                 h_creationBuffer, h_numCreations,
                                 h_deletionBuffer, h_numDeletions);

    if (h_creationBuffer) delete[] h_creationBuffer;
    if (h_deletionBuffer) delete[] h_deletionBuffer;

    t2 = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> secondProcessFlagsTime = t2 - t1;

    //printf("Successfully second processed creation and deletion flags\n");

    // Print timing results if enabled
    if (printTimings) {
        printf("Simulation Step Timings:\n");
        printf("  Memory Allocation: %.3f ms\n", memoryAllocationTime.count());
        printf("  Reset Cells: %.3f ms\n", resetCellsTime);
        if (initCurandTime > 0) printf("  Initialize cuRand: %.3f ms\n", initCurandTime);
        printf("  Memory Copy to Device: %.3f ms\n", memcpyToDeviceTime);
        printf("  Assign Molecules to Cells: %.3f ms\n", assignMoleculesTime);
        printf("  Apply Forces and Update Positions: %.3f ms\n", applyForcesTime);
        printf("  Handle Bindings: %.3f ms\n", handleBindingsTime);
        printf("  Memory Copy from Device: %.3f ms\n", memcpyFromDeviceTime);
        printf("  Process Creation/Deletion Flags: %.3f ms\n", processFlagsTime.count());
        printf("  Reset Creation/Deletion Buffers: %.3f ms\n", resetCreationDeletionBuffersTime);
        printf("  Memory Copy to Device: %.3f ms\n", secondMemcpyToDeviceTime);
        printf("  Handle Reactions and Dissociations: %.3f ms\n", handleReactionsTime);
        printf("  Memory Copy from Device: %.3f ms\n", secondMemcpyFromDeviceTime);
        printf("  Process Creation/Deletion Flags: %.3f ms\n", secondProcessFlagsTime.count());
        
        // Calculate and print total time
        float totalTime = memoryAllocationTime.count() + resetCellsTime + initCurandTime +
                          memcpyToDeviceTime + assignMoleculesTime + applyForcesTime +
                          handleBindingsTime + memcpyFromDeviceTime + processFlagsTime.count() +
                          resetCreationDeletionBuffersTime + secondMemcpyToDeviceTime +
                          handleReactionsTime + secondMemcpyFromDeviceTime + secondProcessFlagsTime.count();
        printf("  Total Calculated Time: %.3f ms\n", totalTime);
        printf("\n");
    }

    // Clean up CUDA events
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    total_simulated_time += dt;

    return cudaStatus;

}

// Function to calculate MAX_MOLECULES_PER_CELL
int calculateMaxMoleculesPerCell(int max_molecules, float simulation_size, float cell_size) {
    int cells_per_axis = static_cast<int>(simulation_size / cell_size);
    int N_cells = cells_per_axis * cells_per_axis * cells_per_axis;

    float lambda = static_cast<float>(max_molecules) / static_cast<float>(N_cells);
    float sigma = sqrt(lambda);
    float expected_max = lambda + sigma * sqrt(2 * log(N_cells));

    // Choose a safety margin constant, e.g., k = 3
    float safety_margin = 3.0f * sigma;

    return static_cast<int>(expected_max + safety_margin + 0.5f);
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

    g_threadsToUse = deviceProp.maxThreadsPerBlock;
    // use the max number of blocks that can be launched per SM
    g_blocksToUse = deviceProp.maxBlocksPerMultiProcessor;
    //printf("Threads to use: %d\n", g_threadsToUse);
    //printf("Blocks to use: %d\n", g_blocksToUse);


    //printf("CUDA Device Properties:\n");
    //printf("  Device name: %s\n", deviceProp.name);
    //printf("  Compute capability: %d.%d\n", deviceProp.major, deviceProp.minor);
    //printf("  Total global memory: %zu bytes\n", deviceProp.totalGlobalMem);
    //printf("  Max threads per block: %d\n", deviceProp.maxThreadsPerBlock);
    //printf("  Max threads dim: (%d, %d, %d)\n", deviceProp.maxThreadsDim[0], deviceProp.maxThreadsDim[1], deviceProp.maxThreadsDim[2]);
    //printf("  Max grid size: (%d, %d, %d)\n", deviceProp.maxGridSize[0], deviceProp.maxGridSize[1], deviceProp.maxGridSize[2]);
    //printf("  Warp size: %d\n", deviceProp.warpSize);
    //printf("  Memory clock rate: %d kHz\n", deviceProp.memoryClockRate);
    //printf("  Memory bus width: %d bits\n", deviceProp.memoryBusWidth);
    //printf("\n");

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

                //printf("Read from file - Space width: %.2f\n", space.width);
                //printf("Read from file - Space height: %.2f\n", space.height);
                //printf("Read from file - Space depth: %.2f\n", space.depth);
            } else {
                fprintf(stderr, "Failed to read simulation space dimensions\n");
                // Set default values
                space.width = 100.0f;
                space.height = 100.0f;
                space.depth = 100.0f;
            }
            
            //printf("Simulation space dimensions: %.2f x %.2f x %.2f\n", space.width, space.height, space.depth);
            break;
        }
    }

    // Calculate max_molecules based on available memory
    size_t totalGlobalMem = deviceProp.totalGlobalMem;
    size_t moleculeSize = sizeof(Molecule);
    size_t moleculeCreationInfoSize = sizeof(MoleculeCreationInfo);
    size_t intSize = sizeof(int);

    // Reserve some memory for other data structures (e.g., 20%)
    size_t availableMemory = totalGlobalMem * 0.8;

    // Calculate max_molecules
    recommended_max_molecules = static_cast<int>(availableMemory / (moleculeSize + moleculeCreationInfoSize + intSize));

    // Calculate MAX_MOLECULES_PER_CELL
    recommended_max_molecules_per_cell = calculateMaxMoleculesPerCell(recommended_max_molecules, space.width, CELL_SIZE);

    printf("Recommended max molecules: %d\n", recommended_max_molecules);
    printf("Recommended max molecules per cell: %d\n", recommended_max_molecules_per_cell);
    printf("Molecule size: %zu bytes\n", moleculeSize);
    printf("Total global memory: %.2f GB\n", totalGlobalMem / 1e9);

    

    // Read number of molecule types
    while (fgets(line, sizeof(line), input_file)) {
        if (line[0] != '#') {
            if (sscanf(line, "%d", &space.num_molecule_types) != 1) {
                fprintf(stderr, "Failed to read number of molecule types\n");
                fclose(input_file);
                return 1;
            }
            //printf("Number of molecule types: %d\n", space.num_molecule_types);
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
                //printf("Molecule type %d (%s): %d\n", molecule_type, molecule_name, count);
                molecule_type++;
            }
        }
    }

    fclose(input_file);

    //printf("Total number of molecules: %d\n", space.num_molecules);

    // Before allocating memory for molecules
    if (space.num_molecules > MAX_MOLECULES) {
        fprintf(stderr, "Error: Number of molecules (%d) exceeds maximum allowed (%d)\n", space.num_molecules, MAX_MOLECULES);
        return 1;
    }

    // Allocate memory for molecules
    molecules = (Molecule*)malloc(MAX_MOLECULES * sizeof(Molecule));
    assert(molecules != nullptr); // Ensure memory allocation was successful

    //printf("Molecules allocated successfully\n");

    // Initialize molecules
    int molecule_index = 0;
    int type_counts[MAX_MOLECULE_TYPES] = {0};  // Array to keep count of each molecule type

    for (int i = 0; i < space.num_molecule_types; i++) {
        MoleculeType currentType = static_cast<MoleculeType>(i);
        int count = space.molecule_counts[i];

        //printf("Creating %d molecules of type %d (%s)\n", count, i, getMoleculeTypeName(currentType));

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
        //printf("Created molecule type %d (%s): %d\n", i, getMoleculeTypeName(currentType), type_counts[i]);
    }

    // Print summary of all molecule types created
    //printf("\nSummary of molecules created:\n");
    for (int i = 0; i < space.num_molecule_types; i++) {
        if (type_counts[i] > 0) {
            //printf("%s: %d\n", getMoleculeTypeName(static_cast<MoleculeType>(i)), type_counts[i]);
        }
    }

    // Check if we've initialized the correct number of molecules
    if (molecule_index != space.num_molecules) {
        fprintf(stderr, "Error: Initialized %d molecules, expected %d\n", molecule_index, space.num_molecules);
        free(molecules);
        return 1;
    }

    //printf("\nTotal molecules initialized successfully: %d\n", molecule_index);

    // Initialize visualization
    initVisualization();

    std::chrono::high_resolution_clock::time_point simulationStepStart, simulationStepEnd;

    // Main simulation loop
    std::chrono::high_resolution_clock::time_point loopStart, loopEnd, renderStart, renderEnd;
    while (!glfwWindowShouldClose(window)) {
        loopStart = std::chrono::high_resolution_clock::now();
        simulationStepStart = std::chrono::high_resolution_clock::now();

        if (!isPaused) {
            // Run a single step of the simulation
            cudaStatus = runSimulationStep(&space, molecules);
            if (cudaStatus != cudaSuccess) {
                fprintf(stderr, "Simulation step failed! Error: %s\n", cudaGetErrorString(cudaStatus));
                break;
            }
        }

        simulationStepEnd = std::chrono::high_resolution_clock::now();

        renderStart = std::chrono::high_resolution_clock::now();
        if (!isRenderingPaused) {
            // Render the current state of the simulation
            renderSimulation(space, std::vector<Molecule>(molecules, molecules + space.num_molecules), total_simulated_time, 1.0f);
        }
        else {
            if (_kbhit()) {  // Check if a key has been pressed
                char ch = _getch();  // Get the pressed key
                if (ch == 'r' || ch == 'R') {  // Check if the key is 'r' or 'R'
                    printf("Rendering %s\n", isRenderingPaused ? "paused" : "resumed");
                    isRenderingPaused = !isRenderingPaused;  // Toggle rendering pause state
                    //printf("Rendering %s\n", isRenderingPaused ? "paused" : "resumed");
                }
                if (ch == 'w' || ch == 'W') {
                    // Write the current state of the simulation to a file
                    FILE* output_file = fopen("output.txt", "w");
                    if (output_file == NULL) {
                        fprintf(stderr, "Failed to open output.txt\n");
                        break;
                    }
                    // Write the current state of the simulation to the file
                    fprintf(output_file, "Current time: %.2f seconds\n", total_simulated_time);
                    // Write the count of each molecule type
                    for (int i = 0; i < space.num_molecule_types; i++) {
                        fprintf(output_file, "%s: %d\n", getMoleculeTypeName(static_cast<MoleculeType>(i)), space.molecule_counts[i]);
                    }
                    fclose(output_file);
                    printf("Simulation state written to output.txt\n");
                    printf("Press the space bar to continue...\n");

                    // Wait for the space bar to be pressed again before continuing
                    while (_getch() != 32) {  // 32 is the ASCII code for space bar
                        clock_t start_time = clock();
                        while (clock() - start_time < 100000); // Approximately 100ms delay
                    }
                }
                if (ch == 'l' || ch == 'L') {
                    // turn on/off printTimings
                    printTimings = !printTimings;
                    printf("PrintTimings %s\n", printTimings ? "enabled" : "disabled");
                }
            }
        }
        renderEnd = std::chrono::high_resolution_clock::now();

        loopEnd = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> loopTime = loopEnd - loopStart;
        std::chrono::duration<double, std::milli> simulationStepTime = simulationStepEnd - simulationStepStart;
        std::chrono::duration<double, std::milli> renderTime = renderEnd - renderStart;
        if (!isPaused && printTimings) {
            printf("Total Loop Time: %.3f ms\n", loopTime.count());
        }
        if (!isPaused && printTimings) {
            printf("Simulation Step Time: %.3f ms\n", simulationStepTime.count());
        }
        if (!isPaused && printTimings) {
            printf("Render Time: %.3f ms\n", renderTime.count());
        }

        // wait for key press to continue
        //printf("Press any key to continue...\n");
        //_getch();

    }



    // Cleanup
    cleanupVisualization();
    free(molecules);

    return 0;
}
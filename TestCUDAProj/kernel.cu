#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <curand.h>
#include <curand_kernel.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "Molecule.h"

// Define constants
#define MAX_MOLECULES 1000000
#define MAX_MOLECULE_TYPES 15

// SimulationSpace structure
struct SimulationSpace {
    int width, height, depth;
    int num_molecules;
    int num_molecule_types;
    int molecule_counts[MAX_MOLECULE_TYPES];
};

// Function prototypes
__global__ void calculateForces(Molecule* molecules, int num_molecules);
__global__ void updatePositions(Molecule* molecules, int num_molecules, SimulationSpace space, float dt);
__global__ void handleInteractions(Molecule* molecules, int num_molecules);
cudaError_t runSimulation(SimulationSpace* space, Molecule* molecules, int num_ticks);

// Constants for force calculations
#define COULOMB_CONSTANT 8.99e9f  // N*m^2/C^2
#define CUTOFF_DISTANCE 2.0f      // nm
#define CUTOFF_DISTANCE_SQ (CUTOFF_DISTANCE * CUTOFF_DISTANCE)

__device__ float3 calculatePairwiseForce(const Molecule& mol1, const Molecule& mol2) {
    float3 force = make_float3(0.0f, 0.0f, 0.0f);
    float3 r;
    r.x = mol2.getX() - mol1.getX();
    r.y = mol2.getY() - mol1.getY();
    r.z = mol2.getZ() - mol1.getZ();

    float distSq = r.x * r.x + r.y * r.y + r.z * r.z;

    if (distSq < CUTOFF_DISTANCE_SQ && distSq > 0.0f) {
        float dist = sqrtf(distSq);
        float invDist = 1.0f / dist;

        // Lennard-Jones force
        float sigma = 0.5f * (mol1.getSigma() + mol2.getSigma());
        float epsilon = sqrtf(mol1.getEpsilon() * mol2.getEpsilon());
        float sigmaOverDist = sigma * invDist;
        float sigmaOverDist6 = sigmaOverDist * sigmaOverDist * sigmaOverDist;
        sigmaOverDist6 = sigmaOverDist6 * sigmaOverDist6;
        float forceMultiplierLJ = 24.0f * epsilon * invDist * sigmaOverDist6 * (1.0f - 2.0f * sigmaOverDist6);

        // Coulomb force
        float chargeProduct = mol1.getCharge() * mol2.getCharge();
        float forceMultiplierCoulomb = COULOMB_CONSTANT * chargeProduct * invDist * invDist;

        float totalForceMultiplier = forceMultiplierLJ + forceMultiplierCoulomb;

        force.x = r.x * totalForceMultiplier * invDist;
        force.y = r.y * totalForceMultiplier * invDist;
        force.z = r.z * totalForceMultiplier * invDist;
    }

    return force;
}

__global__ void calculateForces(Molecule* molecules, int num_molecules, float3* forces) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_molecules) {
        float3 totalForce = make_float3(0.0f, 0.0f, 0.0f);

        for (int j = 0; j < num_molecules; ++j) {
            if (idx != j) {
                float3 pairForce = calculatePairwiseForce(molecules[idx], molecules[j]);
                totalForce.x += pairForce.x;
                totalForce.y += pairForce.y;
                totalForce.z += pairForce.z;
            }
        }

        forces[idx] = totalForce;
    }
}

// Main function
int main() {
    SimulationSpace space;
    Molecule* molecules;
    
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
            sscanf(line, "%d %d %d", &space.width, &space.height, &space.depth);
            break;
        }
    }
    
    // Read number of molecule types
    while (fgets(line, sizeof(line), input_file)) {
        if (line[0] != '#') {
            sscanf(line, "%d", &space.num_molecule_types);
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
                molecule_type++;
            }
        }
    }
    
    fclose(input_file);
    
    // Allocate memory for molecules
    molecules = (Molecule*)malloc(space.num_molecules * sizeof(Molecule));
    
    // Initialize molecules
    int molecule_index = 0;
    for (int i = 0; i < space.num_molecule_types; i++) {
        for (int j = 0; j < space.molecule_counts[i]; j++) {
            switch (static_cast<MoleculeType>(i)) {
                case GLUCOSE:
                    molecules[molecule_index] = Molecule::createGlucose();
                    break;
                case ATP:
                    molecules[molecule_index] = Molecule::createATP();
                    break;
                case ADP:
                    molecules[molecule_index] = Molecule::createADP();
                    break;
                case GLUCOSE_6_PHOSPHATE:
                    molecules[molecule_index] = Molecule::createGlucose6Phosphate();
                    break;
                case FRUCTOSE_6_PHOSPHATE:
                    molecules[molecule_index] = Molecule::createFructose6Phosphate();
                    break;
                case FRUCTOSE_1_6_BISPHOSPHATE:
                    molecules[molecule_index] = Molecule::createFructose16Bisphosphate();
                    break;
                case DIHYDROXYACETONE_PHOSPHATE:
                    molecules[molecule_index] = Molecule::createDihydroxyacetonePhosphate();
                    break;
                case GLYCERALDEHYDE_3_PHOSPHATE:
                    molecules[molecule_index] = Molecule::createGlyceraldehyde3Phosphate();
                    break;
                case _1_3_BISPHOSPHOGLYCERATE:
                    molecules[molecule_index] = Molecule::create13Bisphosphoglycerate();
                    break;
                case _3_PHOSPHOGLYCERATE:
                    molecules[molecule_index] = Molecule::create3Phosphoglycerate();
                    break;
                case _2_PHOSPHOGLYCERATE:
                    molecules[molecule_index] = Molecule::create2Phosphoglycerate();
                    break;
                case PHOSPHOENOLPYRUVATE:
                    molecules[molecule_index] = Molecule::createPhosphoenolpyruvate();
                    break;
                case PYRUVATE:
                    molecules[molecule_index] = Molecule::createPyruvate();
                    break;
                case NAD_PLUS:
                    molecules[molecule_index] = Molecule::createNADPlus();
                    break;
                case NADH:
                    molecules[molecule_index] = Molecule::createNADH();
                    break;
                case PROTON:
                    molecules[molecule_index] = Molecule::createProton();
                    break;
                case INORGANIC_PHOSPHATE:
                    molecules[molecule_index] = Molecule::createInorganicPhosphate();
                    break;
                case WATER:
                    molecules[molecule_index] = Molecule::createWater();
                    break;
                case HEXOKINASE:
                    molecules[molecule_index] = Molecule::createHexokinase();
                    break;
                case GLUCOSE_6_PHOSPHATE_ISOMERASE:
                    molecules[molecule_index] = Molecule::createGlucose6PhosphateIsomerase();
                    break;
                case PHOSPHOFRUCTOKINASE_1:
                    molecules[molecule_index] = Molecule::createPhosphofructokinase1();
                    break;
                case ALDOLASE:
                    molecules[molecule_index] = Molecule::createAldolase();
                    break;
                case TRIOSEPHOSPHATE_ISOMERASE:
                    molecules[molecule_index] = Molecule::createTriosephosphateIsomerase();
                    break;
                case GLYCERALDEHYDE_3_PHOSPHATE_DEHYDROGENASE:
                    molecules[molecule_index] = Molecule::createGlyceraldehyde3PhosphateDehydrogenase();
                    break;
                case PHOSPHOGLYCERATE_KINASE:
                    molecules[molecule_index] = Molecule::createPhosphoglycerateKinase();
                    break;
                case PHOSPHOGLYCERATE_MUTASE:
                    molecules[molecule_index] = Molecule::createPhosphoglycerateMutase();
                    break;
                case ENOLASE:
                    molecules[molecule_index] = Molecule::createEnolase();
                    break;
                case PYRUVATE_KINASE:
                    molecules[molecule_index] = Molecule::createPyruvateKinase();
                    break;
                case AMP:
                    molecules[molecule_index] = Molecule::createAMP();
                    break;
                case CITRATE:
                    molecules[molecule_index] = Molecule::createCitrate();
                    break;
                case FRUCTOSE_2_6_BISPHOSPHATE:
                    molecules[molecule_index] = Molecule::createFructose26Bisphosphate();
                    break;
                default:
                    fprintf(stderr, "Unknown molecule type\n");
                    return 1;
            }
            // Initialize position and velocity randomly
            float x = static_cast<float>(rand()) / RAND_MAX * space.width;
            float y = static_cast<float>(rand()) / RAND_MAX * space.height;
            float z = static_cast<float>(rand()) / RAND_MAX * space.depth;
            float vx = (static_cast<float>(rand()) / RAND_MAX - 0.5f) * 0.1f;
            float vy = (static_cast<float>(rand()) / RAND_MAX - 0.5f) * 0.1f;
            float vz = (static_cast<float>(rand()) / RAND_MAX - 0.5f) * 0.1f;
            molecules[molecule_index].getPosition(x, y, z);
            molecules[molecule_index].getVelocity(vx, vy, vz);
            molecule_index++;
        }
    }

    // Run simulation
    int num_ticks = 1000; // Adjust as needed
    cudaError_t cudaStatus = runSimulation(&space, molecules, num_ticks);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "Simulation failed!");
        return 1;
    }

    // Free memory
    free(molecules);

    return 0;
}

// CUDA kernels
__global__ void calculateForces(Molecule* molecules, int num_molecules) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_molecules) {
        // Implement force calculation here
        // This is a placeholder - you'll need to implement the actual force calculations
        molecules[idx].applyForce(0.01f, 0.01f, 0.01f);
    }
}

__global__ void updatePositions(Molecule* molecules, int num_molecules, SimulationSpace space, float dt) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_molecules) {
        molecules[idx].updatePosition(dt);
        
        // Implement boundary conditions here
        // This is a simple bounce-off-walls condition
        float x, y, z;
        molecules[idx].getPosition(x, y, z);
        float vx, vy, vz;
        molecules[idx].getVelocity(vx, vy, vz);

        if (x < 0 || x >= space.width) vx *= -1;
        if (y < 0 || y >= space.height) vy *= -1;
        if (z < 0 || z >= space.depth) vz *= -1;

        molecules[idx].applyForce(vx - molecules[idx].getVx(), 
                                  vy - molecules[idx].getVy(),
                                  vz - molecules[idx].getVz());
    }
}

// CUDA kernel to handle interactions and reactions
__global__ void handleInteractions(Molecule* molecules, int num_molecules) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_molecules) {
        // Implement molecule interactions and reactions here
        // This is a placeholder - you'll need to implement the actual interaction logic
    }
}

// Function to run the simulation
cudaError_t runSimulation(SimulationSpace* space, Molecule* molecules, int num_ticks) {
    Molecule* dev_molecules = nullptr;
    cudaError_t cudaStatus;

    // Choose which GPU to run on, change this on a multi-GPU system.
    cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
        goto Error;
    }

    // Allocate GPU buffers
    cudaStatus = cudaMalloc((void**)&dev_molecules, space->num_molecules * sizeof(Molecule));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    // Copy molecules to GPU
    cudaStatus = cudaMemcpy(dev_molecules, molecules, space->num_molecules * sizeof(Molecule), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    // Allocate device memory for forces
    float3* dev_forces;
    cudaStatus = cudaMalloc((void**)&dev_forces, space->num_molecules * sizeof(float3));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    // Main simulation loop
    for (int tick = 0; tick < num_ticks; tick++) {
        int threadsPerBlock = 256;
        int blocksPerGrid = (space->num_molecules + threadsPerBlock - 1) / threadsPerBlock;

        calculateForces<<<blocksPerGrid, threadsPerBlock>>>(dev_molecules, space->num_molecules, dev_forces);
        
        // New kernel to apply forces and update positions
        applyForcesAndUpdatePositions<<<blocksPerGrid, threadsPerBlock>>>(dev_molecules, dev_forces, space->num_molecules, *space, 0.01f);

        handleInteractions<<<blocksPerGrid, threadsPerBlock>>>(dev_molecules, space->num_molecules);

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
    }

    // Copy results back to host
    cudaStatus = cudaMemcpy(molecules, dev_molecules, space->num_molecules * sizeof(Molecule), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

Error:
    cudaFree(dev_molecules);
    cudaFree(dev_forces);
    
    return cudaStatus;
}

__global__ void applyForcesAndUpdatePositions(Molecule* molecules, float3* forces, int num_molecules, SimulationSpace space, float dt) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_molecules) {
        // Apply force
        molecules[idx].applyForce(forces[idx].x, forces[idx].y, forces[idx].z);

        // Update position
        molecules[idx].updatePosition(dt);
        
        // Implement boundary conditions
        float x, y, z;
        molecules[idx].getPosition(x, y, z);
        float vx, vy, vz;
        molecules[idx].getVelocity(vx, vy, vz);

        if (x < 0 || x >= space.width) vx *= -1;
        if (y < 0 || y >= space.height) vy *= -1;
        if (z < 0 || z >= space.depth) vz *= -1;

        molecules[idx].setVelocity(vx, vy, vz);
    }
}
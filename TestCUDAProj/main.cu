#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <curand.h>
#include <curand_kernel.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include "SimulationSpace.h"
#include "Molecule.h"
#include "kernel.cu"  // Include the kernel file directly

// Define constants
#define MAX_MOLECULES 1000000
#define MAX_MOLECULE_TYPES 15
#define NUM_REACTION_TYPES 10 // Update this as you add more reaction types

// Constants for force calculations
#define COULOMB_CONSTANT 8.99e9f  // N*m^2/C^2
#define CUTOFF_DISTANCE 2.0f      // nm
#define CUTOFF_DISTANCE_SQ (CUTOFF_DISTANCE * CUTOFF_DISTANCE)
#define EPSILON_0 8.854187817e-12f // Vacuum permittivity
#define K_BOLTZMANN 1.380649e-23f  // Boltzmann constant
#define TEMPERATURE 310.15f        // Temperature in Kelvin (37°C)
#define SOLVENT_DIELECTRIC 78.5f   // Dielectric constant of water at 37°C

// Function prototypes
__device__ float3 calculatePairwiseForce(const Atom& atom1, const Atom& atom2, float invDist, float distSq);
__global__ void calculateForces(Molecule* molecules, int num_molecules, float3* forces);
__global__ void applyForcesAndUpdatePositions(Molecule* molecules, float3* forces, int num_molecules, SimulationSpace space, float dt);
cudaError_t runSimulation(SimulationSpace* space, Molecule* molecules, int num_ticks);

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
    Molecule* dev_molecules = nullptr;
    float3* dev_forces = nullptr;
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

    cudaStatus = cudaMalloc((void**)&dev_forces, space->num_molecules * sizeof(float3));
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

    // Allocate and initialize curandState
    curandState* dev_states;
    cudaMalloc(&dev_states, space->num_molecules * sizeof(curandState));
    int threadsPerBlock = 256;
    int blocksPerGrid = (space->num_molecules + threadsPerBlock - 1) / threadsPerBlock;
    initCurand<<<blocksPerGrid, threadsPerBlock>>>(time(NULL), dev_states, space->num_molecules);

    // Allocate memory for reaction counts
    int* dev_reactionCounts;
    cudaMalloc(&dev_reactionCounts, NUM_REACTION_TYPES * sizeof(int));
    cudaMemset(dev_reactionCounts, 0, NUM_REACTION_TYPES * sizeof(int));

    // Allocate GPU buffers for num_molecules
    int* dev_num_molecules;
    cudaStatus = cudaMalloc((void**)&dev_num_molecules, sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed for dev_num_molecules!");
        goto Error;
    }

    // Copy num_molecules to GPU
    cudaStatus = cudaMemcpy(dev_num_molecules, &space->num_molecules, sizeof(int), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed for dev_num_molecules!");
        goto Error;
    }

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
        fprintf(stderr, "cudaMalloc failed for creation/deletion buffers!");
        goto Error;
    }

    // Initialize numCreations and numDeletions to 0
    cudaMemset(dev_numCreations, 0, sizeof(int));
    cudaMemset(dev_numDeletions, 0, sizeof(int));

    // Allocate host vectors for creation and deletion buffers
    std::vector<MoleculeCreationInfo> h_creationBuffer;
    std::vector<int> h_deletionBuffer;

    // Main simulation loop
    for (int tick = 0; tick < num_ticks; tick++) {
        calculateForces<<<blocksPerGrid, threadsPerBlock>>>(dev_molecules, space->num_molecules, dev_forces);
        applyForcesAndUpdatePositions<<<blocksPerGrid, threadsPerBlock>>>(dev_molecules, dev_forces, space->num_molecules, *space, 0.01f);
        handleInteractions<<<blocksPerGrid, threadsPerBlock>>>(dev_molecules, dev_num_molecules, MAX_MOLECULES, dev_states, dev_reactionCounts, dev_creationBuffer, dev_numCreations, dev_deletionBuffer, dev_numDeletions);

        // Copy creation and deletion buffers from device to host
        int h_numCreations, h_numDeletions;
        cudaMemcpy(&h_numCreations, dev_numCreations, sizeof(int), cudaMemcpyDeviceToHost);
        cudaMemcpy(&h_numDeletions, dev_numDeletions, sizeof(int), cudaMemcpyDeviceToHost);

        h_creationBuffer.resize(h_numCreations);
        h_deletionBuffer.resize(h_numDeletions);
        cudaMemcpy(h_creationBuffer.data(), dev_creationBuffer, h_numCreations * sizeof(MoleculeCreationInfo), cudaMemcpyDeviceToHost);
        cudaMemcpy(h_deletionBuffer.data(), dev_deletionBuffer, h_numDeletions * sizeof(int), cudaMemcpyDeviceToHost);

        // Process creation and deletion flags on the host
        processCreationDeletionFlags(molecules, &space->num_molecules, MAX_MOLECULES, h_creationBuffer.data(), h_numCreations, h_deletionBuffer.data(), h_numDeletions);

        // Update device memory with new molecule data
        cudaMemcpy(dev_molecules, molecules, space->num_molecules * sizeof(Molecule), cudaMemcpyHostToDevice);
        cudaMemcpy(dev_num_molecules, &space->num_molecules, sizeof(int), cudaMemcpyHostToDevice);

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
    cudaFree(dev_states);
    cudaFree(dev_reactionCounts);
    cudaFree(dev_num_molecules);  // Don't forget to free this new allocation
    cudaFree(dev_creationBuffer);
    cudaFree(dev_numCreations);
    cudaFree(dev_deletionBuffer);
    cudaFree(dev_numDeletions);
    
    return cudaStatus;
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

__global__ void calculateForces(Molecule* molecules, int num_molecules, float3* forces) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_molecules) {
        float3 totalForce = make_float3(0.0f, 0.0f, 0.0f);

        for (int j = 0; j < num_molecules; ++j) {
            if (idx != j) {
                // Use the getAtoms() method to access the fixed-length array of atoms
                const Atom* atoms1 = molecules[idx].getAtoms();
                const Atom* atoms2 = molecules[j].getAtoms();
                int atomCount1 = molecules[idx].getAtomCount();
                int atomCount2 = molecules[j].getAtomCount();

                for (int a1 = 0; a1 < atomCount1; ++a1) {
                    for (int a2 = 0; a2 < atomCount2; ++a2) {
                        const Atom& atom1 = atoms1[a1];
                        const Atom& atom2 = atoms2[a2];

                        float3 r;
                        r.x = atom2.getX() - atom1.getX();
                        r.y = atom2.getY() - atom1.getY();
                        r.z = atom2.getZ() - atom1.getZ();

                        float distSq = r.x * r.x + r.y * r.y + r.z * r.z;

                        if (distSq < CUTOFF_DISTANCE_SQ && distSq > 0.0f) {
                            float invDist = rsqrtf(distSq);
                            float3 pairForce = calculatePairwiseForce(atom1, atom2, invDist, distSq);
                            totalForce.x += pairForce.x;
                            totalForce.y += pairForce.y;
                            totalForce.z += pairForce.z;
                        }
                    }
                }
            }
        }

        forces[idx] = totalForce;
    }
}

__global__ void applyForcesAndUpdatePositions(Molecule* molecules, float3* forces, int num_molecules, SimulationSpace space, float dt) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_molecules) {
        Molecule& mol = molecules[idx];
        float totalMass = mol.getTotalMass();
        float3 force = forces[idx];

        // Apply force to molecule's center of mass
        float ax = force.x / totalMass;
        float ay = force.y / totalMass;
        float az = force.z / totalMass;

        // Update velocity
        mol.setVx(mol.getVx() + ax * dt);
        mol.setVy(mol.getVy() + ay * dt);
        mol.setVz(mol.getVz() + az * dt);

        // Update positions of all atoms in the molecule
        Atom* atoms = mol.getAtoms();
        int atomCount = mol.getAtomCount();
        for (int i = 0; i < atomCount; ++i) {
            Atom& atom = atoms[i];
            atom.setPosition(
                fmodf(atom.getX() + mol.getVx() * dt + space.width, space.width),
                fmodf(atom.getY() + mol.getVy() * dt + space.height, space.height),
                fmodf(atom.getZ() + mol.getVz() * dt + space.depth, space.depth)
            );
        }

        // Recalculate Born radii after position update
        mol.calculateBornRadii();
    }
}
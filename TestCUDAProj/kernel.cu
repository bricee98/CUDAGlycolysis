#include <curand_kernel.h>
#include "Molecule.h"
#include "SimulationState.h"
#include "SimulationSpace.h"

// Constants for interaction radius and reaction probabilities
#define INTERACTION_RADIUS 2.0f
#define INTERACTION_RADIUS_SQ (INTERACTION_RADIUS * INTERACTION_RADIUS)
#define BASE_REACTION_PROBABILITY 0.01f
#define ENZYME_CATALYSIS_FACTOR 100.0f
#define NUM_REACTION_TYPES 10 // Update this as you add more reaction types

// Constants for force calculations
#define COULOMB_CONSTANT 8.99e9f  // N*m^2/C^2
#define CUTOFF_DISTANCE 2.0f      // nm
#define CUTOFF_DISTANCE_SQ (CUTOFF_DISTANCE * CUTOFF_DISTANCE)
#define EPSILON_0 8.854187817e-12f // Vacuum permittivity
#define K_BOLTZMANN 1.380649e-23f  // Boltzmann constant
#define TEMPERATURE 310.15f        // Temperature in Kelvin (37°C)
#define SOLVENT_DIELECTRIC 78.5f   // Dielectric constant of water at 37°C

// Helper function to calculate distance squared between two molecules
__device__ float distanceSquared(const Molecule& mol1, const Molecule& mol2) {
    float dx, dy, dz;
    if (mol1.getRepresentation() == ATOMIC && mol2.getRepresentation() == ATOMIC) {
        dx = mol1.getX() - mol2.getX();
        dy = mol1.getY() - mol2.getY();
        dz = mol1.getZ() - mol2.getZ();
    } else {
        dx = mol1.getCenterOfMass().x - mol2.getCenterOfMass().x;
        dy = mol1.getCenterOfMass().y - mol2.getCenterOfMass().y;
        dz = mol1.getCenterOfMass().z - mol2.getCenterOfMass().z;
    }
    return dx*dx + dy*dy + dz*dz;
}

// Helper function to check for enzyme presence
__device__ bool checkEnzymePresence(Molecule* molecules, int num_molecules, const Molecule& substrate, MoleculeType enzymeType) {
    for (int k = 0; k < num_molecules; k++) {
        if (molecules[k].getType() == enzymeType && distanceSquared(substrate, molecules[k]) <= INTERACTION_RADIUS_SQ) {
            return true;
        }
    }
    return false;
}

// Helper function to check for the presence of a specific molecule type
__device__ bool checkMoleculePresence(Molecule* molecules, int num_molecules, MoleculeType moleculeType) {
    for (int k = 0; k < num_molecules; k++) {
        if (molecules[k].getType() == moleculeType) {
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

// Updated helper function to add a new molecule to the simulation
__device__ void addMoleculeToSimulation(Molecule* molecules, int* num_molecules, int max_molecules, const Molecule& newMolecule, float3 position) {
    int newIndex = atomicAdd(num_molecules, 1);
    if (newIndex < max_molecules) {
        molecules[newIndex] = newMolecule;
        if (newMolecule.getRepresentation() == ATOMIC) {
            molecules[newIndex].setPosition(position.x, position.y, position.z);
        } else {
            molecules[newIndex].setCenterOfMass(position);
        }
        
        // Initialize velocity based on Maxwell-Boltzmann distribution
        float kT = K_BOLTZMANN * TEMPERATURE;
        float mass = molecules[newIndex].getTotalMass();
        float stddev = sqrtf(kT / mass);
        
        curandState localState;
        curand_init(clock64(), newIndex, 0, &localState);
        
        float vx = curand_normal(&localState) * stddev;
        float vy = curand_normal(&localState) * stddev;
        float vz = curand_normal(&localState) * stddev;
        
        molecules[newIndex].setVx(vx);
        molecules[newIndex].setVy(vy);
        molecules[newIndex].setVz(vz);
    }
}

// Updated helper function to remove a molecule from the simulation
__device__ void removeMoleculeFromSimulation(Molecule* molecules, int* num_molecules, int index) {
    int lastIndex = atomicSub(num_molecules, 1) - 1;
    if (index != lastIndex) {
        molecules[index] = molecules[lastIndex];
    }
}

// Main interaction kernel (updated to use creation and deletion buffers)
__global__ void handleInteractions(Molecule* molecules, int* num_molecules, int max_molecules, curandState* states, int* reactionCounts, MoleculeCreationInfo* creationBuffer, int* numCreations, int* deletionBuffer, int* numDeletions) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= *num_molecules) return;

    Molecule& mol1 = molecules[idx];
    curandState localState = states[idx];

    for (int j = idx + 1; j < *num_molecules; j++) {
        Molecule& mol2 = molecules[j];

        if (distanceSquared(mol1, mol2) <= INTERACTION_RADIUS_SQ) {
            switch (mol1.getType()) {
                case GLUCOSE:
                    if (mol2.getType() == ATP) {
                        bool enzymePresent = checkEnzymePresence(molecules, *num_molecules, mol1, HEXOKINASE);
                        if (shouldReact(&localState, BASE_REACTION_PROBABILITY, enzymePresent)) {
                            int deleteIndex = atomicAdd(numDeletions, 2);
                            deletionBuffer[deleteIndex] = idx;
                            deletionBuffer[deleteIndex + 1] = j;

                            int createIndex = atomicAdd(numCreations, 2);
                            creationBuffer[createIndex] = {GLUCOSE_6_PHOSPHATE, mol1.getX(), mol1.getY(), mol1.getZ()};
                            creationBuffer[createIndex + 1] = {ADP, mol2.getX(), mol2.getY(), mol2.getZ()};
                            
                            atomicAdd(&reactionCounts[0], 1);
                            return;
                        }
                    }
                    break;

                case GLUCOSE_6_PHOSPHATE:
                    if (checkEnzymePresence(molecules, *num_molecules, mol1, GLUCOSE_6_PHOSPHATE_ISOMERASE)) {
                        if (shouldReact(&localState, BASE_REACTION_PROBABILITY, true)) {
                            int deleteIndex = atomicAdd(numDeletions, 1);
                            deletionBuffer[deleteIndex] = idx;

                            int createIndex = atomicAdd(numCreations, 1);
                            creationBuffer[createIndex] = {FRUCTOSE_6_PHOSPHATE, mol1.getX(), mol1.getY(), mol1.getZ()};

                            atomicAdd(&reactionCounts[1], 1);
                            return;
                        }
                    }
                    break;

                case FRUCTOSE_6_PHOSPHATE:
                    if (mol2.getType() == ATP) {
                        bool enzymePresent = checkEnzymePresence(molecules, *num_molecules, mol1, PHOSPHOFRUCTOKINASE_1);
                        if (shouldReact(&localState, BASE_REACTION_PROBABILITY, enzymePresent)) {
                            int deleteIndex = atomicAdd(numDeletions, 2);
                            deletionBuffer[deleteIndex] = idx;
                            deletionBuffer[deleteIndex + 1] = j;

                            int createIndex = atomicAdd(numCreations, 2);
                            creationBuffer[createIndex] = {FRUCTOSE_1_6_BISPHOSPHATE, mol1.getX(), mol1.getY(), mol1.getZ()};
                            creationBuffer[createIndex + 1] = {ADP, mol2.getX(), mol2.getY(), mol2.getZ()};

                            atomicAdd(&reactionCounts[2], 1);
                            return;
                        }
                    }
                    break;

                case FRUCTOSE_1_6_BISPHOSPHATE:
                    if (checkEnzymePresence(molecules, *num_molecules, mol1, ALDOLASE)) {
                        if (shouldReact(&localState, BASE_REACTION_PROBABILITY, true)) {
                            int deleteIndex = atomicAdd(numDeletions, 1);
                            deletionBuffer[deleteIndex] = idx;

                            int createIndex = atomicAdd(numCreations, 2);
                            creationBuffer[createIndex] = {DIHYDROXYACETONE_PHOSPHATE, mol1.getX(), mol1.getY(), mol1.getZ()};
                            creationBuffer[createIndex + 1] = {GLYCERALDEHYDE_3_PHOSPHATE, mol1.getX(), mol1.getY(), mol1.getZ()};

                            atomicAdd(&reactionCounts[3], 1);
                            return;
                        }
                    }
                    break;

                case DIHYDROXYACETONE_PHOSPHATE:
                    if (checkEnzymePresence(molecules, *num_molecules, mol1, TRIOSEPHOSPHATE_ISOMERASE)) {
                        if (shouldReact(&localState, BASE_REACTION_PROBABILITY, true)) {
                            int deleteIndex = atomicAdd(numDeletions, 1);
                            deletionBuffer[deleteIndex] = idx;

                            int createIndex = atomicAdd(numCreations, 1);
                            creationBuffer[createIndex] = {GLYCERALDEHYDE_3_PHOSPHATE, mol1.getX(), mol1.getY(), mol1.getZ()};

                            atomicAdd(&reactionCounts[4], 1);
                            return;
                        }
                    }
                    break;

                case GLYCERALDEHYDE_3_PHOSPHATE:
                    if (mol2.getType() == NAD_PLUS && checkMoleculePresence(molecules, *num_molecules, INORGANIC_PHOSPHATE)) {
                        bool enzymePresent = checkEnzymePresence(molecules, *num_molecules, mol1, GLYCERALDEHYDE_3_PHOSPHATE_DEHYDROGENASE);
                        if (shouldReact(&localState, BASE_REACTION_PROBABILITY, enzymePresent)) {
                            int deleteIndex = atomicAdd(numDeletions, 2);
                            deletionBuffer[deleteIndex] = idx;
                            deletionBuffer[deleteIndex + 1] = j;

                            int createIndex = atomicAdd(numCreations, 3);
                            creationBuffer[createIndex] = {_1_3_BISPHOSPHOGLYCERATE, mol1.getX(), mol1.getY(), mol1.getZ()};
                            creationBuffer[createIndex + 1] = {NADH, mol2.getX(), mol2.getY(), mol2.getZ()};
                            creationBuffer[createIndex + 2] = {PROTON, mol1.getX(), mol1.getY(), mol1.getZ()};

                            atomicAdd(&reactionCounts[5], 1);
                            return;
                        }
                    }
                    break;

                case _1_3_BISPHOSPHOGLYCERATE:
                    if (mol2.getType() == ADP) {
                        bool enzymePresent = checkEnzymePresence(molecules, *num_molecules, mol1, PHOSPHOGLYCERATE_KINASE);
                        if (shouldReact(&localState, BASE_REACTION_PROBABILITY, enzymePresent)) {
                            int deleteIndex = atomicAdd(numDeletions, 2);
                            deletionBuffer[deleteIndex] = idx;
                            deletionBuffer[deleteIndex + 1] = j;

                            int createIndex = atomicAdd(numCreations, 2);
                            creationBuffer[createIndex] = {_3_PHOSPHOGLYCERATE, mol1.getX(), mol1.getY(), mol1.getZ()};
                            creationBuffer[createIndex + 1] = {ATP, mol2.getX(), mol2.getY(), mol2.getZ()};

                            atomicAdd(&reactionCounts[6], 1);
                            return;
                        }
                    }
                    break;

                case _3_PHOSPHOGLYCERATE:
                    if (checkEnzymePresence(molecules, *num_molecules, mol1, PHOSPHOGLYCERATE_MUTASE)) {
                        if (shouldReact(&localState, BASE_REACTION_PROBABILITY, true)) {
                            int deleteIndex = atomicAdd(numDeletions, 1);
                            deletionBuffer[deleteIndex] = idx;

                            int createIndex = atomicAdd(numCreations, 1);
                            creationBuffer[createIndex] = {_2_PHOSPHOGLYCERATE, mol1.getX(), mol1.getY(), mol1.getZ()};

                            atomicAdd(&reactionCounts[7], 1);
                            return;
                        }
                    }
                    break;

                case _2_PHOSPHOGLYCERATE:
                    if (checkEnzymePresence(molecules, *num_molecules, mol1, ENOLASE)) {
                        if (shouldReact(&localState, BASE_REACTION_PROBABILITY, true)) {
                            int deleteIndex = atomicAdd(numDeletions, 1);
                            deletionBuffer[deleteIndex] = idx;

                            int createIndex = atomicAdd(numCreations, 2);
                            creationBuffer[createIndex] = {PHOSPHOENOLPYRUVATE, mol1.getX(), mol1.getY(), mol1.getZ()};
                            creationBuffer[createIndex + 1] = {WATER, mol1.getX(), mol1.getY(), mol1.getZ()};

                            atomicAdd(&reactionCounts[8], 1);
                            return;
                        }
                    }
                    break;

                case PHOSPHOENOLPYRUVATE:
                    if (mol2.getType() == ADP) {
                        bool enzymePresent = checkEnzymePresence(molecules, *num_molecules, mol1, PYRUVATE_KINASE);
                        if (shouldReact(&localState, BASE_REACTION_PROBABILITY, enzymePresent)) {
                            int deleteIndex = atomicAdd(numDeletions, 2);
                            deletionBuffer[deleteIndex] = idx;
                            deletionBuffer[deleteIndex + 1] = j;

                            int createIndex = atomicAdd(numCreations, 2);
                            creationBuffer[createIndex] = {PYRUVATE, mol1.getX(), mol1.getY(), mol1.getZ()};
                            creationBuffer[createIndex + 1] = {ATP, mol2.getX(), mol2.getY(), mol2.getZ()};

                            atomicAdd(&reactionCounts[9], 1);
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

// Add this kernel to initialize curand states
__global__ void initCurand(unsigned long long seed, curandState *state, int num_molecules) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_molecules) {
        curand_init(seed, idx, 0, &state[idx]);
    }
}

__device__ float3 calculatePairwiseForce(const Molecule& mol1, const Molecule& mol2, float3 r, float distSq) {
    float3 force = make_float3(0.0f, 0.0f, 0.0f);
    float invDist = rsqrtf(distSq);

    if (mol1.getRepresentation() == ATOMIC && mol2.getRepresentation() == ATOMIC) {
        // Use existing atomic-level force calculation
        const Atom* atoms1 = mol1.getAtoms();
        const Atom* atoms2 = mol2.getAtoms();
        int atomCount1 = mol1.getAtomCount();
        int atomCount2 = mol2.getAtomCount();

        for (int i = 0; i < atomCount1; ++i) {
            for (int j = 0; j < atomCount2; ++j) {
                const Atom& atom1 = atoms1[i];
                const Atom& atom2 = atoms2[j];

                float3 atomR;
                atomR.x = atom2.getX() - atom1.getX();
                atomR.y = atom2.getY() - atom1.getY();
                atomR.z = atom2.getZ() - atom1.getZ();
                float atomDistSq = atomR.x * atomR.x + atomR.y * atomR.y + atomR.z * atomR.z;
                float atomInvDist = rsqrtf(atomDistSq);

                // Lennard-Jones potential
                float sigma = 0.5f * (atom1.getVanDerWaalsRadius() + atom2.getVanDerWaalsRadius());
                float epsilon = sqrtf(atom1.getEpsilon() * atom2.getEpsilon()); // Combining rule
                float sigmaOverDist = sigma * atomInvDist;
                float sigmaOverDist6 = sigmaOverDist * sigmaOverDist * sigmaOverDist;
                sigmaOverDist6 = sigmaOverDist6 * sigmaOverDist6;
                float forceMultiplierLJ = 24.0f * epsilon * atomInvDist * sigmaOverDist6 * (1.0f - 2.0f * sigmaOverDist6);

                // Coulomb force with Generalized Born model
                float fGB = sqrtf(atomDistSq + atom1.getBornRadius() * atom2.getBornRadius() * expf(-atomDistSq / (4.0f * atom1.getBornRadius() * atom2.getBornRadius())));
                float dielectric = 1.0f / (1.0f / SOLVENT_DIELECTRIC - 1.0f / 1.0f) * expf(-atomDistSq / (4.0f * atom1.getBornRadius() * atom2.getBornRadius()));
                float forceMultiplierCoulomb = COULOMB_CONSTANT * atom1.getCharge() * atom2.getCharge() * (1.0f / atomDistSq - 1.0f / (fGB * fGB)) / dielectric;

                float totalForceMultiplier = forceMultiplierLJ + forceMultiplierCoulomb;

                force.x += atomR.x * totalForceMultiplier * atomInvDist;
                force.y += atomR.y * totalForceMultiplier * atomInvDist;
                force.z += atomR.z * totalForceMultiplier * atomInvDist;
            }
        }
    } else {
        // Simplified force calculation for coarse-grained molecules
        float combinedRadius = mol1.getRadius() + mol2.getRadius();
        float epsilon = 10.0f;  // Adjust this value based on your needs
        float sigma = combinedRadius / powf(2.0f, 1.0f/6.0f);
        
        float sigmaOverDist = sigma * invDist;
        float sigmaOverDist6 = sigmaOverDist * sigmaOverDist * sigmaOverDist;
        sigmaOverDist6 = sigmaOverDist6 * sigmaOverDist6;
        
        float forceMultiplier = 24.0f * epsilon * invDist * sigmaOverDist6 * (1.0f - 2.0f * sigmaOverDist6);
        
        force.x = r.x * forceMultiplier * invDist;
        force.y = r.y * forceMultiplier * invDist;
        force.z = r.z * forceMultiplier * invDist;
    }

    return force;
}

__global__ void calculateForces(Molecule* molecules, int num_molecules, float3* forces) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_molecules) {
        float3 totalForce = make_float3(0.0f, 0.0f, 0.0f);

        for (int j = 0; j < num_molecules; ++j) {
            if (idx != j) {
                float3 r;
                if (molecules[idx].getRepresentation() == ATOMIC && molecules[j].getRepresentation() == ATOMIC) {
                    r.x = molecules[j].getX() - molecules[idx].getX();
                    r.y = molecules[j].getY() - molecules[idx].getY();
                    r.z = molecules[j].getZ() - molecules[idx].getZ();
                } else {
                    r.x = molecules[j].getCenterOfMass().x - molecules[idx].getCenterOfMass().x;
                    r.y = molecules[j].getCenterOfMass().y - molecules[idx].getCenterOfMass().y;
                    r.z = molecules[j].getCenterOfMass().z - molecules[idx].getCenterOfMass().z;
                }

                float distSq = r.x * r.x + r.y * r.y + r.z * r.z;

                if (distSq < CUTOFF_DISTANCE_SQ && distSq > 0.0f) {
                    float3 pairForce = calculatePairwiseForce(molecules[idx], molecules[j], r, distSq);
                    totalForce.x += pairForce.x;
                    totalForce.y += pairForce.y;
                    totalForce.z += pairForce.z;
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

        if (mol.getRepresentation() == ATOMIC) {
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
            mol.calculateBornRadii();
        } else {
            // Update center of mass for coarse-grained molecules
            float3 com = mol.getCenterOfMass();
            com.x = fmodf(com.x + mol.getVx() * dt + space.width, space.width);
            com.y = fmodf(com.y + mol.getVy() * dt + space.height, space.height);
            com.z = fmodf(com.z + mol.getVz() * dt + space.depth, space.depth);
            mol.setCenterOfMass(com);
        }
    }
}
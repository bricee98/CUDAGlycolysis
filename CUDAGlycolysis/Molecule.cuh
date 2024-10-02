#pragma once

#include <cuda_runtime.h>
#include "Atom.cuh"

#define MAX_ATOMS_PER_MOLECULE 100

// Enumeration for molecule types (unchanged)
enum MoleculeType {
    // Substrates and products
    GLUCOSE,
    ATP,
    ADP,
    GLUCOSE_6_PHOSPHATE,
    FRUCTOSE_6_PHOSPHATE,
    FRUCTOSE_1_6_BISPHOSPHATE,
    DIHYDROXYACETONE_PHOSPHATE,
    GLYCERALDEHYDE_3_PHOSPHATE,
    _1_3_BISPHOSPHOGLYCERATE,
    _3_PHOSPHOGLYCERATE,
    _2_PHOSPHOGLYCERATE,
    PHOSPHOENOLPYRUVATE,
    PYRUVATE,
    NAD_PLUS,
    NADH,
    PROTON,
    INORGANIC_PHOSPHATE,
    WATER,

    // Enzymes
    HEXOKINASE,
    GLUCOSE_6_PHOSPHATE_ISOMERASE,
    PHOSPHOFRUCTOKINASE_1,
    ALDOLASE,
    TRIOSEPHOSPHATE_ISOMERASE,
    GLYCERALDEHYDE_3_PHOSPHATE_DEHYDROGENASE,
    PHOSPHOGLYCERATE_KINASE,
    PHOSPHOGLYCERATE_MUTASE,
    ENOLASE,
    PYRUVATE_KINASE,

    // Regulatory molecules
    AMP,
    CITRATE,
    FRUCTOSE_2_6_BISPHOSPHATE
};

// Enumeration for molecule representation
enum MoleculeRepresentation {
    ATOMIC = 0,
    COARSE_GRAINED = 1
};

class Molecule {
public:
    // Member variables
    float3 centerOfMass;
    float vx, vy, vz;
    MoleculeType type;
    int atomCount;
    bool markedForDeletion;
    MoleculeType creationFlag;
    MoleculeRepresentation representation;
     // For position
    float radius;         // For coarse-grained molecules
    float mass;           // For coarse-grained molecules
    Atom atoms[MAX_ATOMS_PER_MOLECULE];
    float totalCharge;

    // Constructors
    __host__ __device__ Molecule();

    // Static creation functions for all molecule types
    static __host__ Molecule createGlucose();
    static __host__ Molecule createATP();
    static __host__ Molecule createADP();
    static __host__ Molecule createGlucose6Phosphate();
    static __host__ Molecule createFructose6Phosphate();
    static __host__ Molecule createFructose16Bisphosphate();
    static __host__ Molecule createDihydroxyacetonePhosphate();
    static __host__ Molecule createGlyceraldehyde3Phosphate();
    static __host__ Molecule create13Bisphosphoglycerate();
    static __host__ Molecule create3Phosphoglycerate();
    static __host__ Molecule create2Phosphoglycerate();
    static __host__ Molecule createPhosphoenolpyruvate();
    static __host__ Molecule createPyruvate();
    static __host__ Molecule createNADPlus();
    static __host__ Molecule createNADH();
    static __host__ Molecule createProton();
    static __host__ Molecule createInorganicPhosphate();
    static __host__ Molecule createWater();
    static __host__ Molecule createHexokinase();
    static __host__ Molecule createGlucose6PhosphateIsomerase();
    static __host__ Molecule createPhosphofructokinase1();
    static __host__ Molecule createAldolase();
    static __host__ Molecule createTriosephosphateIsomerase();
    static __host__ Molecule createGlyceraldehyde3PhosphateDehydrogenase();
    static __host__ Molecule createPhosphoglycerateKinase();
    static __host__ Molecule createPhosphoglycerateMutase();
    static __host__ Molecule createEnolase();
    static __host__ Molecule createPyruvateKinase();
    static __host__ Molecule createAMP();
    static __host__ Molecule createCitrate();
    static __host__ Molecule createFructose26Bisphosphate();

    // Utility functions
    __host__ __device__ float getTotalMass() const;
    __host__ __device__ void initializeAtoms();
    
    // Method to get the molecule's type
    __host__ MoleculeType getType() const {
        return type;
    }

    // Method to get the molecule's position
    __host__ void getPosition(float& x, float& y, float& z) const {
        x = centerOfMass.x;
        y = centerOfMass.y;
        z = centerOfMass.z;
    }

private:
    // Helper functions
    void initializeAtomPositions(); // Host-only function
    void calculateBornRadii();      // Host-only function
};

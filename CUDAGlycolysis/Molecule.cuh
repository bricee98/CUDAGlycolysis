#pragma once

#include <cuda_runtime.h>

#define MAX_ATOMS_PER_MOLECULE 100

// Enumeration for molecule types (unchanged)
enum MoleculeType {
    NONE = -1,  // Add this line at the beginning of the enum

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
    PHOSPHOGLUCOSE_ISOMERASE,
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
    FRUCTOSE_2_6_BISPHOSPHATE,

    // New enzyme complex types
    HEXOKINASE_GLUCOSE_COMPLEX,
    HEXOKINASE_GLUCOSE_ATP_COMPLEX,
    GLUCOSE_6_PHOSPHATE_ISOMERASE_COMPLEX,
    FRUCTOSE_6_PHOSPHATE_ISOMERASE_COMPLEX,
    PHOSPHOFRUCTOKINASE_1_COMPLEX,
    PHOSPHOFRUCTOKINASE_1_ATP_COMPLEX,
    FRUCTOSE_1_6_BISPHOSPHATE_ALDOLASE_COMPLEX,
    GLYCERALDEHYDE_3_PHOSPHATE_ALDOLASE_COMPLEX,
    GLYCERALDEHYDE_3_PHOSPHATE_ALDOLASE_DHAP_COMPLEX,
    DHAP_TRIOSEPHOSPHATE_ISOMERASE_COMPLEX,
    GLYCERALDEHYDE_3_PHOSPHATE_TRIOSEPHOSPHATE_ISOMERASE_COMPLEX,
    GLYCERALDEHYDE_3_PHOSPHATE_DEHYDROGENASE_COMPLEX,
    GLYCERALDEHYDE_3_PHOSPHATE_DEHYDROGENASE_NAD_PLUS_COMPLEX,
    GLYCERALDEHYDE_3_PHOSPHATE_DEHYDROGENASE_NAD_PLUS_INORGANIC_PHOSPHATE_COMPLEX,
    PHOSPHOGLYCERATE_KINASE_COMPLEX,
    PHOSPHOGLYCERATE_KINASE_ADP_COMPLEX,
    PHOSPHOGLYCERATE_MUTASE_COMPLEX,
    ENOLASE_COMPLEX,
    PYRUVATE_KINASE_COMPLEX,
    PYRUVATE_KINASE_ADP_COMPLEX
};

class Molecule {
public:
    // Member variables
    float3 centerOfMass;
    MoleculeType type;
    bool markedForDeletion;
    MoleculeType creationFlag;
    float radius;         // For all molecules
    float mass;           // For all molecules
    float totalCharge;    // For all molecules

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

    // New static creation functions for enzyme complexes
    static __host__ Molecule createHexokinaseGlucoseComplex();
    static __host__ Molecule createHexokinaseGlucoseATPComplex();
    static __host__ Molecule createGlucose6PhosphateIsomeraseComplex();
    static __host__ Molecule createFructose6PhosphateIsomeraseComplex();
    static __host__ Molecule createPhosphofructokinase1Complex();
    static __host__ Molecule createPhosphofructokinase1ATPComplex();
    static __host__ Molecule createFructose16BisphosphateAldolaseComplex();
    static __host__ Molecule createGlyceraldehyde3PhosphateAldolaseComplex();
    static __host__ Molecule createGlyceraldehyde3PhosphateAldolaseDHAPComplex();
    static __host__ Molecule createDHAPTriosephosphateIsomeraseComplex();
    static __host__ Molecule createGlyceraldehyde3PhosphateTriosephosphateIsomeraseComplex();
    static __host__ Molecule createGlyceraldehyde3PhosphateDehydrogenaseComplex();
    static __host__ Molecule createGlyceraldehyde3PhosphateDehydrogenaseNADPlusComplex();
    static __host__ Molecule createGlyceraldehyde3PhosphateDehydrogenaseNADPlusInorganicPhosphateComplex();
    static __host__ Molecule createPhosphoglycerateKinaseComplex();
    static __host__ Molecule createPhosphoglycerateKinaseADPComplex();
    static __host__ Molecule createPhosphoglycerateMutaseComplex();
    static __host__ Molecule createEnolaseComplex();
    static __host__ Molecule createPyruvateKinaseComplex();
    static __host__ Molecule createPyruvateKinaseADPComplex();

    // Utility functions
    __host__ __device__ float getTotalMass() const { return mass; }
    
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

    // Add this line with the other static creation functions
    static __host__ Molecule createNone();

private:
    // Helper functions
    // void initializeAtomPositions(); // Host-only function
    // void calculateBornRadii();      // Host-only function
};

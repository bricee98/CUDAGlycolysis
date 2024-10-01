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

// Add a new enum to distinguish between atomic and coarse-grained molecules
enum MoleculeRepresentation {
    ATOMIC = 0,
    COARSE_GRAINED = 1
};

class Molecule {
public:
    __host__ __device__ Molecule();
    __host__ __device__ ~Molecule();

    __host__ __device__ void updatePosition(float dt);
    __host__ __device__ void applyForce(float fx, float fy, float fz);

    __host__ __device__ float getTotalMass() const;
    __host__ __device__ void getPosition(float& outX, float& outY, float& outZ) const;
    __host__ __device__ void getVelocity(float& outVx, float& outVy, float& outVz) const;
    __host__ __device__ void setVelocity(float newVx, float newVy, float newVz);

    __host__ __device__ float getVx() const { return vx; }
    __host__ __device__ float getVy() const { return vy; }
    __host__ __device__ float getVz() const { return vz; }
    __host__ __device__ void setVx(float newVx) { vx = newVx; }
    __host__ __device__ void setVy(float newVy) { vy = newVy; }
    __host__ __device__ void setVz(float newVz) { vz = newVz; }

    __host__ __device__ void calculateBornRadii();

    // Static creation functions for all molecule types
    static __host__ __device__ Molecule createGlucose();
    static __host__ __device__ Molecule createATP();
    static __host__ __device__ Molecule createADP();
    static __host__ __device__ Molecule createGlucose6Phosphate();
    static __host__ __device__ Molecule createFructose6Phosphate();
    static __host__ __device__ Molecule createFructose16Bisphosphate();
    static __host__ __device__ Molecule createDihydroxyacetonePhosphate();
    static __host__ __device__ Molecule createGlyceraldehyde3Phosphate();
    static __host__ __device__ Molecule create13Bisphosphoglycerate();
    static __host__ __device__ Molecule create3Phosphoglycerate();
    static __host__ __device__ Molecule create2Phosphoglycerate();
    static __host__ __device__ Molecule createPhosphoenolpyruvate();
    static __host__ __device__ Molecule createPyruvate();
    static __host__ __device__ Molecule createNADPlus();
    static __host__ __device__ Molecule createNADH();
    static __host__ __device__ Molecule createProton();
    static __host__ __device__ Molecule createInorganicPhosphate();
    static __host__ __device__ Molecule createWater();
    static __host__ __device__ Molecule createHexokinase();
    static __host__ __device__ Molecule createGlucose6PhosphateIsomerase();
    static __host__ __device__ Molecule createPhosphofructokinase1();
    static __host__ __device__ Molecule createAldolase();
    static __host__ __device__ Molecule createTriosephosphateIsomerase();
    static __host__ __device__ Molecule createGlyceraldehyde3PhosphateDehydrogenase();
    static __host__ __device__ Molecule createPhosphoglycerateKinase();
    static __host__ __device__ Molecule createPhosphoglycerateMutase();
    static __host__ __device__ Molecule createEnolase();
    static __host__ __device__ Molecule createPyruvateKinase();
    static __host__ __device__ Molecule createAMP();
    static __host__ __device__ Molecule createCitrate();
    static __host__ __device__ Molecule createFructose26Bisphosphate();

    __host__ __device__ const Atom* getAtoms() const { return atoms; }
    __host__ __device__ Atom* getAtoms() { return atoms; }
    __host__ __device__ int getAtomCount() const { return atomCount; }

    __host__ __device__ float getX() const;
    __host__ __device__ float getY() const;
    __host__ __device__ float getZ() const;
    __host__ __device__ MoleculeType getType() const { return type; }

    __host__ __device__ void setPosition(float x, float y, float z);

    __host__ __device__ bool isMarkedForDeletion() const { return markedForDeletion; }
    __host__ __device__ void markForDeletion() { markedForDeletion = true; }
    
    __host__ __device__ MoleculeType getCreationFlag() const { return creationFlag; }
    __host__ __device__ void setCreationFlag(MoleculeType type) { creationFlag = type; }

    __host__ __device__ MoleculeRepresentation getRepresentation() const { return representation; }
    __host__ __device__ void setRepresentation(MoleculeRepresentation rep) { representation = rep; }

    __host__ __device__ void updateCenterOfMass();

    // For coarse-grained molecules
    __host__ __device__ float3 getCenterOfMass() const;
    __host__ __device__ float getRadius() const { return radius; }
    __host__ __device__ float getMass() const { return mass; }
    __host__ __device__ void setCenterOfMass(float3 com) { centerOfMass = com; }
    __host__ __device__ void setRadius(float r) { radius = r; }
    __host__ __device__ void setMass(float m) { mass = m; }

    Atom atoms[MAX_ATOMS_PER_MOLECULE];

protected:
    MoleculeType type;
    int atomCount;
    float vx, vy, vz;

    __host__ __device__ float calculateDistance(const Atom& atom1, const Atom& atom2) const;
    __host__ __device__ float calculateOverlap(const Atom& atom1, const Atom& atom2, float distance) const;
    __host__ __device__ float calculateSecondShellCorrection(const Atom& atom1, const Atom& atom2, float distance) const;
    __host__ __device__ void initializeAtomPositions();

private:
    bool markedForDeletion;
    MoleculeType creationFlag;
    MoleculeRepresentation representation;
    float3 centerOfMass;  // For coarse-grained molecules
    float radius;  // For coarse-grained molecules
    float mass;  // For coarse-grained molecules
};
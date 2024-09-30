#pragma once

#include <cuda_runtime.h>
#include <vector>
#include <string>

// Enumeration for atom types
enum AtomType {
    HYDROGEN,
    CARBON,
    NITROGEN,
    OXYGEN,
    PHOSPHORUS,
    SULFUR
};

// Structure to represent an individual atom
struct Atom {
    AtomType type;
    float x, y, z;
    float charge;
    float radius;
    float mass;
    float bornRadius;       // Add this line
    float inverseBornRadius; // Add this line
};

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

class Molecule {
public:
    Molecule();
    virtual ~Molecule();

    void updatePosition(float dt);
    void applyForce(float fx, float fy, float fz);

    MoleculeType getType() const { return type; }
    float getTotalMass() const;
    void getPosition(float& x, float& y, float& z) const;
    void getVelocity(float& vx, float& vy, float& vz) const;

    // Static creation functions for all molecule types
    static Molecule createGlucose();
    static Molecule createATP();
    static Molecule createADP();
    static Molecule createGlucose6Phosphate();
    static Molecule createFructose6Phosphate();
    static Molecule createFructose16Bisphosphate();
    static Molecule createDihydroxyacetonePhosphate();
    static Molecule createGlyceraldehyde3Phosphate();
    static Molecule create13Bisphosphoglycerate();
    static Molecule create3Phosphoglycerate();
    static Molecule create2Phosphoglycerate();
    static Molecule createPhosphoenolpyruvate();
    static Molecule createPyruvate();
    static Molecule createNADPlus();
    static Molecule createNADH();
    static Molecule createProton();
    static Molecule createInorganicPhosphate();
    static Molecule createWater();
    static Molecule createHexokinase();
    static Molecule createGlucose6PhosphateIsomerase();
    static Molecule createPhosphofructokinase1();
    static Molecule createAldolase();
    static Molecule createTriosephosphateIsomerase();
    static Molecule createGlyceraldehyde3PhosphateDehydrogenase();
    static Molecule createPhosphoglycerateKinase();
    static Molecule createPhosphoglycerateMutase();
    static Molecule createEnolase();
    static Molecule createPyruvateKinase();
    static Molecule createAMP();
    static Molecule createCitrate();
    static Molecule createFructose26Bisphosphate();

    const std::vector<Atom>& getAtoms() const { return atoms; }
    float getBornRadius() const { return bornRadius; }

protected:
    MoleculeType type;
    std::vector<Atom> atoms;
    float vx, vy, vz;
    float bornRadius; // Effective Born radius for GB model

    void calculateBornRadius();
    float calculateDistance(const Atom& atom1, const Atom& atom2) const;
    float calculateOverlap(const Atom& atom1, const Atom& atom2, float distance) const;
    float calculateSecondShellCorrection(const Atom& atom1, const Atom& atom2, float distance) const;
    float initialEstimate(AtomType type) const;
    void initializeAtomPositions();

public:
    float getVx() const { return vx; }
    float getVy() const { return vy; }
    float getVz() const { return vz; }

    float getX() const; // Center of mass X
    float getY() const; // Center of mass Y
    float getZ() const; // Center of mass Z
};
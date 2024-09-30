#include "Molecule.h"
#include <cmath>

Molecule::Molecule()
    : type(WATER), x(0), y(0), z(0), vx(0), vy(0), vz(0), mass(18.0f) {}

Molecule::~Molecule() {}

void Molecule::updatePosition(float dt) {
    x += vx * dt;
    y += vy * dt;
    z += vz * dt;
}

void Molecule::applyForce(float fx, float fy, float fz) {
    vx += fx / mass;
    vy += fy / mass;
    vz += fz / mass;
}

void Molecule::getPosition(float& outX, float& outY, float& outZ) const {
    outX = x;
    outY = y;
    outZ = z;
}

void Molecule::getVelocity(float& outVx, float& outVy, float& outVz) const {
    outVx = vx;
    outVy = vy;
    outVz = vz;
}

// Static creation functions for all molecule types
Molecule Molecule::createGlucose() {
    Molecule m;
    m.type = GLUCOSE;
    m.mass = 180.16f;
    m.charge = 0.0f;  // Glucose is neutral
    m.sigma = 0.5f;   // Approximate size in nanometers
    m.epsilon = 0.1f; // Arbitrary value, adjust based on desired interaction strength
    return m;
}

Molecule Molecule::createATP() { Molecule m; m.type = ATP; m.mass = 507.18f; return m; }
Molecule Molecule::createADP() { Molecule m; m.type = ADP; m.mass = 427.2f; return m; }
Molecule Molecule::createGlucose6Phosphate() { Molecule m; m.type = GLUCOSE_6_PHOSPHATE; m.mass = 260.14f; return m; }
Molecule Molecule::createFructose6Phosphate() { Molecule m; m.type = FRUCTOSE_6_PHOSPHATE; m.mass = 260.14f; return m; }
Molecule Molecule::createFructose16Bisphosphate() { Molecule m; m.type = FRUCTOSE_1_6_BISPHOSPHATE; m.mass = 340.12f; return m; }
Molecule Molecule::createDihydroxyacetonePhosphate() { Molecule m; m.type = DIHYDROXYACETONE_PHOSPHATE; m.mass = 170.06f; return m; }
Molecule Molecule::createGlyceraldehyde3Phosphate() { Molecule m; m.type = GLYCERALDEHYDE_3_PHOSPHATE; m.mass = 170.06f; return m; }
Molecule Molecule::create13Bisphosphoglycerate() { Molecule m; m.type = _1_3_BISPHOSPHOGLYCERATE; m.mass = 266.05f; return m; }
Molecule Molecule::create3Phosphoglycerate() { Molecule m; m.type = _3_PHOSPHOGLYCERATE; m.mass = 186.06f; return m; }
Molecule Molecule::create2Phosphoglycerate() { Molecule m; m.type = _2_PHOSPHOGLYCERATE; m.mass = 186.06f; return m; }
Molecule Molecule::createPhosphoenolpyruvate() { Molecule m; m.type = PHOSPHOENOLPYRUVATE; m.mass = 168.04f; return m; }
Molecule Molecule::createPyruvate() { Molecule m; m.type = PYRUVATE; m.mass = 88.06f; return m; }
Molecule Molecule::createNADPlus() { Molecule m; m.type = NAD_PLUS; m.mass = 663.43f; return m; }
Molecule Molecule::createNADH() { Molecule m; m.type = NADH; m.mass = 664.44f; return m; }
Molecule Molecule::createProton() { Molecule m; m.type = PROTON; m.mass = 1.00794f; return m; }
Molecule Molecule::createInorganicPhosphate() { Molecule m; m.type = INORGANIC_PHOSPHATE; m.mass = 94.97f; return m; }
Molecule Molecule::createWater() { Molecule m; m.type = WATER; m.mass = 18.02f; return m; }
Molecule Molecule::createHexokinase() { Molecule m; m.type = HEXOKINASE; m.mass = 100000.0f; return m; }
Molecule Molecule::createGlucose6PhosphateIsomerase() { Molecule m; m.type = GLUCOSE_6_PHOSPHATE_ISOMERASE; m.mass = 60000.0f; return m; }
Molecule Molecule::createPhosphofructokinase1() { Molecule m; m.type = PHOSPHOFRUCTOKINASE_1; m.mass = 85000.0f; return m; }
Molecule Molecule::createAldolase() { Molecule m; m.type = ALDOLASE; m.mass = 160000.0f; return m; }
Molecule Molecule::createTriosephosphateIsomerase() { Molecule m; m.type = TRIOSEPHOSPHATE_ISOMERASE; m.mass = 27000.0f; return m; }
Molecule Molecule::createGlyceraldehyde3PhosphateDehydrogenase() { Molecule m; m.type = GLYCERALDEHYDE_3_PHOSPHATE_DEHYDROGENASE; m.mass = 150000.0f; return m; }
Molecule Molecule::createPhosphoglycerateKinase() { Molecule m; m.type = PHOSPHOGLYCERATE_KINASE; m.mass = 45000.0f; return m; }
Molecule Molecule::createPhosphoglycerateMutase() { Molecule m; m.type = PHOSPHOGLYCERATE_MUTASE; m.mass = 30000.0f; return m; }
Molecule Molecule::createEnolase() { Molecule m; m.type = ENOLASE; m.mass = 82000.0f; return m; }
Molecule Molecule::createPyruvateKinase() { Molecule m; m.type = PYRUVATE_KINASE; m.mass = 240000.0f; return m; }
Molecule Molecule::createAMP() { Molecule m; m.type = AMP; m.mass = 347.2f; return m; }
Molecule Molecule::createCitrate() { Molecule m; m.type = CITRATE; m.mass = 192.12f; return m; }
Molecule Molecule::createFructose26Bisphosphate() { Molecule m; m.type = FRUCTOSE_2_6_BISPHOSPHATE; m.mass = 340.12f; return m; }
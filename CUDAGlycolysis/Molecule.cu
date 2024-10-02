#include "Molecule.cuh"
#include <cmath>
#include <cstdlib>

__host__ Molecule::Molecule()
    : type(WATER), atomCount(0), vx(0.0f), vy(0.0f), vz(0.0f),
      markedForDeletion(false), creationFlag(WATER), representation(ATOMIC),
      centerOfMass(make_float3(0.0f, 0.0f, 0.0f)), radius(0.0f), mass(0.0f) {}

__host__ float Molecule::getTotalMass() const {
    if (representation == COARSE_GRAINED) {
        return mass;
    } else {
        float totalMass = 0.0f;
        for (int i = 0; i < atomCount; ++i) {
            totalMass += atoms[i].mass;
        }
        return totalMass;
    }
}

void Molecule::initializeAtomPositions() {
    // Simple grid placement for demonstration
    float spacing = 0.1f;
    int gridSize = std::ceil(std::pow((float)atomCount, 1.0f / 3.0f));
    int idx = 0;
    for (int x = 0; x < gridSize && idx < atomCount; ++x) {
        for (int y = 0; y < gridSize && idx < atomCount; ++y) {
            for (int z = 0; z < gridSize && idx < atomCount; ++z) {
                atoms[idx].x = x * spacing;
                atoms[idx].y = y * spacing;
                atoms[idx].z = z * spacing;
                idx++;
            }
        }
    }
}

void Molecule::calculateBornRadii() {
    // Simplified calculation
    for (int i = 0; i < atomCount; ++i) {
        atoms[i].bornRadius = atoms[i].vanDerWaalsRadius;
    }
}

__host__ void Molecule::initializeAtoms() {
    initializeAtomPositions();
    calculateBornRadii();

    // Update center of mass, total mass, and total charge
    if (representation == ATOMIC) {
        float totalMass = 0.0f;
        float totalCharge = 0.0f;  // Add this line
        centerOfMass = make_float3(0.0f, 0.0f, 0.0f);
        for (int i = 0; i < atomCount; ++i) {
            float mass = atoms[i].mass;
            float charge = atoms[i].charge;  // Add this line
            centerOfMass.x += atoms[i].x * mass;
            centerOfMass.y += atoms[i].y * mass;
            centerOfMass.z += atoms[i].z * mass;
            totalMass += mass;
            totalCharge += charge;  // Add this line
        }
        centerOfMass.x /= totalMass;
        centerOfMass.y /= totalMass;
        centerOfMass.z /= totalMass;
        mass = totalMass;
        this->totalCharge = totalCharge;  // Add this line
    } else {
        // For coarse-grained molecules, assign an effective totalCharge
        this->totalCharge = 0.0f;  // Adjust as needed
    }
}

// Helper function to set random initial velocity
__host__ void setRandomVelocity(Molecule& m) {
    float maxInitialVelocity = 0.3f;
    m.vx = ((float)rand() / RAND_MAX) * 2.0f * maxInitialVelocity - maxInitialVelocity;
    m.vy = ((float)rand() / RAND_MAX) * 2.0f * maxInitialVelocity - maxInitialVelocity;
    m.vz = ((float)rand() / RAND_MAX) * 2.0f * maxInitialVelocity - maxInitialVelocity;
}

// Implement the static creation functions for all molecule types
__host__ Molecule Molecule::createGlucose() {
    Molecule m;
    m.type = GLUCOSE;
    m.representation = ATOMIC;
    m.atomCount = 24; // 6 C, 12 H, 6 O

    int idx = 0;
    for (int i = 0; i < 6; ++i) {
        m.atoms[idx++] = Atom(CARBON, 0.0f, 0.0f, 0.0f, 0.0f, 12.01f);
    }
    for (int i = 0; i < 12; ++i) {
        m.atoms[idx++] = Atom(HYDROGEN, 0.0f, 0.0f, 0.0f, 0.0f, 1.008f);
    }
    for (int i = 0; i < 6; ++i) {
        m.atoms[idx++] = Atom(OXYGEN, 0.0f, 0.0f, 0.0f, 0.0f, 16.00f);
    }

    m.initializeAtoms();
    setRandomVelocity(m);
    return m;
}

__host__ Molecule Molecule::createATP() {
    Molecule m;
    m.type = ATP;
    m.representation = ATOMIC;
    m.atomCount = 45; // 10 C, 16 H, 13 N, 3 O, 3 P

    int idx = 0;
    for (int i = 0; i < 10; ++i) {
        m.atoms[idx++] = Atom(CARBON, 0.0f, 0.0f, 0.0f, 0.0f, 12.01f);
    }
    for (int i = 0; i < 16; ++i) {
        m.atoms[idx++] = Atom(HYDROGEN, 0.0f, 0.0f, 0.0f, 0.0f, 1.008f);
    }
    for (int i = 0; i < 13; ++i) {
        m.atoms[idx++] = Atom(NITROGEN, 0.0f, 0.0f, 0.0f, 0.0f, 14.01f);
    }
    for (int i = 0; i < 3; ++i) {
        m.atoms[idx++] = Atom(OXYGEN, 0.0f, 0.0f, 0.0f, 0.0f, 16.00f);
    }
    for (int i = 0; i < 3; ++i) {
        m.atoms[idx++] = Atom(PHOSPHORUS, 0.0f, 0.0f, 0.0f, 0.0f, 30.97f);
    }

    m.initializeAtoms();
    setRandomVelocity(m);
    return m;
}

__host__ Molecule Molecule::createADP() {
    Molecule m;
    m.type = ADP;
    m.representation = ATOMIC;
    m.atomCount = 39; // 10 C, 15 H, 13 N, 2 O, 2 P

    int idx = 0;
    for (int i = 0; i < 10; ++i) {
        m.atoms[idx++] = Atom(CARBON, 0.0f, 0.0f, 0.0f, 0.0f, 12.01f);
    }
    for (int i = 0; i < 15; ++i) {
        m.atoms[idx++] = Atom(HYDROGEN, 0.0f, 0.0f, 0.0f, 0.0f, 1.008f);
    }
    for (int i = 0; i < 13; ++i) {
        m.atoms[idx++] = Atom(NITROGEN, 0.0f, 0.0f, 0.0f, 0.0f, 14.01f);
    }
    for (int i = 0; i < 2; ++i) {
        m.atoms[idx++] = Atom(OXYGEN, 0.0f, 0.0f, 0.0f, 0.0f, 16.00f);
    }
    for (int i = 0; i < 2; ++i) {
        m.atoms[idx++] = Atom(PHOSPHORUS, 0.0f, 0.0f, 0.0f, 0.0f, 30.97f);
    }

    m.initializeAtoms();
    setRandomVelocity(m);
    return m;
}

__host__ Molecule Molecule::createGlucose6Phosphate() {
    Molecule m;
    m.type = GLUCOSE_6_PHOSPHATE;
    m.representation = ATOMIC;
    m.atomCount = 28; // 6 C, 13 H, 9 O, 1 P

    int idx = 0;
    for (int i = 0; i < 6; ++i) {
        m.atoms[idx++] = Atom(CARBON, 0.0f, 0.0f, 0.0f, 0.0f, 12.01f);
    }
    for (int i = 0; i < 13; ++i) {
        m.atoms[idx++] = Atom(HYDROGEN, 0.0f, 0.0f, 0.0f, 0.0f, 1.008f);
    }
    for (int i = 0; i < 9; ++i) {
        m.atoms[idx++] = Atom(OXYGEN, 0.0f, 0.0f, 0.0f, 0.0f, 16.00f);
    }
    m.atoms[idx++] = Atom(PHOSPHORUS, 0.0f, 0.0f, 0.0f, 0.0f, 30.97f);

    m.initializeAtoms();
    setRandomVelocity(m);
    return m;
}

__host__ Molecule Molecule::createFructose6Phosphate() {
    Molecule m;
    m.type = FRUCTOSE_6_PHOSPHATE;
    m.representation = ATOMIC;
    m.atomCount = 28; // 6 C, 13 H, 9 O, 1 P

    int idx = 0;
    for (int i = 0; i < 6; ++i) {
        m.atoms[idx++] = Atom(CARBON, 0.0f, 0.0f, 0.0f, 0.0f, 12.01f);
    }
    for (int i = 0; i < 13; ++i) {
        m.atoms[idx++] = Atom(HYDROGEN, 0.0f, 0.0f, 0.0f, 0.0f, 1.008f);
    }
    for (int i = 0; i < 9; ++i) {
        m.atoms[idx++] = Atom(OXYGEN, 0.0f, 0.0f, 0.0f, 0.0f, 16.00f);
    }
    m.atoms[idx++] = Atom(PHOSPHORUS, 0.0f, 0.0f, 0.0f, 0.0f, 30.97f);

    m.initializeAtoms();
    setRandomVelocity(m);
    return m;
}

__host__ Molecule Molecule::createFructose16Bisphosphate() {
    Molecule m;
    m.type = FRUCTOSE_1_6_BISPHOSPHATE;
    m.representation = ATOMIC;
    m.atomCount = 32; // 6 C, 14 H, 12 O, 2 P

    int idx = 0;
    for (int i = 0; i < 6; ++i) {
        m.atoms[idx++] = Atom(CARBON, 0.0f, 0.0f, 0.0f, 0.0f, 12.01f);
    }
    for (int i = 0; i < 14; ++i) {
        m.atoms[idx++] = Atom(HYDROGEN, 0.0f, 0.0f, 0.0f, 0.0f, 1.008f);
    }
    for (int i = 0; i < 12; ++i) {
        m.atoms[idx++] = Atom(OXYGEN, 0.0f, 0.0f, 0.0f, 0.0f, 16.00f);
    }
    for (int i = 0; i < 2; ++i) {
        m.atoms[idx++] = Atom(PHOSPHORUS, 0.0f, 0.0f, 0.0f, 0.0f, 30.97f);
    }

    m.initializeAtoms();
    setRandomVelocity(m);
    return m;
}

__host__ Molecule Molecule::createDihydroxyacetonePhosphate() {
    Molecule m;
    m.type = DIHYDROXYACETONE_PHOSPHATE;
    m.representation = ATOMIC;
    m.atomCount = 16; // 3 C, 7 H, 6 O, 1 P

    int idx = 0;
    for (int i = 0; i < 3; ++i) {
        m.atoms[idx++] = Atom(CARBON, 0.0f, 0.0f, 0.0f, 0.0f, 12.01f);
    }
    for (int i = 0; i < 7; ++i) {
        m.atoms[idx++] = Atom(HYDROGEN, 0.0f, 0.0f, 0.0f, 0.0f, 1.008f);
    }
    for (int i = 0; i < 6; ++i) {
        m.atoms[idx++] = Atom(OXYGEN, 0.0f, 0.0f, 0.0f, 0.0f, 16.00f);
    }
    m.atoms[idx++] = Atom(PHOSPHORUS, 0.0f, 0.0f, 0.0f, 0.0f, 30.97f);

    m.initializeAtoms();
    setRandomVelocity(m);
    return m;
}

__host__ Molecule Molecule::createGlyceraldehyde3Phosphate() {
    Molecule m;
    m.type = GLYCERALDEHYDE_3_PHOSPHATE;
    m.representation = ATOMIC;
    m.atomCount = 16; // 3 C, 7 H, 6 O, 1 P

    int idx = 0;
    for (int i = 0; i < 3; ++i) {
        m.atoms[idx++] = Atom(CARBON, 0.0f, 0.0f, 0.0f, 0.0f, 12.01f);
    }
    for (int i = 0; i < 7; ++i) {
        m.atoms[idx++] = Atom(HYDROGEN, 0.0f, 0.0f, 0.0f, 0.0f, 1.008f);
    }
    for (int i = 0; i < 6; ++i) {
        m.atoms[idx++] = Atom(OXYGEN, 0.0f, 0.0f, 0.0f, 0.0f, 16.00f);
    }
    m.atoms[idx++] = Atom(PHOSPHORUS, 0.0f, 0.0f, 0.0f, 0.0f, 30.97f);

    m.initializeAtoms();
    setRandomVelocity(m);
    return m;
}

__host__ Molecule Molecule::create13Bisphosphoglycerate() {
    Molecule m;
    m.type = _1_3_BISPHOSPHOGLYCERATE;
    m.representation = ATOMIC;
    m.atomCount = 20; // 3 C, 7 H, 10 O, 2 P

    int idx = 0;
    for (int i = 0; i < 3; ++i) {
        m.atoms[idx++] = Atom(CARBON, 0.0f, 0.0f, 0.0f, 0.0f, 12.01f);
    }
    for (int i = 0; i < 7; ++i) {
        m.atoms[idx++] = Atom(HYDROGEN, 0.0f, 0.0f, 0.0f, 0.0f, 1.008f);
    }
    for (int i = 0; i < 10; ++i) {
        m.atoms[idx++] = Atom(OXYGEN, 0.0f, 0.0f, 0.0f, 0.0f, 16.00f);
    }
    for (int i = 0; i < 2; ++i) {
        m.atoms[idx++] = Atom(PHOSPHORUS, 0.0f, 0.0f, 0.0f, 0.0f, 30.97f);
    }

    m.initializeAtoms();
    setRandomVelocity(m);
    return m;
}

__host__ Molecule Molecule::create3Phosphoglycerate() {
    Molecule m;
    m.type = _3_PHOSPHOGLYCERATE;
    m.representation = ATOMIC;
    m.atomCount = 16; // 3 C, 7 H, 7 O, 1 P

    int idx = 0;
    for (int i = 0; i < 3; ++i) {
        m.atoms[idx++] = Atom(CARBON, 0.0f, 0.0f, 0.0f, 0.0f, 12.01f);
    }
    for (int i = 0; i < 7; ++i) {
        m.atoms[idx++] = Atom(HYDROGEN, 0.0f, 0.0f, 0.0f, 0.0f, 1.008f);
    }
    for (int i = 0; i < 7; ++i) {
        m.atoms[idx++] = Atom(OXYGEN, 0.0f, 0.0f, 0.0f, 0.0f, 16.00f);
    }
    m.atoms[idx++] = Atom(PHOSPHORUS, 0.0f, 0.0f, 0.0f, 0.0f, 30.97f);

    m.initializeAtoms();
    setRandomVelocity(m);
    return m;
}

__host__ Molecule Molecule::create2Phosphoglycerate() {
    Molecule m;
    m.type = _2_PHOSPHOGLYCERATE;
    m.representation = ATOMIC;
    m.atomCount = 16; // 3 C, 7 H, 7 O, 1 P

    int idx = 0;
    for (int i = 0; i < 3; ++i) {
        m.atoms[idx++] = Atom(CARBON, 0.0f, 0.0f, 0.0f, 0.0f, 12.01f);
    }
    for (int i = 0; i < 7; ++i) {
        m.atoms[idx++] = Atom(HYDROGEN, 0.0f, 0.0f, 0.0f, 0.0f, 1.008f);
    }
    for (int i = 0; i < 7; ++i) {
        m.atoms[idx++] = Atom(OXYGEN, 0.0f, 0.0f, 0.0f, 0.0f, 16.00f);
    }
    m.atoms[idx++] = Atom(PHOSPHORUS, 0.0f, 0.0f, 0.0f, 0.0f, 30.97f);

    m.initializeAtoms();
    setRandomVelocity(m);
    return m;
}

__host__ Molecule Molecule::createPhosphoenolpyruvate() {
    Molecule m;
    m.type = PHOSPHOENOLPYRUVATE;
    m.representation = ATOMIC;
    m.atomCount = 13; // 3 C, 5 H, 6 O, 1 P

    int idx = 0;
    for (int i = 0; i < 3; ++i) {
        m.atoms[idx++] = Atom(CARBON, 0.0f, 0.0f, 0.0f, 0.0f, 12.01f);
    }
    for (int i = 0; i < 5; ++i) {
        m.atoms[idx++] = Atom(HYDROGEN, 0.0f, 0.0f, 0.0f, 0.0f, 1.008f);
    }
    for (int i = 0; i < 6; ++i) {
        m.atoms[idx++] = Atom(OXYGEN, 0.0f, 0.0f, 0.0f, 0.0f, 16.00f);
    }
    m.atoms[idx++] = Atom(PHOSPHORUS, 0.0f, 0.0f, 0.0f, 0.0f, 30.97f);

    m.initializeAtoms();
    setRandomVelocity(m);
    return m;
}

__host__ Molecule Molecule::createPyruvate() {
    Molecule m;
    m.type = PYRUVATE;
    m.representation = ATOMIC;
    m.atomCount = 10; // 3 C, 4 H, 3 O

    int idx = 0;
    for (int i = 0; i < 3; ++i) {
        m.atoms[idx++] = Atom(CARBON, 0.0f, 0.0f, 0.0f, 0.0f, 12.01f);
    }
    for (int i = 0; i < 4; ++i) {
        m.atoms[idx++] = Atom(HYDROGEN, 0.0f, 0.0f, 0.0f, 0.0f, 1.008f);
    }
    for (int i = 0; i < 3; ++i) {
        m.atoms[idx++] = Atom(OXYGEN, 0.0f, 0.0f, 0.0f, 0.0f, 16.00f);
    }

    m.initializeAtoms();
    setRandomVelocity(m);
    return m;
}

__host__ Molecule Molecule::createNADPlus() {
    Molecule m;
    m.type = NAD_PLUS;
    m.representation = ATOMIC;
    m.atomCount = 66; // Approximate atom count for NAD+

    // Simplified atom assignment
    int idx = 0;
    for (int i = 0; i < 21; ++i) m.atoms[idx++] = Atom(CARBON, 0.0f, 0.0f, 0.0f, 0.0f, 12.01f);
    for (int i = 0; i < 27; ++i) m.atoms[idx++] = Atom(HYDROGEN, 0.0f, 0.0f, 0.0f, 0.0f, 1.008f);
    for (int i = 0; i < 7; ++i) m.atoms[idx++] = Atom(NITROGEN, 0.0f, 0.0f, 0.0f, 0.0f, 14.01f);
    for (int i = 0; i < 15; ++i) m.atoms[idx++] = Atom(OXYGEN, 0.0f, 0.0f, 0.0f, 0.0f, 16.00f);
    for (int i = 0; i < 2; ++i) m.atoms[idx++] = Atom(PHOSPHORUS, 0.0f, 0.0f, 0.0f, 0.0f, 30.97f);

    m.initializeAtoms();
    setRandomVelocity(m);
    return m;
}

__host__ Molecule Molecule::createNADH() {
    Molecule m;
    m.type = NADH;
    m.representation = ATOMIC;
    m.atomCount = 68; // Approximate atom count for NADH

    // Simplified atom assignment
    int idx = 0;
    for (int i = 0; i < 21; ++i) m.atoms[idx++] = Atom(CARBON, 0.0f, 0.0f, 0.0f, 0.0f, 12.01f);
    for (int i = 0; i < 29; ++i) m.atoms[idx++] = Atom(HYDROGEN, 0.0f, 0.0f, 0.0f, 0.0f, 1.008f);
    for (int i = 0; i < 7; ++i) m.atoms[idx++] = Atom(NITROGEN, 0.0f, 0.0f, 0.0f, 0.0f, 14.01f);
    for (int i = 0; i < 15; ++i) m.atoms[idx++] = Atom(OXYGEN, 0.0f, 0.0f, 0.0f, 0.0f, 16.00f);
    for (int i = 0; i < 2; ++i) m.atoms[idx++] = Atom(PHOSPHORUS, 0.0f, 0.0f, 0.0f, 0.0f, 30.97f);

    m.initializeAtoms();
    setRandomVelocity(m);
    return m;
}

__host__ Molecule Molecule::createProton() {
    Molecule m;
    m.type = PROTON;
    m.representation = ATOMIC;
    m.atomCount = 1;

    m.atoms[0] = Atom(HYDROGEN, 0.0f, 0.0f, 0.0f, 1.0f, 1.008f);

    m.initializeAtoms();
    setRandomVelocity(m);
    return m;
}

__host__ Molecule Molecule::createInorganicPhosphate() {
    Molecule m;
    m.type = INORGANIC_PHOSPHATE;
    m.representation = ATOMIC;
    m.atomCount = 5; // 1 P, 4 O

    int idx = 0;
    m.atoms[idx++] = Atom(PHOSPHORUS, 0.0f, 0.0f, 0.0f, 0.0f, 30.97f);
    for (int i = 0; i < 4; ++i) {
        m.atoms[idx++] = Atom(OXYGEN, 0.0f, 0.0f, 0.0f, 0.0f, 16.00f);
    }

    m.initializeAtoms();
    setRandomVelocity(m);
    return m;
}

__host__ Molecule Molecule::createWater() {
    Molecule m;
    m.type = WATER;
    m.representation = ATOMIC;
    m.atomCount = 3; // 1 O, 2 H

    m.atoms[0] = Atom(OXYGEN, 0.0f, 0.0f, 0.0f, 0.0f, 16.00f);
    m.atoms[1] = Atom(HYDROGEN, 0.0f, 0.0f, 0.0f, 0.0f, 1.008f);
    m.atoms[2] = Atom(HYDROGEN, 0.0f, 0.0f, 0.0f, 0.0f, 1.008f);

    m.initializeAtoms();
    setRandomVelocity(m);
    return m;
}

// Helper function to create coarse-grained enzymes
__host__ Molecule createCoarseGrainedEnzyme(MoleculeType type, float radius, float mass) {
    Molecule m;
    m.type = type;
    m.representation = COARSE_GRAINED;
    m.radius = radius;
    m.mass = mass;
    m.centerOfMass = make_float3(0.0f, 0.0f, 0.0f);
    setRandomVelocity(m);
    return m;
}

__host__ Molecule Molecule::createHexokinase() {
    return createCoarseGrainedEnzyme(HEXOKINASE, 5.0f, 100000.0f);
}

__host__ Molecule Molecule::createGlucose6PhosphateIsomerase() {
    return createCoarseGrainedEnzyme(GLUCOSE_6_PHOSPHATE_ISOMERASE, 4.8f, 95000.0f);
}

__host__ Molecule Molecule::createPhosphofructokinase1() {
    return createCoarseGrainedEnzyme(PHOSPHOFRUCTOKINASE_1, 5.2f, 105000.0f);
}

__host__ Molecule Molecule::createAldolase() {
    return createCoarseGrainedEnzyme(ALDOLASE, 5.1f, 102000.0f);
}

__host__ Molecule Molecule::createTriosephosphateIsomerase() {
    return createCoarseGrainedEnzyme(TRIOSEPHOSPHATE_ISOMERASE, 4.5f, 90000.0f);
}

__host__ Molecule Molecule::createGlyceraldehyde3PhosphateDehydrogenase() {
    return createCoarseGrainedEnzyme(GLYCERALDEHYDE_3_PHOSPHATE_DEHYDROGENASE, 5.3f, 106000.0f);
}

__host__ Molecule Molecule::createPhosphoglycerateKinase() {
    return createCoarseGrainedEnzyme(PHOSPHOGLYCERATE_KINASE, 4.9f, 98000.0f);
}

__host__ Molecule Molecule::createPhosphoglycerateMutase() {
    return createCoarseGrainedEnzyme(PHOSPHOGLYCERATE_MUTASE, 4.7f, 94000.0f);
}

__host__ Molecule Molecule::createEnolase() {
    return createCoarseGrainedEnzyme(ENOLASE, 5.0f, 100000.0f);
}

__host__ Molecule Molecule::createPyruvateKinase() {
    return createCoarseGrainedEnzyme(PYRUVATE_KINASE, 5.4f, 108000.0f);
}

__host__ Molecule Molecule::createAMP() {
    Molecule m;
    m.type = AMP;
    m.representation = ATOMIC;
    m.atomCount = 33; // Approximate atom count for AMP

    // Simplified atom assignment
    int idx = 0;
    for (int i = 0; i < 10; ++i) m.atoms[idx++] = Atom(CARBON, 0.0f, 0.0f, 0.0f, 0.0f, 12.01f);
    for (int i = 0; i < 14; ++i) m.atoms[idx++] = Atom(HYDROGEN, 0.0f, 0.0f, 0.0f, 0.0f, 1.008f);
    for (int i = 0; i < 5; ++i) m.atoms[idx++] = Atom(NITROGEN, 0.0f, 0.0f, 0.0f, 0.0f, 14.01f);
    for (int i = 0; i < 7; ++i) m.atoms[idx++] = Atom(OXYGEN, 0.0f, 0.0f, 0.0f, 0.0f, 16.00f);
    m.atoms[idx++] = Atom(PHOSPHORUS, 0.0f, 0.0f, 0.0f, 0.0f, 30.97f);

    m.initializeAtoms();
    setRandomVelocity(m);
    return m;
}

__host__ Molecule Molecule::createCitrate() {
    Molecule m;
    m.type = CITRATE;
    m.representation = ATOMIC;
    m.atomCount = 19; // 6 C, 8 H, 7 O

    int idx = 0;
    for (int i = 0; i < 6; ++i) m.atoms[idx++] = Atom(CARBON, 0.0f, 0.0f, 0.0f, 0.0f, 12.01f);
    for (int i = 0; i < 8; ++i) m.atoms[idx++] = Atom(HYDROGEN, 0.0f, 0.0f, 0.0f, 0.0f, 1.008f);
    for (int i = 0; i < 7; ++i) m.atoms[idx++] = Atom(OXYGEN, 0.0f, 0.0f, 0.0f, 0.0f, 16.00f);

    m.initializeAtoms();
    setRandomVelocity(m);
    return m;
}

__host__ Molecule Molecule::createFructose26Bisphosphate() {
    Molecule m;
    m.type = FRUCTOSE_2_6_BISPHOSPHATE;
    m.representation = ATOMIC;
    m.atomCount = 32; // 6 C, 14 H, 12 O, 2 P

    int idx = 0;
    for (int i = 0; i < 6; ++i) m.atoms[idx++] = Atom(CARBON, 0.0f, 0.0f, 0.0f, 0.0f, 12.01f);
    for (int i = 0; i < 14; ++i) m.atoms[idx++] = Atom(HYDROGEN, 0.0f, 0.0f, 0.0f, 0.0f, 1.008f);
    for (int i = 0; i < 12; ++i) m.atoms[idx++] = Atom(OXYGEN, 0.0f, 0.0f, 0.0f, 0.0f, 16.00f);
    for (int i = 0; i < 2; ++i) m.atoms[idx++] = Atom(PHOSPHORUS, 0.0f, 0.0f, 0.0f, 0.0f, 30.97f);

    m.initializeAtoms();
    setRandomVelocity(m);
    return m;
}
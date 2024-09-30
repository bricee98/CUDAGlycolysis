#include "Molecule.h"
#include <cmath>

__host__ __device__ Molecule::Molecule()
    : type(WATER), vx(0), vy(0), vz(0), markedForDeletion(false), creationFlag(WATER), atomCount(0) {}

__host__ __device__ Molecule::~Molecule() {}

__host__ __device__ void Molecule::updatePosition(float dt) {
    for (int i = 0; i < atomCount; ++i) {
        atoms[i].setPosition(
            atoms[i].getX() + vx * dt,
            atoms[i].getY() + vy * dt,
            atoms[i].getZ() + vz * dt
        );
    }
    calculateBornRadii();
}

__host__ __device__ void Molecule::applyForce(float fx, float fy, float fz) {
    float totalMass = getTotalMass();
    vx += fx / totalMass;
    vy += fy / totalMass;
    vz += fz / totalMass;
}

__host__ __device__ float Molecule::getTotalMass() const {
    float totalMass = 0;
    for (int i = 0; i < atomCount; ++i) {
        totalMass += atoms[i].getMass();
    }
    return totalMass;
}

__host__ __device__ void Molecule::getPosition(float& outX, float& outY, float& outZ) const {
    outX = getX();
    outY = getY();
    outZ = getZ();
}

__host__ __device__ void Molecule::getVelocity(float& outVx, float& outVy, float& outVz) const {
    outVx = vx;
    outVy = vy;
    outVz = vz;
}

__host__ __device__ float Molecule::getX() const {
    float totalMass = getTotalMass();
    float centerX = 0;
    for (int i = 0; i < atomCount; ++i) {
        centerX += atoms[i].getX() * atoms[i].getMass();
    }
    return centerX / totalMass;
}

__host__ __device__ float Molecule::getY() const {
    float totalMass = getTotalMass();
    float centerY = 0;
    for (int i = 0; i < atomCount; ++i) {
        centerY += atoms[i].getY() * atoms[i].getMass();
    }
    return centerY / totalMass;
}

__host__ __device__ float Molecule::getZ() const {
    float totalMass = getTotalMass();
    float centerZ = 0;
    for (int i = 0; i < atomCount; ++i) {
        centerZ += atoms[i].getZ() * atoms[i].getMass();
    }
    return centerZ / totalMass;
}

__host__ __device__ void Molecule::calculateBornRadii() {
    const float minAllowedRadius = 0.1f;

    for (int i = 0; i < atomCount; ++i) {
        atoms[i].setBornRadius(atoms[i].getVanDerWaalsRadius());
        
        for (int j = 0; j < atomCount; ++j) {
            if (i != j) {
                float distance = calculateDistance(atoms[i], atoms[j]);
                atoms[i].setBornRadius(atoms[i].getBornRadius() - calculateOverlap(atoms[i], atoms[j], distance));
            }
        }
        
        atoms[i].setInverseBornRadius(1.0f / fmaxf(atoms[i].getBornRadius(), minAllowedRadius));
    }
    
    for (int i = 0; i < atomCount; ++i) {
        float correction = 0.0f;
        for (int j = 0; j < atomCount; ++j) {
            if (i != j) {
                float distance = calculateDistance(atoms[i], atoms[j]);
                correction += calculateSecondShellCorrection(atoms[i], atoms[j], distance);
            }
        }
        atoms[i].setInverseBornRadius(atoms[i].getInverseBornRadius() + correction);
    }
}

__host__ __device__ float Molecule::calculateDistance(const Atom& atom1, const Atom& atom2) const {
    float dx = atom1.getX() - atom2.getX();
    float dy = atom1.getY() - atom2.getY();
    float dz = atom1.getZ() - atom2.getZ();
    return std::sqrt(dx*dx + dy*dy + dz*dz);
}

__host__ __device__ float Molecule::calculateOverlap(const Atom& atom1, const Atom& atom2, float distance) const {
    // Implement formula for calculating atomic volume overlap
    // This is a simplified example; you may need a more sophisticated formula
    float sumRadii = atom1.getVanDerWaalsRadius() + atom2.getVanDerWaalsRadius();
    if (distance >= sumRadii) return 0.0f;
    return 0.5f * (sumRadii - distance) * (sumRadii - distance) * (sumRadii + 2.0f * distance) / (distance * distance);
}

__host__ __device__ float Molecule::calculateSecondShellCorrection(const Atom& atom1, const Atom& atom2, float distance) const {
    // Implement more sophisticated correction based on atom positions
    // This is a placeholder; you'll need to implement the actual formula
    return 0.0f;
}

__host__ __device__ void Molecule::initializeAtomPositions() {
    // This is a simplified initialization. You may need a more sophisticated
    // approach based on actual molecular geometry.
    float spacing = 1.5f; // Arbitrary spacing between atoms
    float x = 0.0f, y = 0.0f, z = 0.0f;

    for (int i = 0; i < atomCount; ++i) {
        atoms[i].setPosition(x, y, z);

        // Move to the next position
        x += spacing;
        if (x > 5.0f * spacing) {
            x = 0.0f;
            y += spacing;
            if (y > 5.0f * spacing) {
                y = 0.0f;
                z += spacing;
            }
        }
    }
}

__host__ __device__ Molecule Molecule::createGlucose() {
    Molecule m;
    m.type = GLUCOSE;
    m.atomCount = 0;
    
    // C6H12O6
    for (int i = 0; i < 6 && m.atomCount < MAX_ATOMS_PER_MOLECULE; ++i) {
        m.atoms[m.atomCount++] = Atom(CARBON, 0, 0, 0, 0.1f, 12.0107f);
    }
    for (int i = 0; i < 12 && m.atomCount < MAX_ATOMS_PER_MOLECULE; ++i) {
        m.atoms[m.atomCount++] = Atom(HYDROGEN, 0, 0, 0, 0.05f, 1.00794f);
    }
    for (int i = 0; i < 6 && m.atomCount < MAX_ATOMS_PER_MOLECULE; ++i) {
        m.atoms[m.atomCount++] = Atom(OXYGEN, 0, 0, 0, -0.2f, 15.9994f);
    }

    m.initializeAtomPositions();
    m.calculateBornRadii();
    return m;
}

__host__ __device__ Molecule Molecule::createATP() {
    Molecule m;
    m.type = ATP;
    
    // C10H16N5O13P3
    for (int i = 0; i < 10; ++i) {
        m.atoms[m.atomCount++] = Atom(CARBON, 0, 0, 0, 0.1f, 12.0107f);
    }
    for (int i = 0; i < 16; ++i) {
        m.atoms[m.atomCount++] = Atom(HYDROGEN, 0, 0, 0, 0.05f, 1.00794f);
    }
    for (int i = 0; i < 5; ++i) {
        m.atoms[m.atomCount++] = Atom(NITROGEN, 0, 0, 0, -0.3f, 14.0067f);
    }
    for (int i = 0; i < 13; ++i) {
        m.atoms[m.atomCount++] = Atom(OXYGEN, 0, 0, 0, -0.2f, 15.9994f);
    }
    for (int i = 0; i < 3; ++i) {
        m.atoms[m.atomCount++] = Atom(PHOSPHORUS, 0, 0, 0, 0.3f, 30.973762f);
    }

    m.initializeAtomPositions();
    m.calculateBornRadii();
    return m;
}

__host__ __device__ Molecule Molecule::createADP() {
    Molecule m;
    m.type = ADP;
    
    // C10H15N5O10P2
    for (int i = 0; i < 10; ++i) {
        m.atoms[m.atomCount++] = Atom(CARBON, 0, 0, 0, 0.1f, 12.0107f);
    }
    for (int i = 0; i < 15; ++i) {
        m.atoms[m.atomCount++] = Atom(HYDROGEN, 0, 0, 0, 0.05f, 1.00794f);
    }
    for (int i = 0; i < 5; ++i) {
        m.atoms[m.atomCount++] = Atom(NITROGEN, 0, 0, 0, -0.3f, 14.0067f);
    }
    for (int i = 0; i < 10; ++i) {
        m.atoms[m.atomCount++] = Atom(OXYGEN, 0, 0, 0, -0.2f, 15.9994f);
    }
    for (int i = 0; i < 2; ++i) {
        m.atoms[m.atomCount++] = Atom(PHOSPHORUS, 0, 0, 0, 0.3f, 30.973762f);
    }

    m.initializeAtomPositions();
    m.calculateBornRadii();
    return m;
}

__host__ __device__ Molecule Molecule::createGlucose6Phosphate() {
    Molecule m;
    m.type = GLUCOSE_6_PHOSPHATE;
    
    // C6H13O9P
    for (int i = 0; i < 6; ++i) {
        m.atoms[m.atomCount++] = Atom(CARBON, 0, 0, 0, 0.1f, 12.0107f);
    }
    for (int i = 0; i < 13; ++i) {
        m.atoms[m.atomCount++] = Atom(HYDROGEN, 0, 0, 0, 0.05f, 1.00794f);
    }
    for (int i = 0; i < 9; ++i) {
        m.atoms[m.atomCount++] = Atom(OXYGEN, 0, 0, 0, -0.2f, 15.9994f);
    }
    m.atoms[m.atomCount++] = Atom(PHOSPHORUS, 0, 0, 0, 0.3f, 30.973762f);

    m.initializeAtomPositions();
    m.calculateBornRadii();
    return m;
}

__host__ __device__ Molecule Molecule::createWater() {
    Molecule m;
    m.type = WATER;
    
    // H2O
    m.atoms[m.atomCount++] = Atom(OXYGEN, 0, 0, 0, -0.834f, 15.9994f);
    m.atoms[m.atomCount++] = Atom(HYDROGEN, 0, 0, 0, 0.417f, 1.00794f);
    m.atoms[m.atomCount++] = Atom(HYDROGEN, 0, 0, 0, 0.417f, 1.00794f);

    m.initializeAtomPositions();
    m.calculateBornRadii();
    return m;
}

__host__ __device__ Molecule Molecule::createHexokinase() {
    Molecule m;
    m.type = HEXOKINASE;
    m.representation = COARSE_GRAINED;
    m.centerOfMass = make_float3(0.0f, 0.0f, 0.0f);
    m.radius = 5.0f;  // Approximate radius of the enzyme
    m.mass = 100000.0f;  // Approximate mass of the enzyme
    return m;
}

__host__ __device__ Molecule Molecule::createFructose6Phosphate() {
    Molecule m;
    m.type = FRUCTOSE_6_PHOSPHATE;
    
    // C6H13O9P
    for (int i = 0; i < 6; ++i) {
        m.atoms[m.atomCount++] = Atom(CARBON, 0, 0, 0, 0.1f, 12.0107f);
    }
    for (int i = 0; i < 13; ++i) {
        m.atoms[m.atomCount++] = Atom(HYDROGEN, 0, 0, 0, 0.05f, 1.00794f);
    }
    for (int i = 0; i < 9; ++i) {
        m.atoms[m.atomCount++] = Atom(OXYGEN, 0, 0, 0, -0.2f, 15.9994f);
    }
    m.atoms[m.atomCount++] = Atom(PHOSPHORUS, 0, 0, 0, 0.3f, 30.973762f);

    m.initializeAtomPositions();
    m.calculateBornRadii();
    return m;
}

__host__ __device__ Molecule Molecule::createFructose16Bisphosphate() {
    Molecule m;
    m.type = FRUCTOSE_1_6_BISPHOSPHATE;
    
    // C6H14O12P2
    for (int i = 0; i < 6; ++i) {
        m.atoms[m.atomCount++] = Atom(CARBON, 0, 0, 0, 0.1f, 12.0107f);
    }
    for (int i = 0; i < 14; ++i) {
        m.atoms[m.atomCount++] = Atom(HYDROGEN, 0, 0, 0, 0.05f, 1.00794f);
    }
    for (int i = 0; i < 12; ++i) {
        m.atoms[m.atomCount++] = Atom(OXYGEN, 0, 0, 0, -0.2f, 15.9994f);
    }
    for (int i = 0; i < 2; ++i) {
        m.atoms[m.atomCount++] = Atom(PHOSPHORUS, 0, 0, 0, 0.3f, 30.973762f);
    }

    m.initializeAtomPositions();
    m.calculateBornRadii();
    return m;
}

__host__ __device__ Molecule Molecule::createDihydroxyacetonePhosphate() {
    Molecule m;
    m.type = DIHYDROXYACETONE_PHOSPHATE;
    
    // C3H7O6P
    for (int i = 0; i < 3; ++i) {
        m.atoms[m.atomCount++] = Atom(CARBON, 0, 0, 0, 0.1f, 12.0107f);
    }
    for (int i = 0; i < 7; ++i) {
        m.atoms[m.atomCount++] = Atom(HYDROGEN, 0, 0, 0, 0.05f, 1.00794f);
    }
    for (int i = 0; i < 6; ++i) {
        m.atoms[m.atomCount++] = Atom(OXYGEN, 0, 0, 0, -0.2f, 15.9994f);
    }
    m.atoms[m.atomCount++] = Atom(PHOSPHORUS, 0, 0, 0, 0.3f, 30.973762f);

    m.initializeAtomPositions();
    m.calculateBornRadii();
    return m;
}

__host__ __device__ Molecule Molecule::createGlyceraldehyde3Phosphate() {
    Molecule m;
    m.type = GLYCERALDEHYDE_3_PHOSPHATE;
    
    // C3H7O6P
    for (int i = 0; i < 3; ++i) {
        m.atoms[m.atomCount++] = Atom(CARBON, 0, 0, 0, 0.1f, 12.0107f);
    }
    for (int i = 0; i < 7; ++i) {
        m.atoms[m.atomCount++] = Atom(HYDROGEN, 0, 0, 0, 0.05f, 1.00794f);
    }
    for (int i = 0; i < 6; ++i) {
        m.atoms[m.atomCount++] = Atom(OXYGEN, 0, 0, 0, -0.2f, 15.9994f);
    }
    m.atoms[m.atomCount++] = Atom(PHOSPHORUS, 0, 0, 0, 0.3f, 30.973762f);

    m.initializeAtomPositions();
    m.calculateBornRadii();
    return m;
}

__host__ __device__ Molecule Molecule::create13Bisphosphoglycerate() {
    Molecule m;
    m.type = _1_3_BISPHOSPHOGLYCERATE;
    
    // C3H8O10P2
    for (int i = 0; i < 3; ++i) {
        m.atoms[m.atomCount++] = Atom(CARBON, 0, 0, 0, 0.1f, 12.0107f);
    }
    for (int i = 0; i < 8; ++i) {
        m.atoms[m.atomCount++] = Atom(HYDROGEN, 0, 0, 0, 0.05f, 1.00794f);
    }
    for (int i = 0; i < 10; ++i) {
        m.atoms[m.atomCount++] = Atom(OXYGEN, 0, 0, 0, -0.2f, 15.9994f);
    }
    for (int i = 0; i < 2; ++i) {
        m.atoms[m.atomCount++] = Atom(PHOSPHORUS, 0, 0, 0, 0.3f, 30.973762f);
    }

    m.initializeAtomPositions();
    m.calculateBornRadii();
    return m;
}

__host__ __device__ Molecule Molecule::create3Phosphoglycerate() {
    Molecule m;
    m.type = _3_PHOSPHOGLYCERATE;
    
    // C3H7O7P
    for (int i = 0; i < 3; ++i) {
        m.atoms[m.atomCount++] = Atom(CARBON, 0, 0, 0, 0.1f, 12.0107f);
    }
    for (int i = 0; i < 7; ++i) {
        m.atoms[m.atomCount++] = Atom(HYDROGEN, 0, 0, 0, 0.05f, 1.00794f);
    }
    for (int i = 0; i < 7; ++i) {
        m.atoms[m.atomCount++] = Atom(OXYGEN, 0, 0, 0, -0.2f, 15.9994f);
    }
    m.atoms[m.atomCount++] = Atom(PHOSPHORUS, 0, 0, 0, 0.3f, 30.973762f);

    m.initializeAtomPositions();
    m.calculateBornRadii();
    return m;
}

__host__ __device__ Molecule Molecule::create2Phosphoglycerate() {
    Molecule m;
    m.type = _2_PHOSPHOGLYCERATE;
    
    // C3H7O7P
    for (int i = 0; i < 3; ++i) {
        m.atoms[m.atomCount++] = Atom(CARBON, 0, 0, 0, 0.1f, 12.0107f);
    }
    for (int i = 0; i < 7; ++i) {
        m.atoms[m.atomCount++] = Atom(HYDROGEN, 0, 0, 0, 0.05f, 1.00794f);
    }
    for (int i = 0; i < 7; ++i) {
        m.atoms[m.atomCount++] = Atom(OXYGEN, 0, 0, 0, -0.2f, 15.9994f);
    }
    m.atoms[m.atomCount++] = Atom(PHOSPHORUS, 0, 0, 0, 0.3f, 30.973762f);

    m.initializeAtomPositions();
    m.calculateBornRadii();
    return m;
}

__host__ __device__ Molecule Molecule::createPhosphoenolpyruvate() {
    Molecule m;
    m.type = PHOSPHOENOLPYRUVATE;
    
    // C3H5O6P
    for (int i = 0; i < 3; ++i) {
        m.atoms[m.atomCount++] = Atom(CARBON, 0, 0, 0, 0.1f, 12.0107f);
    }
    for (int i = 0; i < 5; ++i) {
        m.atoms[m.atomCount++] = Atom(HYDROGEN, 0, 0, 0, 0.05f, 1.00794f);
    }
    for (int i = 0; i < 6; ++i) {
        m.atoms[m.atomCount++] = Atom(OXYGEN, 0, 0, 0, -0.2f, 15.9994f);
    }
    m.atoms[m.atomCount++] = Atom(PHOSPHORUS, 0, 0, 0, 0.3f, 30.973762f);

    m.initializeAtomPositions();
    m.calculateBornRadii();
    return m;
}

__host__ __device__ Molecule Molecule::createPyruvate() {
    Molecule m;
    m.type = PYRUVATE;
    
    // C3H4O3
    for (int i = 0; i < 3; ++i) {
        m.atoms[m.atomCount++] = Atom(CARBON, 0, 0, 0, 0.1f, 12.0107f);
    }
    for (int i = 0; i < 4; ++i) {
        m.atoms[m.atomCount++] = Atom(HYDROGEN, 0, 0, 0, 0.05f, 1.00794f);
    }
    for (int i = 0; i < 3; ++i) {
        m.atoms[m.atomCount++] = Atom(OXYGEN, 0, 0, 0, -0.2f, 15.9994f);
    }

    m.initializeAtomPositions();
    m.calculateBornRadii();
    return m;
}

__host__ __device__ Molecule Molecule::createNADPlus() {
    Molecule m;
    m.type = NAD_PLUS;
    
    // C21H28N7O14P2 (simplified)
    for (int i = 0; i < 21; ++i) {
        m.atoms[m.atomCount++] = Atom(CARBON, 0, 0, 0, 0.1f, 12.0107f);
    }
    for (int i = 0; i < 28; ++i) {
        m.atoms[m.atomCount++] = Atom(HYDROGEN, 0, 0, 0, 0.05f, 1.00794f);
    }
    for (int i = 0; i < 7; ++i) {
        m.atoms[m.atomCount++] = Atom(NITROGEN, 0, 0, 0, -0.3f, 14.0067f);
    }
    for (int i = 0; i < 14; ++i) {
        m.atoms[m.atomCount++] = Atom(OXYGEN, 0, 0, 0, -0.2f, 15.9994f);
    }
    for (int i = 0; i < 2; ++i) {
        m.atoms[m.atomCount++] = Atom(PHOSPHORUS, 0, 0, 0, 0.3f, 30.973762f);
    }

    m.initializeAtomPositions();
    m.calculateBornRadii();
    return m;
}

__host__ __device__ Molecule Molecule::createNADH() {
    Molecule m;
    m.type = NADH;
    
    // C21H29N7O14P2 (simplified)
    for (int i = 0; i < 21; ++i) {
        m.atoms[m.atomCount++] = Atom(CARBON, 0, 0, 0, 0.1f, 12.0107f);
    }
    for (int i = 0; i < 29; ++i) {
        m.atoms[m.atomCount++] = Atom(HYDROGEN, 0, 0, 0, 0.05f, 1.00794f);
    }
    for (int i = 0; i < 7; ++i) {
        m.atoms[m.atomCount++] = Atom(NITROGEN, 0, 0, 0, -0.3f, 14.0067f);
    }
    for (int i = 0; i < 14; ++i) {
        m.atoms[m.atomCount++] = Atom(OXYGEN, 0, 0, 0, -0.2f, 15.9994f);
    }
    for (int i = 0; i < 2; ++i) {
        m.atoms[m.atomCount++] = Atom(PHOSPHORUS, 0, 0, 0, 0.3f, 30.973762f);
    }

    m.initializeAtomPositions();
    m.calculateBornRadii();
    return m;
}

__host__ __device__ Molecule Molecule::createProton() {
    Molecule m;
    m.type = PROTON;
    
    // H+
    m.atoms[m.atomCount++] = Atom(HYDROGEN, 0, 0, 0, 1.0f, 1.00794f);

    m.initializeAtomPositions();
    m.calculateBornRadii();
    return m;
}

__host__ __device__ Molecule Molecule::createInorganicPhosphate() {
    Molecule m;
    m.type = INORGANIC_PHOSPHATE;
    
    // PO4^3-
    m.atoms[m.atomCount++] = Atom(PHOSPHORUS, 0, 0, 0, 0.3f, 30.973762f);
    for (int i = 0; i < 4; ++i) {
        m.atoms[m.atomCount++] = Atom(OXYGEN, 0, 0, 0, -0.825f, 15.9994f);
    }

    m.initializeAtomPositions();
    m.calculateBornRadii();
    return m;
}

// Simplified representations for enzymes
__host__ __device__ Molecule Molecule::createGlucose6PhosphateIsomerase() {
    Molecule m;
    m.type = GLUCOSE_6_PHOSPHATE_ISOMERASE;
    m.representation = COARSE_GRAINED;
    m.centerOfMass = make_float3(0.0f, 0.0f, 0.0f);
    m.radius = 4.5f;  // Approximate radius of the enzyme
    m.mass = 90000.0f;  // Approximate mass of the enzyme
    return m;
}

__host__ __device__ Molecule Molecule::createPhosphofructokinase1() {
    Molecule m;
    m.type = PHOSPHOFRUCTOKINASE_1;
    m.representation = COARSE_GRAINED;
    m.centerOfMass = make_float3(0.0f, 0.0f, 0.0f);
    m.radius = 5.5f;  // Approximate radius of the enzyme
    m.mass = 110000.0f;  // Approximate mass of the enzyme
    return m;
}

__host__ __device__ Molecule Molecule::createAldolase() {
    Molecule m;
    m.type = ALDOLASE;
    m.representation = COARSE_GRAINED;
    m.centerOfMass = make_float3(0.0f, 0.0f, 0.0f);
    m.radius = 5.0f;  // Approximate radius of the enzyme
    m.mass = 95000.0f;  // Approximate mass of the enzyme
    return m;
}

__host__ __device__ Molecule Molecule::createTriosephosphateIsomerase() {
    Molecule m;
    m.type = TRIOSEPHOSPHATE_ISOMERASE;
    m.representation = COARSE_GRAINED;
    m.centerOfMass = make_float3(0.0f, 0.0f, 0.0f);
    m.radius = 4.0f;  // Approximate radius of the enzyme
    m.mass = 80000.0f;  // Approximate mass of the enzyme
    return m;
}

__host__ __device__ Molecule Molecule::createGlyceraldehyde3PhosphateDehydrogenase() {
    Molecule m;
    m.type = GLYCERALDEHYDE_3_PHOSPHATE_DEHYDROGENASE;
    m.representation = COARSE_GRAINED;
    m.centerOfMass = make_float3(0.0f, 0.0f, 0.0f);
    m.radius = 5.2f;  // Approximate radius of the enzyme
    m.mass = 105000.0f;  // Approximate mass of the enzyme
    return m;
}

__host__ __device__ Molecule Molecule::createPhosphoglycerateKinase() {
    Molecule m;
    m.type = PHOSPHOGLYCERATE_KINASE;
    m.representation = COARSE_GRAINED;
    m.centerOfMass = make_float3(0.0f, 0.0f, 0.0f);
    m.radius = 4.8f;  // Approximate radius of the enzyme
    m.mass = 92000.0f;  // Approximate mass of the enzyme
    return m;
}

__host__ __device__ Molecule Molecule::createPhosphoglycerateMutase() {
    Molecule m;
    m.type = PHOSPHOGLYCERATE_MUTASE;
    m.representation = COARSE_GRAINED;
    m.centerOfMass = make_float3(0.0f, 0.0f, 0.0f);
    m.radius = 4.3f;  // Approximate radius of the enzyme
    m.mass = 85000.0f;  // Approximate mass of the enzyme
    return m;
}

__host__ __device__ Molecule Molecule::createEnolase() {
    Molecule m;
    m.type = ENOLASE;
    m.representation = COARSE_GRAINED;
    m.centerOfMass = make_float3(0.0f, 0.0f, 0.0f);
    m.radius = 4.7f;  // Approximate radius of the enzyme
    m.mass = 88000.0f;  // Approximate mass of the enzyme
    return m;
}

__host__ __device__ Molecule Molecule::createPyruvateKinase() {
    Molecule m;
    m.type = PYRUVATE_KINASE;
    m.representation = COARSE_GRAINED;
    m.centerOfMass = make_float3(0.0f, 0.0f, 0.0f);
    m.radius = 5.3f;  // Approximate radius of the enzyme
    m.mass = 108000.0f;  // Approximate mass of the enzyme
    return m;
}

__host__ __device__ void Molecule::setPosition(float newX, float newY, float newZ) {
  
    // Update positions of all atoms relative to the new center
    float dx = newX - getX();
    float dy = newY - getY();
    float dz = newZ - getZ();
    
    for (int i = 0; i < atomCount; ++i) {
        atoms[i].setPosition(atoms[i].getX() + dx, atoms[i].getY() + dy, atoms[i].getZ() + dz);
    }
}
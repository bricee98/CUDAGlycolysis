#include "Molecule.h"
#include <cmath>

Molecule::Molecule()
    : type(WATER), vx(0), vy(0), vz(0), bornRadius(0) {}

Molecule::~Molecule() {}

void Molecule::updatePosition(float dt) {
    for (Atom& atom : atoms) {
        atom.x += vx * dt;
        atom.y += vy * dt;
        atom.z += vz * dt;
    }
    calculateBornRadius();
}

void Molecule::applyForce(float fx, float fy, float fz) {
    float totalMass = getTotalMass();
    vx += fx / totalMass;
    vy += fy / totalMass;
    vz += fz / totalMass;
}

float Molecule::getTotalMass() const {
    float totalMass = 0;
    for (const Atom& atom : atoms) {
        totalMass += atom.mass;
    }
    return totalMass;
}

void Molecule::getPosition(float& outX, float& outY, float& outZ) const {
    outX = getX();
    outY = getY();
    outZ = getZ();
}

void Molecule::getVelocity(float& outVx, float& outVy, float& outVz) const {
    outVx = vx;
    outVy = vy;
    outVz = vz;
}

float Molecule::getX() const {
    float totalMass = getTotalMass();
    float centerX = 0;
    for (const Atom& atom : atoms) {
        centerX += atom.x * atom.mass;
    }
    return centerX / totalMass;
}

float Molecule::getY() const {
    float totalMass = getTotalMass();
    float centerY = 0;
    for (const Atom& atom : atoms) {
        centerY += atom.y * atom.mass;
    }
    return centerY / totalMass;
}

float Molecule::getZ() const {
    float totalMass = getTotalMass();
    float centerZ = 0;
    for (const Atom& atom : atoms) {
        centerZ += atom.z * atom.mass;
    }
    return centerZ / totalMass;
}

void Molecule::calculateBornRadius() {
    const float minAllowedRadius = 0.1f; // Minimum allowed radius to prevent division by zero

    for (Atom& atom : atoms) {
        // Initial estimate based on atom type
        atom.bornRadius = initialEstimate(atom.type);
        
        // First shell correction
        for (const Atom& otherAtom : atoms) {
            if (&atom != &otherAtom) {
                float distance = calculateDistance(atom, otherAtom);
                atom.bornRadius -= calculateOverlap(atom, otherAtom, distance);
            }
        }
        
        // Convert to inverse Born radius
        atom.inverseBornRadius = 1.0f / std::max(atom.bornRadius, minAllowedRadius);
    }
    
    // Second shell correction (optional, for even more accuracy)
    for (Atom& atom : atoms) {
        float correction = 0.0f;
        for (const Atom& otherAtom : atoms) {
            if (&atom != &otherAtom) {
                float distance = calculateDistance(atom, otherAtom);
                correction += calculateSecondShellCorrection(atom, otherAtom, distance);
            }
        }
        atom.inverseBornRadius += correction;
    }
}

float Molecule::calculateDistance(const Atom& atom1, const Atom& atom2) const {
    float dx = atom1.x - atom2.x;
    float dy = atom1.y - atom2.y;
    float dz = atom1.z - atom2.z;
    return std::sqrt(dx*dx + dy*dy + dz*dz);
}

float Molecule::calculateOverlap(const Atom& atom1, const Atom& atom2, float distance) const {
    // Implement formula for calculating atomic volume overlap
    // This is a simplified example; you may need a more sophisticated formula
    float sumRadii = atom1.radius + atom2.radius;
    if (distance >= sumRadii) return 0.0f;
    return 0.5f * (sumRadii - distance) * (sumRadii - distance) * (sumRadii + 2.0f * distance) / (distance * distance);
}

float Molecule::calculateSecondShellCorrection(const Atom& atom1, const Atom& atom2, float distance) const {
    // Implement more sophisticated correction based on atom positions
    // This is a placeholder; you'll need to implement the actual formula
    return 0.0f;
}

float Molecule::initialEstimate(AtomType type) const {
    // Provide initial estimates based on atom type
    switch (type) {
        case HYDROGEN: return 1.2f;
        case CARBON: return 1.7f;
        case NITROGEN: return 1.55f;
        case OXYGEN: return 1.52f;
        case PHOSPHORUS: return 1.8f;
        default: return 1.5f; // Default value
    }
}

void Molecule::initializeAtomPositions() {
    // This is a simplified initialization. You may need a more sophisticated
    // approach based on actual molecular geometry.
    float spacing = 1.5f; // Arbitrary spacing between atoms
    float x = 0.0f, y = 0.0f, z = 0.0f;

    for (Atom& atom : atoms) {
        atom.x = x;
        atom.y = y;
        atom.z = z;

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

// Example of a molecule creation function
Molecule Molecule::createGlucose() {
    Molecule m;
    m.type = GLUCOSE;
    
    // C6H12O6
    for (int i = 0; i < 6; ++i) {
        m.atoms.push_back({CARBON, 0, 0, 0, 0.1f, 1.7f, 12.0107f, 0.0f, 0.0f});
    }
    for (int i = 0; i < 12; ++i) {
        m.atoms.push_back({HYDROGEN, 0, 0, 0, 0.05f, 1.2f, 1.00794f, 0.0f, 0.0f});
    }
    for (int i = 0; i < 6; ++i) {
        m.atoms.push_back({OXYGEN, 0, 0, 0, -0.2f, 1.52f, 15.9994f, 0.0f, 0.0f});
    }

    m.initializeAtomPositions();
    m.calculateBornRadius();
    return m;
}

Molecule Molecule::createATP() {
    Molecule m;
    m.type = ATP;
    
    // C10H16N5O13P3
    for (int i = 0; i < 10; ++i) m.atoms.push_back({CARBON, 0, 0, 0, 0.1f, 1.7f, 12.0107f, 0.0f, 0.0f});
    for (int i = 0; i < 16; ++i) m.atoms.push_back({HYDROGEN, 0, 0, 0, 0.05f, 1.2f, 1.00794f, 0.0f, 0.0f});
    for (int i = 0; i < 5; ++i) m.atoms.push_back({NITROGEN, 0, 0, 0, -0.3f, 1.55f, 14.0067f, 0.0f, 0.0f});
    for (int i = 0; i < 13; ++i) m.atoms.push_back({OXYGEN, 0, 0, 0, -0.2f, 1.52f, 15.9994f, 0.0f, 0.0f});
    for (int i = 0; i < 3; ++i) m.atoms.push_back({PHOSPHORUS, 0, 0, 0, 0.3f, 1.8f, 30.973762f, 0.0f, 0.0f});

    m.initializeAtomPositions();
    m.calculateBornRadius();
    return m;
}

Molecule Molecule::createADP() {
    Molecule m;
    m.type = ADP;
    
    // C10H15N5O10P2
    for (int i = 0; i < 10; ++i) m.atoms.push_back({CARBON, 0, 0, 0, 0.1f, 1.7f, 12.0107f, 0.0f, 0.0f});
    for (int i = 0; i < 15; ++i) m.atoms.push_back({HYDROGEN, 0, 0, 0, 0.05f, 1.2f, 1.00794f, 0.0f, 0.0f});
    for (int i = 0; i < 5; ++i) m.atoms.push_back({NITROGEN, 0, 0, 0, -0.3f, 1.55f, 14.0067f, 0.0f, 0.0f});
    for (int i = 0; i < 10; ++i) m.atoms.push_back({OXYGEN, 0, 0, 0, -0.2f, 1.52f, 15.9994f, 0.0f, 0.0f});
    for (int i = 0; i < 2; ++i) m.atoms.push_back({PHOSPHORUS, 0, 0, 0, 0.3f, 1.8f, 30.973762f, 0.0f, 0.0f});

    m.initializeAtomPositions();
    m.calculateBornRadius();
    return m;
}

Molecule Molecule::createGlucose6Phosphate() {
    Molecule m;
    m.type = GLUCOSE_6_PHOSPHATE;
    
    // C6H13O9P
    for (int i = 0; i < 6; ++i) m.atoms.push_back({CARBON, 0, 0, 0, 0.1f, 1.7f, 12.0107f, 0.0f, 0.0f});
    for (int i = 0; i < 13; ++i) m.atoms.push_back({HYDROGEN, 0, 0, 0, 0.05f, 1.2f, 1.00794f, 0.0f, 0.0f});
    for (int i = 0; i < 9; ++i) m.atoms.push_back({OXYGEN, 0, 0, 0, -0.2f, 1.52f, 15.9994f, 0.0f, 0.0f});
    m.atoms.push_back({PHOSPHORUS, 0, 0, 0, 0.3f, 1.8f, 30.973762f, 0.0f, 0.0f});

    m.initializeAtomPositions();
    m.calculateBornRadius();
    return m;
}

Molecule Molecule::createFructose6Phosphate() {
    Molecule m;
    m.type = FRUCTOSE_6_PHOSPHATE;
    
    // C6H13O9P (isomer of Glucose-6-Phosphate)
    for (int i = 0; i < 6; ++i) m.atoms.push_back({CARBON, 0, 0, 0, 0.1f, 1.7f, 12.0107f, 0.0f, 0.0f});
    for (int i = 0; i < 13; ++i) m.atoms.push_back({HYDROGEN, 0, 0, 0, 0.05f, 1.2f, 1.00794f, 0.0f, 0.0f});
    for (int i = 0; i < 9; ++i) m.atoms.push_back({OXYGEN, 0, 0, 0, -0.2f, 1.52f, 15.9994f, 0.0f, 0.0f});
    m.atoms.push_back({PHOSPHORUS, 0, 0, 0, 0.3f, 1.8f, 30.973762f, 0.0f, 0.0f});

    m.initializeAtomPositions();
    m.calculateBornRadius();
    return m;
}

Molecule Molecule::createFructose16Bisphosphate() {
    Molecule m;
    m.type = FRUCTOSE_1_6_BISPHOSPHATE;
    
    // C6H14O12P2
    for (int i = 0; i < 6; ++i) m.atoms.push_back({CARBON, 0, 0, 0, 0.1f, 1.7f, 12.0107f, 0.0f, 0.0f});
    for (int i = 0; i < 14; ++i) m.atoms.push_back({HYDROGEN, 0, 0, 0, 0.05f, 1.2f, 1.00794f, 0.0f, 0.0f});
    for (int i = 0; i < 12; ++i) m.atoms.push_back({OXYGEN, 0, 0, 0, -0.2f, 1.52f, 15.9994f, 0.0f, 0.0f});
    for (int i = 0; i < 2; ++i) m.atoms.push_back({PHOSPHORUS, 0, 0, 0, 0.3f, 1.8f, 30.973762f, 0.0f, 0.0f});

    m.initializeAtomPositions();
    m.calculateBornRadius();
    return m;
}

Molecule Molecule::createDihydroxyacetonePhosphate() {
    Molecule m;
    m.type = DIHYDROXYACETONE_PHOSPHATE;
    
    // C3H7O6P
    for (int i = 0; i < 3; ++i) m.atoms.push_back({CARBON, 0, 0, 0, 0.1f, 1.7f, 12.0107f, 0.0f, 0.0f});
    for (int i = 0; i < 7; ++i) m.atoms.push_back({HYDROGEN, 0, 0, 0, 0.05f, 1.2f, 1.00794f, 0.0f, 0.0f});
    for (int i = 0; i < 6; ++i) m.atoms.push_back({OXYGEN, 0, 0, 0, -0.2f, 1.52f, 15.9994f, 0.0f, 0.0f});
    m.atoms.push_back({PHOSPHORUS, 0, 0, 0, 0.3f, 1.8f, 30.973762f, 0.0f, 0.0f});

    m.initializeAtomPositions();
    m.calculateBornRadius();
    return m;
}

Molecule Molecule::createGlyceraldehyde3Phosphate() {
    Molecule m;
    m.type = GLYCERALDEHYDE_3_PHOSPHATE;
    
    // C3H7O6P
    for (int i = 0; i < 3; ++i) m.atoms.push_back({CARBON, 0, 0, 0, 0.1f, 1.7f, 12.0107f, 0.0f, 0.0f});
    for (int i = 0; i < 7; ++i) m.atoms.push_back({HYDROGEN, 0, 0, 0, 0.05f, 1.2f, 1.00794f, 0.0f, 0.0f});
    for (int i = 0; i < 6; ++i) m.atoms.push_back({OXYGEN, 0, 0, 0, -0.2f, 1.52f, 15.9994f, 0.0f, 0.0f});
    m.atoms.push_back({PHOSPHORUS, 0, 0, 0, 0.3f, 1.8f, 30.973762f, 0.0f, 0.0f});

    m.initializeAtomPositions();
    m.calculateBornRadius();
    return m;
}

Molecule Molecule::create13Bisphosphoglycerate() {
    Molecule m;
    m.type = _1_3_BISPHOSPHOGLYCERATE;
    
    // C3H8O10P2
    for (int i = 0; i < 3; ++i) m.atoms.push_back({CARBON, 0, 0, 0, 0.1f, 1.7f, 12.0107f, 0.0f, 0.0f});
    for (int i = 0; i < 8; ++i) m.atoms.push_back({HYDROGEN, 0, 0, 0, 0.05f, 1.2f, 1.00794f, 0.0f, 0.0f});
    for (int i = 0; i < 10; ++i) m.atoms.push_back({OXYGEN, 0, 0, 0, -0.2f, 1.52f, 15.9994f, 0.0f, 0.0f});
    for (int i = 0; i < 2; ++i) m.atoms.push_back({PHOSPHORUS, 0, 0, 0, 0.3f, 1.8f, 30.973762f, 0.0f, 0.0f});

    m.initializeAtomPositions();
    m.calculateBornRadius();
    return m;
}

Molecule Molecule::create3Phosphoglycerate() {
    Molecule m;
    m.type = _3_PHOSPHOGLYCERATE;
    
    // C3H7O7P
    for (int i = 0; i < 3; ++i) m.atoms.push_back({CARBON, 0, 0, 0, 0.1f, 1.7f, 12.0107f, 0.0f, 0.0f});
    for (int i = 0; i < 7; ++i) m.atoms.push_back({HYDROGEN, 0, 0, 0, 0.05f, 1.2f, 1.00794f, 0.0f, 0.0f});
    for (int i = 0; i < 7; ++i) m.atoms.push_back({OXYGEN, 0, 0, 0, -0.2f, 1.52f, 15.9994f, 0.0f, 0.0f});
    m.atoms.push_back({PHOSPHORUS, 0, 0, 0, 0.3f, 1.8f, 30.973762f, 0.0f, 0.0f});

    m.initializeAtomPositions();
    m.calculateBornRadius();
    return m;
}

Molecule Molecule::create2Phosphoglycerate() {
    Molecule m;
    m.type = _2_PHOSPHOGLYCERATE;
    
    // C3H7O7P
    for (int i = 0; i < 3; ++i) m.atoms.push_back({CARBON, 0, 0, 0, 0.1f, 1.7f, 12.0107f, 0.0f, 0.0f});
    for (int i = 0; i < 7; ++i) m.atoms.push_back({HYDROGEN, 0, 0, 0, 0.05f, 1.2f, 1.00794f, 0.0f, 0.0f});
    for (int i = 0; i < 7; ++i) m.atoms.push_back({OXYGEN, 0, 0, 0, -0.2f, 1.52f, 15.9994f, 0.0f, 0.0f});
    m.atoms.push_back({PHOSPHORUS, 0, 0, 0, 0.3f, 1.8f, 30.973762f, 0.0f, 0.0f});

    m.initializeAtomPositions();
    m.calculateBornRadius();
    return m;
}

Molecule Molecule::createPhosphoenolpyruvate() {
    Molecule m;
    m.type = PHOSPHOENOLPYRUVATE;
    
    // C3H5O6P
    for (int i = 0; i < 3; ++i) m.atoms.push_back({CARBON, 0, 0, 0, 0.1f, 1.7f, 12.0107f, 0.0f, 0.0f});
    for (int i = 0; i < 5; ++i) m.atoms.push_back({HYDROGEN, 0, 0, 0, 0.05f, 1.2f, 1.00794f, 0.0f, 0.0f});
    for (int i = 0; i < 6; ++i) m.atoms.push_back({OXYGEN, 0, 0, 0, -0.2f, 1.52f, 15.9994f, 0.0f, 0.0f});
    m.atoms.push_back({PHOSPHORUS, 0, 0, 0, 0.3f, 1.8f, 30.973762f, 0.0f, 0.0f});

    m.initializeAtomPositions();
    m.calculateBornRadius();
    return m;
}

Molecule Molecule::createPyruvate() {
    Molecule m;
    m.type = PYRUVATE;
    
    // C3H4O3
    for (int i = 0; i < 3; ++i) m.atoms.push_back({CARBON, 0, 0, 0, 0.1f, 1.7f, 12.0107f, 0.0f, 0.0f});
    for (int i = 0; i < 4; ++i) m.atoms.push_back({HYDROGEN, 0, 0, 0, 0.05f, 1.2f, 1.00794f, 0.0f, 0.0f});
    for (int i = 0; i < 3; ++i) m.atoms.push_back({OXYGEN, 0, 0, 0, -0.2f, 1.52f, 15.9994f, 0.0f, 0.0f});

    m.initializeAtomPositions();
    m.calculateBornRadius();
    return m;
}

Molecule Molecule::createNADPlus() {
    Molecule m;
    m.type = NAD_PLUS;
    
    // C21H28N7O14P2
    for (int i = 0; i < 21; ++i) m.atoms.push_back({CARBON, 0, 0, 0, 0.1f, 1.7f, 12.0107f, 0.0f, 0.0f});
    for (int i = 0; i < 28; ++i) m.atoms.push_back({HYDROGEN, 0, 0, 0, 0.05f, 1.2f, 1.00794f, 0.0f, 0.0f});
    for (int i = 0; i < 7; ++i) m.atoms.push_back({NITROGEN, 0, 0, 0, -0.3f, 1.55f, 14.0067f, 0.0f, 0.0f});
    for (int i = 0; i < 14; ++i) m.atoms.push_back({OXYGEN, 0, 0, 0, -0.2f, 1.52f, 15.9994f, 0.0f, 0.0f});
    for (int i = 0; i < 2; ++i) m.atoms.push_back({PHOSPHORUS, 0, 0, 0, 0.3f, 1.8f, 30.973762f, 0.0f, 0.0f});

    m.initializeAtomPositions();
    m.calculateBornRadius();
    return m;
}
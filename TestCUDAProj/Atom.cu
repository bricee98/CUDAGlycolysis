#include "Atom.cuh"
#include <cmath>

// Add default constructor implementation
Atom::Atom()
    : type(HYDROGEN), x(0.0f), y(0.0f), z(0.0f), charge(0.0f), mass(1.0f) {
    initializeAtomProperties();
}

// Existing constructor implementation
Atom::Atom(AtomType type, float x, float y, float z, float charge, float mass)
    : type(type), x(x), y(y), z(z), charge(charge), mass(mass) {
    initializeAtomProperties();
}

__host__ __device__ void Atom::setPosition(float newX, float newY, float newZ) {
    x = newX;
    y = newY;
    z = newZ;
}

__host__ __device__ void Atom::setBornRadius(float newBornRadius) {
    bornRadius = newBornRadius;
}

__host__ __device__ void Atom::setInverseBornRadius(float newInverseBornRadius) {
    inverseBornRadius = newInverseBornRadius;
}

void Atom::initializeAtomProperties() {
    switch (type) {
        case HYDROGEN:
            vanDerWaalsRadius = 1.2f;
            epsilon = 0.0157f;
            break;
        case CARBON:
            vanDerWaalsRadius = 1.7f;
            epsilon = 0.1094f;
            break;
        case NITROGEN:
            vanDerWaalsRadius = 1.55f;
            epsilon = 0.0769f;
            break;
        case OXYGEN:
            vanDerWaalsRadius = 1.52f;
            epsilon = 0.2100f;
            break;
        case PHOSPHORUS:
            vanDerWaalsRadius = 1.8f;
            epsilon = 0.2000f;
            break;
        case SULFUR:
            vanDerWaalsRadius = 1.8f;
            epsilon = 0.2500f;
            break;
        default:
            vanDerWaalsRadius = 1.5f;
            epsilon = 0.1000f;
    }
    
    // Initialize Born radius to van der Waals radius as a starting point
    bornRadius = vanDerWaalsRadius;
    inverseBornRadius = 1.0f / bornRadius;
}
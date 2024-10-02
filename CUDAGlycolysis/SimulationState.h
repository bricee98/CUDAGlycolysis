#pragma once
#include "Molecule.cuh"
#include <vector>

struct MoleculeCreationInfo {
    MoleculeType type;
    float x, y, z;
};

struct SimulationState {
    std::vector<Molecule> molecules;
    std::vector<MoleculeCreationInfo> moleculesToCreate;
    std::vector<int> moleculesToDelete;
};
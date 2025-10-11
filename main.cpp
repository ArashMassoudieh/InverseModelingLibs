/*
 * OpenHydroQual - Environmental Modeling Platform
 * Copyright (C) 2025 Arash Massoudieh
 *
 * Sample test for CIndividual class
 */

#include "Individual.h"
#include <iostream>
#include <cassert>
#include <cmath>
#include <cstdlib>
#include <ctime>
#include "GA.h"
#include "MCMC.h"

using namespace std;

/**
 * @brief Test crossover operation between two individuals
 *
 * This test verifies that:
 * 1. Crossover produces two valid offspring
 * 2. Offspring parameters are within valid ranges
 * 3. Offspring inherit characteristics from both parents
 * 4. Parent tracking is maintained
 */
void test_crossover_creates_valid_offspring()
{
    cout << "Running test_crossover_creates_valid_offspring... ";

    // Seed random number generator for reproducibility
    srand(12345);

    // Create two parent individuals with 5 parameters each
    const int nParams = 5;
    CIndividual parent1(nParams);
    CIndividual parent2(nParams);

    // Set up parameter ranges and precision for both parents
    for (int i = 0; i < nParams; i++)
    {
        parent1.minrange[i] = 0.0;
        parent1.maxrange[i] = 10.0;
        parent1.precision[i] = 3;

        parent2.minrange[i] = 0.0;
        parent2.maxrange[i] = 10.0;
        parent2.precision[i] = 3;
    }

    // Initialize parent1 with low values
    for (int i = 0; i < nParams; i++)
    {
        parent1.x[i] = 2.0;  // All parameters = 2.0
    }

    // Initialize parent2 with high values
    for (int i = 0; i < nParams; i++)
    {
        parent2.x[i] = 8.0;  // All parameters = 8.0
    }

    // Create offspring
    CIndividual offspring1(nParams);
    CIndividual offspring2(nParams);

    // Copy ranges to offspring
    for (int i = 0; i < nParams; i++)
    {
        offspring1.minrange[i] = parent1.minrange[i];
        offspring1.maxrange[i] = parent1.maxrange[i];
        offspring1.precision[i] = parent1.precision[i];

        offspring2.minrange[i] = parent2.minrange[i];
        offspring2.maxrange[i] = parent2.maxrange[i];
        offspring2.precision[i] = parent2.precision[i];
    }

    // Perform crossover
    cross(parent1, parent2, offspring1, offspring2);

    // Verification 1: Check that offspring parameters are within valid ranges
    for (int i = 0; i < nParams; i++)
    {
        assert(offspring1.x[i] >= offspring1.minrange[i] &&
               offspring1.x[i] <= offspring1.maxrange[i]);
        assert(offspring2.x[i] >= offspring2.minrange[i] &&
               offspring2.x[i] <= offspring2.maxrange[i]);
    }

    // Verification 2: Check that at least some parameters have changed
    // (crossover should mix parent genes)
    bool offspring1_different_from_parent1 = false;
    bool offspring2_different_from_parent2 = false;

    for (int i = 0; i < nParams; i++)
    {
        if (fabs(offspring1.x[i] - parent1.x[i]) > 0.001)
            offspring1_different_from_parent1 = true;
        if (fabs(offspring2.x[i] - parent2.x[i]) > 0.001)
            offspring2_different_from_parent2 = true;
    }

    // At least one offspring should differ from its corresponding parent
    assert(offspring1_different_from_parent1 || offspring2_different_from_parent2);

    // Verification 3: Offspring values should be between parent values
    // (or at least influenced by both parents)
    int params_in_parent_range = 0;
    for (int i = 0; i < nParams; i++)
    {
        double min_parent = min(parent1.x[i], parent2.x[i]);
        double max_parent = max(parent1.x[i], parent2.x[i]);

        // Due to binary encoding, offspring might be slightly outside parent range
        // but should be close
        if (offspring1.x[i] >= min_parent - 0.5 &&
            offspring1.x[i] <= max_parent + 0.5)
            params_in_parent_range++;
        if (offspring2.x[i] >= min_parent - 0.5 &&
            offspring2.x[i] <= max_parent + 0.5)
            params_in_parent_range++;
    }

    // Most offspring parameters should be influenced by parent values
    assert(params_in_parent_range > nParams);  // More than half

    // Verification 4: Test parent tracking
    offspring1.SetParents(0, 1);
    assert(offspring1.GetParents().size() == 2);
    assert(offspring1.GetParents()[0] == 0);
    assert(offspring1.GetParents()[1] == 1);

    offspring2.SetParents(5);  // Clone parent
    assert(offspring2.GetParents().size() == 2);
    assert(offspring2.GetParents()[0] == 5);
    assert(offspring2.GetParents()[1] == 5);

    // Print some details for manual inspection
    cout << "PASSED" << endl;
    cout << "  Parent1 sample parameters: ";
    for (int i = 0; i < min(3, nParams); i++)
        cout << parent1.x[i] << " ";
    cout << endl;

    cout << "  Parent2 sample parameters: ";
    for (int i = 0; i < min(3, nParams); i++)
        cout << parent2.x[i] << " ";
    cout << endl;

    cout << "  Offspring1 sample parameters: ";
    for (int i = 0; i < min(3, nParams); i++)
        cout << offspring1.x[i] << " ";
    cout << endl;

    cout << "  Offspring2 sample parameters: ";
    for (int i = 0; i < min(3, nParams); i++)
        cout << offspring2.x[i] << " ";
    cout << endl;

    cout << "  All offspring parameters within valid ranges: YES" << endl;
    cout << "  Offspring show genetic mixing: YES" << endl;
}

int main()
{
    cout << "================================" << endl;
    cout << "CIndividual Test Suite" << endl;
    cout << "================================" << endl << endl;

    try
    {
        test_crossover_creates_valid_offspring();

        cout << endl << "================================" << endl;
        cout << "All tests PASSED!" << endl;
        cout << "================================" << endl;
        return 0;
    }
    catch (const exception& e)
    {
        cout << endl << "================================" << endl;
        cout << "Test FAILED: " << e.what() << endl;
        cout << "================================" << endl;
        return 1;
    }
}

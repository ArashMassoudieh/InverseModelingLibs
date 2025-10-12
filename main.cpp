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
#include "polynomialmodel.h"
#include <iostream>

using namespace std;

int main()
{
    // Create synthetic data: y = 2 + 3x - 0.5x^2
    std::vector<double> x_data;
    std::vector<double> y_data;

    for (double x = 0.0; x <= 10.0; x += 0.5) {
        x_data.push_back(x);
        double y = 2.0 + 3.0*x - 0.5*x*x;  // True polynomial
        y_data.push_back(y + 0.1);  // Add small noise
    }

    // Create polynomial model (degree 2 = quadratic)
    PolynomialModel model(2, x_data, y_data);

    // Create GA optimizer
    CGA<PolynomialModel> ga(&model);

    // Configure GA parameters
    ga.SetProperty("maxpop", "50");           // Population size
    ga.SetProperty("ngen", "100");            // Number of generations
    ga.SetProperty("pcross", "0.9");          // Crossover probability
    ga.SetProperty("pmute", "0.02");          // Mutation probability
    ga.SetProperty("outputfile", "ga_results.txt");

    // Run optimization
    std::cout << "Starting GA optimization..." << std::endl;
    int bestIndex = ga.optimize();

    // Get results
    const std::vector<double>& finalParams = ga.getFinalParams();
    double bestFitness = ga.getMaxFitness();

    // Print results
    std::cout << "\nOptimization complete!" << std::endl;
    std::cout << "Best fitness (SSE): " << bestFitness << std::endl;
    std::cout << "Optimized parameters:" << std::endl;

    const std::vector<std::string>& paramNames = ga.getParamNames();
    for (size_t i = 0; i < finalParams.size(); ++i) {
        std::cout << "  " << paramNames[i] << " = " << finalParams[i] << std::endl;
    }

    std::cout << "\nTrue values: a0=2.0, a1=3.0, a2=-0.5" << std::endl;

    return 0;
}

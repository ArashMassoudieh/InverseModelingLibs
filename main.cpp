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
#include "levenbergmarquardt.h"
#include "MCMC.h"
#include "NormalDistributionModel.h"

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

    // Create LM optimizer
    LevenbergMarquardt<PolynomialModel> lm(&model);

    // Configure LM parameters (optional - these are defaults)
    lm.GetParameters().maxIterations = 100;
    lm.GetParameters().tolerance = 1e-6;
    lm.GetParameters().paramTolerance = 1e-6;
    lm.GetParameters().objectiveTolerance = 1e-6;
    lm.GetParameters().lambda = 0.01;
    lm.GetParameters().verbose = true;

    // Alternatively, use SetParameter method:
    // lm.SetParameter("max_iterations", "100");
    // lm.SetParameter("tolerance", "1e-6");
    // lm.SetParameter("verbose", "true");

    // Run optimization
    std::cout << "Starting Levenberg-Marquardt optimization..." << std::endl;
    LMResult result = lm.Optimize();

    // Print results
    std::cout << "\nOptimization complete!" << std::endl;
    std::cout << "Converged: " << (result.converged ? "Yes" : "No") << std::endl;
    std::cout << "Message: " << result.message << std::endl;
    std::cout << "Iterations: " << result.iterations << std::endl;
    std::cout << "Final SSE: " << result.objectiveValue << std::endl;

    std::cout << "\nOptimized parameters:" << std::endl;
    for (size_t i = 0; i < result.parameters.size(); ++i) {
        std::cout << "  " << model.Parameters()[i]->GetName()
        << " = " << result.parameters[i]
        << " ± " << result.standardErrors[i] << std::endl;
    }

    std::cout << "\nTrue values: a0=2.0, a1=3.0, a2=-0.5" << std::endl;

    // Optional: Print correlation matrix
    if (result.covariance.getnumrows() > 0) {
        std::cout << "\nParameter covariance matrix:" << std::endl;
        for (int i = 0; i < result.covariance.getnumrows(); ++i) {
            for (int j = 0; j < result.covariance.getnumcols(); ++j) {
                std::cout << std::setw(12) << result.covariance(i, j) << " ";
            }
            std::cout << std::endl;
        }
    }

    // Create MCMC optimizer
    CMCMC<PolynomialModel> mcmc(&model);

    // Configure MCMC parameters using SetProperty
    mcmc.SetProperty("number_of_samples", "50000");
    mcmc.SetProperty("number_of_chains", "4");
    mcmc.SetProperty("number_of_burnout_samples", "10000");
    mcmc.SetProperty("record_interval", "10");
    mcmc.SetProperty("initial_perturbation_factor", "1.0");
    mcmc.SetProperty("perturbation_change_scale", "0.75");
    mcmc.SetProperty("acceptance_rate", "0.234");
    mcmc.SetProperty("number_of_post_estimate_realizations", "100");
    mcmc.SetProperty("number_of_threads", "4");
    mcmc.SetProperty("samples_filename", "mcmc_results.txt");

    // Alternative: Direct access to settings struct
    // mcmc.settings.total_number_of_samples = 10000;
    // mcmc.settings.number_of_chains = 4;
    // mcmc.settings.burnout_samples = 2000;
    // mcmc.settings.save_interval = 10;
    // mcmc.settings.initial_perturbation_factor = 1.0;
    // mcmc.settings.perturbation_change_scale = 0.75;
    // mcmc.settings.acceptance_rate = 0.234;
    // mcmc.settings.number_of_post_estimate_realizations = 100;
    // mcmc.settings.numberOfThreads = 4;

    // Set output file paths
    mcmc.fileInformation.outputpath = "./output/";
    mcmc.fileInformation.outputfilename = "./output/mcmc_samples.txt";
    mcmc.fileInformation.detailfilename = "./output/mcmc_details.txt";

    // Set percentiles for output (optional)
    mcmc.outputPercentiles = {0.025, 0.5, 0.975};  // 95% credible interval + median

    // Run MCMC sampling
    std::cout << "Starting MCMC sampling..." << std::endl;
    mcmc.Perform();

    // Access results from files or from the mcmc object
    const auto& samples = mcmc.GetParameterSamples();
    const auto& logPosteriors = mcmc.GetLogPosterior();

    // Print summary statistics
    std::cout << "\nMCMC sampling complete!" << std::endl;
    std::cout << "Total samples: " << mcmc.GetSettings().total_number_of_samples << std::endl;
    std::cout << "Effective samples: "
              << (mcmc.GetSettings().total_number_of_samples - mcmc.GetSettings().burnout_samples)
              << std::endl;
    std::cout << "Final acceptance rate: " << mcmc.GetAcceptanceRate() * 100 << "%" << std::endl;

    // Calculate posterior means and credible intervals manually from samples
    std::cout << "\nPosterior statistics:" << std::endl;
    std::cout << std::setw(10) << "Parameter"
              << std::setw(15) << "Mean"
              << std::setw(15) << "2.5%"
              << std::setw(15) << "97.5%"
              << std::endl;
    std::cout << std::string(55, '-') << std::endl;

    for (size_t i = 0; i < mcmc.GetSettings().number_of_parameters; ++i) {
        // Calculate statistics from samples (skipping burnout)
        std::vector<double> paramSamples;
        for (size_t j = mcmc.GetSettings().burnout_samples; j < samples.size(); ++j) {
            paramSamples.push_back(samples[j][i]);
        }

        // Sort for percentiles
        std::sort(paramSamples.begin(), paramSamples.end());

        // Calculate mean
        double mean = std::accumulate(paramSamples.begin(), paramSamples.end(), 0.0)
                      / paramSamples.size();

        // Calculate percentiles
        size_t idx_025 = static_cast<size_t>(0.025 * paramSamples.size());
        size_t idx_975 = static_cast<size_t>(0.975 * paramSamples.size());

        std::cout << std::setw(10) << model.Parameters()[i]->GetName()
                  << std::setw(15) << std::fixed << std::setprecision(4) << mean
                  << std::setw(15) << paramSamples[idx_025]
                  << std::setw(15) << paramSamples[idx_975]
                  << std::endl;
    }

    std::cout << "\nTrue values: a0=2.0, a1=3.0, a2=-0.5" << std::endl;

    // Results are also written to files:
    std::cout << "\nOutput files:" << std::endl;
    std::cout << "  Samples: " << mcmc.fileInformation.outputfilename << std::endl;
    std::cout << "  Details: " << mcmc.fileInformation.detailfilename << std::endl;
    std::cout << "  Posterior distributions: " << mcmc.fileInformation.outputpath
              << "Posterior_Distributions.txt" << std::endl;
    std::cout << "  Posterior percentiles: " << mcmc.fileInformation.outputpath
              << "Posterior_Percentiles.txt" << std::endl;

    // ========================================================================
    // Calculate and Display Parameter Correlation Matrix
    // ========================================================================

    std::cout << "\n========================================" << std::endl;
    std::cout << "PARAMETER CORRELATION ANALYSIS" << std::endl;
    std::cout << "========================================\n" << std::endl;

    // Calculate correlation matrix (skip burnout samples)
    CMatrix_arma correlationMatrix = mcmc.CalculateParameterCorrelation(mcmc.GetSettings().burnout_samples);

    // Display correlation matrix using toString
    std::cout << "Parameter Correlation Matrix:" << std::endl;
    std::cout << std::string(80, '-') << std::endl;
    std::vector<std::string> matrixLines = correlationMatrix.toString("", paramNames, paramNames);
    for (const auto& line : matrixLines)
    {
        std::cout << line << std::endl;
    }
    std::cout << std::string(80, '-') << std::endl;

    // Highlight high correlations (absolute value > 0.7)
    std::cout << "\nHigh Correlations (|r| > 0.7):" << std::endl;
    bool foundHighCorr = false;
    for (size_t i = 0; i < mcmc.GetSettings().number_of_parameters; ++i)
    {
        for (size_t j = i + 1; j < mcmc.GetSettings().number_of_parameters; ++j)
        {
            double corr = correlationMatrix(i, j);
            if (std::abs(corr) > 0.7)
            {
                std::cout << "  " << std::setw(10) << std::left << model.Parameters()[i]->GetName()
                << " <-> "
                << std::setw(10) << std::left << model.Parameters()[j]->GetName()
                << ": " << std::setw(7) << std::right << std::fixed << std::setprecision(4) << corr;

                // Add interpretation
                if (corr > 0.9)
                    std::cout << "  (very strong positive)";
                else if (corr > 0.7)
                    std::cout << "  (strong positive)";
                else if (corr < -0.9)
                    std::cout << "  (very strong negative)";
                else if (corr < -0.7)
                    std::cout << "  (strong negative)";

                std::cout << std::endl;
                foundHighCorr = true;
            }
        }
    }

    if (!foundHighCorr)
    {
        std::cout << "  None found - parameters are well-separated" << std::endl;
    }

    // Highlight moderate correlations (0.5 < |r| < 0.7)
    std::cout << "\nModerate Correlations (0.5 < |r| < 0.7):" << std::endl;
    bool foundModerateCorr = false;
    for (size_t i = 0; i < mcmc.GetSettings().number_of_parameters; ++i)
    {
        for (size_t j = i + 1; j < mcmc.GetSettings().number_of_parameters; ++j)
        {
            double corr = correlationMatrix(i, j);
            if (std::abs(corr) > 0.5 && std::abs(corr) <= 0.7)
            {
                std::cout << "  " << std::setw(10) << std::left << model.Parameters()[i]->GetName()
                << " <-> "
                << std::setw(10) << std::left << model.Parameters()[j]->GetName()
                << ": " << std::setw(7) << std::right << std::fixed << std::setprecision(4) << corr
                << std::endl;
                foundModerateCorr = true;
            }
        }
    }

    if (!foundModerateCorr)
    {
        std::cout << "  None found" << std::endl;
    }

    std::cout << "\n========================================\n" << std::endl;

    // Write correlation matrix to file using built-in writetofile
    std::string corrFileRaw = mcmc.fileInformation.outputpath + "Parameter_Correlations.txt";
    try
    {
        correlationMatrix.writetofile(corrFileRaw);
        std::cout << "✓ Correlation matrix written to: " << corrFileRaw << std::endl;
    }
    catch (const std::exception& e)
    {
        std::cerr << "✗ Error writing correlation matrix: " << e.what() << std::endl;
    }

    // Write a summary report with interpretation
    std::string corrSummary = mcmc.fileInformation.outputpath + "Correlation_Summary.txt";
    try
    {
        std::ofstream summaryFile(corrSummary);
        if (!summaryFile.is_open())
        {
            throw std::runtime_error("Could not open file: " + corrSummary);
        }

        summaryFile << "PARAMETER CORRELATION ANALYSIS SUMMARY\n";
        summaryFile << "======================================\n\n";
        summaryFile << "MCMC Run Information:\n";
        summaryFile << "  Total samples: " << mcmc.GetSettings().total_number_of_samples << "\n";
        summaryFile << "  Burn-in samples: " << mcmc.GetSettings().burnout_samples << "\n";
        summaryFile << "  Effective samples: " << (mcmc.GetSettings().total_number_of_samples - mcmc.GetSettings().burnout_samples) << "\n";
        summaryFile << "  Number of parameters: " << mcmc.GetSettings().number_of_parameters << "\n\n";

        summaryFile << "CORRELATION MATRIX:\n";
        summaryFile << "-------------------\n";
        std::vector<std::string> matrixLines = correlationMatrix.toString("", paramNames, paramNames);
        for (const auto& line : matrixLines)
        {
            summaryFile << line << "\n";
        }
        summaryFile << "\n";

        summaryFile << "HIGH CORRELATIONS (|r| > 0.7):\n";
        summaryFile << "------------------------------\n";
        bool anyHigh = false;
        for (size_t i = 0; i < mcmc.GetSettings().number_of_parameters; ++i)
        {
            for (size_t j = i + 1; j < mcmc.GetSettings().number_of_parameters; ++j)
            {
                double corr = correlationMatrix(i, j);
                if (std::abs(corr) > 0.7)
                {
                    summaryFile << model.Parameters()[i]->GetName() << " <-> "
                                << model.Parameters()[j]->GetName()
                                << ": " << std::fixed << std::setprecision(4) << corr << "\n";
                    anyHigh = true;
                }
            }
        }
        if (!anyHigh) summaryFile << "None found\n";

        summaryFile << "\nMODERATE CORRELATIONS (0.5 < |r| < 0.7):\n";
        summaryFile << "----------------------------------------\n";
        bool anyModerate = false;
        for (size_t i = 0; i < mcmc.GetSettings().number_of_parameters; ++i)
        {
            for (size_t j = i + 1; j < mcmc.GetSettings().number_of_parameters; ++j)
            {
                double corr = correlationMatrix(i, j);
                if (std::abs(corr) > 0.5 && std::abs(corr) <= 0.7)
                {
                    summaryFile << model.Parameters()[i]->GetName() << " <-> "
                                << model.Parameters()[j]->GetName()
                                << ": " << std::fixed << std::setprecision(4) << corr << "\n";
                    anyModerate = true;
                }
            }
        }
        if (!anyModerate) summaryFile << "None found\n";

        summaryFile << "\nINTERPRETATION:\n";
        summaryFile << "---------------\n";
        summaryFile << "High correlation (|r| > 0.7) means parameters move together and are difficult\n";
        summaryFile << "to identify separately from the data. This can lead to:\n";
        summaryFile << "  - Wide credible intervals\n";
        summaryFile << "  - Slow MCMC convergence\n";
        summaryFile << "  - Difficulty in parameter interpretation\n\n";
        summaryFile << "Consider:\n";
        summaryFile << "  - Reparameterizing the model\n";
        summaryFile << "  - Adding more informative data\n";
        summaryFile << "  - Using stronger priors on correlated parameters\n";

        summaryFile.close();
        std::cout << "✓ Correlation summary written to: " << corrSummary << std::endl;
    }
    catch (const std::exception& e)
    {
        std::cerr << "✗ Error writing correlation summary: " << e.what() << std::endl;
    }

    std::cout << "\n========================================\n" << std::endl;

    std::cout << "========================================" << std::endl;
    std::cout << "MCMC TEST: Normal Distribution Parameters" << std::endl;
    std::cout << "========================================" << std::endl;

    // Create model
    NormalDistributionModel modelNormal;

    // Generate synthetic data with known parameters
    double TRUE_MEAN = 5.0;
    double TRUE_STD = 2.0;
    int N_DATA_POINTS = 100;  // Number of data points to generate

    modelNormal.GenerateSyntheticData(TRUE_MEAN, TRUE_STD, N_DATA_POINTS);

    // Create MCMC optimizer
    CMCMC<NormalDistributionModel> mcmcNormal(&modelNormal);

    // Configure MCMC parameters
    mcmcNormal.SetProperty("number_of_samples", "20000");
    mcmcNormal.SetProperty("number_of_chains", "4");
    mcmcNormal.SetProperty("number_of_burnout_samples", "5000");
    mcmcNormal.SetProperty("record_interval", "10");
    mcmcNormal.SetProperty("perturbation_change_scale", "0.7");
    mcmcNormal.SetProperty("acceptance_rate", "0.234");
    mcmcNormal.SetProperty("number_of_threads", "4");

    // Set output paths
    mcmcNormal.fileInformation.outputpath = "./output/";
    mcmcNormal.fileInformation.outputfilename = "./output/normal_mcmc_samples.txt";
    mcmcNormal.fileInformation.detailfilename = "./output/normal_mcmc_details.txt";

    // Run MCMC sampling
    std::cout << "\nStarting MCMC sampling..." << std::endl;
    mcmcNormal.Perform();

    // Get results
    const auto& samplesNormal = mcmcNormal.GetParameterSamples();
    const auto& logPosteriorsNormal = mcmcNormal.GetLogPosterior();

    // Calculate posterior statistics manually from samples
    std::cout << "\n========================================" << std::endl;
    std::cout << "RESULTS COMPARISON" << std::endl;
    std::cout << "========================================" << std::endl;

    // Calculate means and credible intervals
    int burnin = mcmcNormal.GetSettings().burnout_samples;
    int n_effective = samplesNormal.size() - burnin;

    for (size_t param_idx = 0; param_idx < 2; ++param_idx)
    {
        // Collect post-burnin samples for this parameter
        std::vector<double> param_samples;
        param_samples.reserve(n_effective);
        for (size_t i = burnin; i < samplesNormal.size(); ++i)  // FIXED: was samples
        {
            param_samples.push_back(samplesNormal[i][param_idx]);  // FIXED: was samples
        }

        // Sort for percentiles
        std::sort(param_samples.begin(), param_samples.end());

        // Calculate statistics
        double mean = std::accumulate(param_samples.begin(), param_samples.end(), 0.0)
                      / param_samples.size();

        size_t idx_025 = static_cast<size_t>(0.025 * param_samples.size());
        size_t idx_500 = static_cast<size_t>(0.500 * param_samples.size());
        size_t idx_975 = static_cast<size_t>(0.975 * param_samples.size());

        std::string param_name = modelNormal.Parameters()[param_idx]->GetName();  // FIXED: was model
        double true_value = (param_idx == 0) ? TRUE_MEAN : TRUE_STD;

        std::cout << "\n" << param_name << " (true value = "
                  << std::fixed << std::setprecision(4) << true_value << "):" << std::endl;
        std::cout << "  Posterior mean:      " << std::setw(10) << mean << std::endl;
        std::cout << "  Posterior median:    " << std::setw(10) << param_samples[idx_500] << std::endl;
        std::cout << "  95% Credible Int:   [" << std::setw(10) << param_samples[idx_025]
                  << ", " << std::setw(10) << param_samples[idx_975] << "]" << std::endl;
        std::cout << "  Interval width:      " << std::setw(10)
                  << (param_samples[idx_975] - param_samples[idx_025]) << std::endl;

        // Check if true value is in credible interval
        bool in_interval = (true_value >= param_samples[idx_025]) &&
                           (true_value <= param_samples[idx_975]);
        std::cout << "  True value in CI:    " << (in_interval ? "YES ✓" : "NO ✗") << std::endl;

        // Calculate error
        double error = std::abs(mean - true_value);
        double rel_error = error / true_value * 100.0;
        std::cout << "  Absolute error:      " << std::setw(10) << error << std::endl;
        std::cout << "  Relative error:      " << std::setw(10) << std::setprecision(2)
                  << rel_error << "%" << std::endl;
    }

    // Compare with analytical solution
    std::cout << "\n========================================" << std::endl;
    std::cout << "ANALYTICAL VS MCMC COMPARISON" << std::endl;
    std::cout << "========================================" << std::endl;

    double sample_mean = modelNormal.CalculateSampleMean();
    double sample_std = modelNormal.CalculateSampleStd();

    std::cout << "\nMaximum Likelihood Estimates (analytical):" << std::endl;
    std::cout << "  Mean (μ):    " << std::fixed << std::setprecision(6) << sample_mean << std::endl;
    std::cout << "  Std Dev (σ): " << sample_std << std::endl;

    std::cout << "\nMCMC Posterior Means:" << std::endl;
    std::vector<double> mcmc_means(2);
    for (size_t i = 0; i < 2; ++i)
    {
        double sum = 0.0;
        for (size_t j = burnin; j < samplesNormal.size(); ++j)  // FIXED: was samples
        {
            sum += samplesNormal[j][i];  // FIXED: was samples
        }
        mcmc_means[i] = sum / (samplesNormal.size() - burnin);  // FIXED: was samples
    }
    std::cout << "  Mean (μ):    " << mcmc_means[0] << std::endl;
    std::cout << "  Std Dev (σ): " << mcmc_means[1] << std::endl;

    std::cout << "\nDifference (MCMC - Analytical):" << std::endl;
    std::cout << "  Mean (μ):    " << (mcmc_means[0] - sample_mean) << std::endl;
    std::cout << "  Std Dev (σ): " << (mcmc_means[1] - sample_std) << std::endl;

    std::cout << "\nFinal acceptance rate: "
              << (mcmcNormal.GetAcceptanceRate() * 100.0) << "%" << std::endl;  // FIXED: was mcmc

    // ========================================================================
    // Calculate and Display Parameter Correlation Matrix
    // ========================================================================

    std::cout << "\n========================================" << std::endl;
    std::cout << "PARAMETER CORRELATION ANALYSIS" << std::endl;
    std::cout << "========================================\n" << std::endl;

    CMatrix_arma correlationMatrixNormal = mcmcNormal.CalculateParameterCorrelation(burnin);

    // Prepare parameter names
    std::vector<std::string> paramNamesNormal = {"mu", "sigma"};

    // Display correlation matrix
    std::cout << "Correlation Matrix:" << std::endl;
    std::vector<std::string> matrixLinesNormal = correlationMatrixNormal.toString("", paramNamesNormal, paramNamesNormal);  // FIXED: was paramNames twice, and was matrixLines in loop
    for (const auto& line : matrixLinesNormal)  // FIXED: was matrixLines
    {
        std::cout << line << std::endl;
    }

    std::cout << "\nExpected: Low correlation (~0) between mean and std deviation" << std::endl;
    std::cout << "Actual correlation: " << std::fixed << std::setprecision(4)
              << correlationMatrixNormal(0, 1) << std::endl;  // FIXED: was correlationMatrix

    if (std::abs(correlationMatrixNormal(0, 1)) < 0.3)  // FIXED: was correlationMatrix
    {
        std::cout << "✓ Parameters are nearly independent (as expected)" << std::endl;
    }
    else
    {
        std::cout << "⚠ Unexpected correlation detected" << std::endl;
    }

    // Write correlation to file
    std::string corrFile = "./output/normal_correlations.txt";
    correlationMatrixNormal.writetofile(corrFile);
    std::cout << "\n✓ Correlation matrix written to: " << corrFile << std::endl;

    std::cout << "\n========================================" << std::endl;
    std::cout << "MCMC TEST COMPLETE" << std::endl;
    std::cout << "========================================" << std::endl;

    return 0;


}

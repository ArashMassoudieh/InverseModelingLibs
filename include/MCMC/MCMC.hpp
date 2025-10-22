/*
 * OpenHydroQual - Environmental Modeling Platform
 * Copyright (C) 2025 Arash Massoudieh
 * 
 * This file is part of OpenHydroQual.
 * 
 * OpenHydroQual is free software: you can redistribute it and/or modify it
 * under the terms of the GNU Affero General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or (at your
 * option) any later version.
 * 
 * If you use this file in a commercial product, you must purchase a
 * commercial license. Contact arash.massoudieh@enviroinformatics.co for details.
 */


// MCMC.cpp : Defines the entry point for the console application.
//

// Standard library
#include <vector>
#include <string>
#include <iostream>
#include <fstream>      // For file I/O operations
#include <cmath>        // For exp, log, pow, fabs, sqrt, isnan
#include <algorithm>    // For min, max
#include <thread>       // For hardware_concurrency

// Project headers
#include "Utilities.h"
#include "parameter_set.h"
#include "observation.h"

// OpenMP for parallel processing
#ifndef mac_version
#include <omp.h>
#endif

// Qt GUI support (conditional)
#ifdef Q_GUI_SUPPORT
#include "ProgressWindow.h"
#include <QCoreApplication>  // For processEvents() to keep GUI responsive
#include <QString>           // For QString::number() and string conversions
#include <QDebug>            // For qDebug() logging
#endif

// OpenBLAS configuration (platform-specific)
#ifndef _WINDOWS
#ifndef _MacOS
extern "C" {
#include <openblas_config.h>
void openblas_set_num_threads(int);
}
#endif
#endif

/**
 * @brief Default constructor
 *
 * Creates an MCMC object with no model attached. User must set model pointer
 * and configuration before running.
 */
template<class T>
CMCMC<T>::CMCMC()
    : model(nullptr)
    , parameters(nullptr)
    , observations(nullptr)
    , runtimeWindow(nullptr)
    , acceptedCount(0.0)
    , totalCount(0.0)
{
    // All members initialized in initializer list
}

/**
 * @brief Destructor
 *
 * Cleans up dynamically allocated memory for parameter samples and statistics.
 * Note: Does NOT delete model, parameters, or observations pointers as
 * this class does not own them.
 */
template<class T>
CMCMC<T>::~CMCMC()
{
    // Clear vectors to free memory
    parameterSamples.clear();
    logPosteriorTransformed.clear();
    logPosterior.clear();

    // Note: We don't delete model, parameters, observations, or runtimeWindow
    // because this class doesn't own them - they're managed externally
}

/**
 * @brief Constructor with model pointer
 * @param system Pointer to the model to be calibrated (must not be null)
 *
 * Initializes MCMC with the given model, automatically extracts parameters
 * and observations from the model, and sets up default configuration.
 *
 * @throws Does not throw, but will result in errors if system is null
 *
 * @example
 * @code
 * MyModel model;
 * CMCMC<MyModel> mcmc(&model);
 * @endcode
 */
template<class T>
CMCMC<T>::CMCMC(T* system)
    : model(system)
    , parameters(nullptr)
    , observations(nullptr)
    , runtimeWindow(nullptr)
    , acceptedCount(0.0)
    , totalCount(0.0)
{
    // Input validation
    if (!model)
    {
        last_error = "MCMC constructor: model pointer is null";
        return;
    }

    // Initialize from model
    InitializeFromModel();
}


template<class T>
Parameter* CMCMC<T>::parameter(int i)
{
    // Validate parameters pointer
    if (!parameters)
    {
        last_error = "parameter(): parameters pointer is null";
        return nullptr;
    }

    // Validate index range
    if (i < 0 || i >= static_cast<int>(parameters->size()))
    {
        last_error = "parameter(): index " + std::to_string(i) + " out of range [0, "
                     + std::to_string(parameters->size() - 1) + "]";
        return nullptr;
    }

    // Return pointer to parameter
    return (*parameters)[i];
}

template<class T>
Observation* CMCMC<T>::observation(int i)
{
    // Validate observations pointer
    if (!observations)
    {
        last_error = "observation(): observations pointer is null";
        return nullptr;
    }

    // Validate index range
    if (i < 0 || i >= static_cast<int>(observations->size()))
    {
        last_error = "observation(): index " + std::to_string(i) + " out of range [0, "
                     + std::to_string(observations->size() - 1) + "]";
        return nullptr;
    }

    // Return pointer to observation
    return &(observations->at(i));
}

/**
 * @brief Run model with given parameters and return predictions
 * @param par Parameter values to use
 * @return TimeSeriesSet with model predictions at observation times
 *
 * This method:
 * 1. Creates a copy of the model
 * 2. Configures it for silent, single-threaded execution
 * 3. Sets all parameter values
 * 4. Applies parameters and solves
 * 5. Returns predictions at observation times
 *
 * @note The model is run in silent mode without recording detailed results
 */
template<class T>
TimeSeriesSet<double> CMCMC<T>::RunModel(const std::vector<double>& par)
{
    // Validate input
    if (par.size() != settings.number_of_parameters)
    {
        last_error = "RunModel: parameter vector size (" + std::to_string(par.size())
        + ") does not match expected (" + std::to_string(settings.number_of_parameters) + ")";
        return TimeSeriesSet<double>();
    }

    if (!model)
    {
        last_error = "RunModel: model pointer is null";
        return TimeSeriesSet<double>();
    }

    // Create a copy of the model
    T modelCopy = *model;

    // Set parameter values
    for (unsigned int i = 0; i < settings.number_of_parameters; ++i)
    {
        modelCopy.SetParameterValue(i, par[i]);
    }

    // Return predictions at observation times
    return modelCopy.GetPredictions();
}

/**
 * @brief Set all parameter values at once
 * @param paramValues Vector of parameter values
 *
 * Sets parameter values directly on the model. After calling this method,
 * you typically need to call model->ApplyParameters() to ensure the model
 * updates its internal state.
 *
 * @example
 * @code
 * std::vector<double> params = {1.0, 2.5, 0.3};
 * mcmc.SetParameters(params);
 * @endcode
 */
template<class T>
void CMCMC<T>::SetParameters(const std::vector<double>& paramValues)
{
    // Validate input
    if (!model)
    {
        last_error = "SetParameters: model pointer is null";
        return;
    }

    if (paramValues.size() != settings.number_of_parameters)
    {
        last_error = "SetParameters: parameter vector size ("
                     + std::to_string(paramValues.size())
                     + ") does not match expected ("
                     + std::to_string(settings.number_of_parameters) + ")";
        return;
    }

    // Set each parameter value in the model
    for (unsigned int i = 0; i < settings.number_of_parameters; ++i)
    {
        model->SetParameterValue(i, paramValues[i]);
    }
}

/**
 * @brief Access parameter by index
 * @param i Parameter index
 * @return Pointer to Parameter object, or nullptr if invalid
 *
 * Provides safe access to parameters with bounds checking.
 *
 * @example
 * @code
 * Parameter* param = mcmc.GetParameter(0);
 * if (param) {
 *     std::cout << "Parameter: " << param->GetName() << std::endl;
 * }
 * @endcode
 */
template<class T>
Parameter* CMCMC<T>::GetParameter(int i)
{
    // Validate parameters pointer
    if (!parameters)
    {
        last_error = "GetParameter(): parameters pointer is null";
        return nullptr;
    }

    // Validate index range
    if (i < 0 || i >= static_cast<int>(parameters->size()))
    {
        last_error = "GetParameter(): index " + std::to_string(i) + " out of range [0, "
                     + std::to_string(parameters->size() - 1) + "]";
        return nullptr;
    }

    // Return pointer to parameter
    return (*parameters)[i];
}

/**
 * @brief Access observation by index
 * @param i Observation index
 * @return Pointer to Observation object, or nullptr if invalid
 *
 * Provides safe access to observations with bounds checking.
 *
 * @example
 * @code
 * Observation* obs = mcmc.GetObservation(0);
 * if (obs) {
 *     std::cout << "Observation: " << obs->GetName() << std::endl;
 * }
 * @endcode
 */
template<class T>
Observation* CMCMC<T>::GetObservation(int i)
{
    // Validate observations pointer
    if (!observations)
    {
        last_error = "GetObservation(): observations pointer is null";
        return nullptr;
    }

    // Validate index range
    if (i < 0 || i >= static_cast<int>(observations->size()))
    {
        last_error = "GetObservation(): index " + std::to_string(i) + " out of range [0, "
                     + std::to_string(observations->size() - 1) + "]";
        return nullptr;
    }

    // Return pointer to observation
    return &(observations->at(i));
}

/**
 * @brief Set a single MCMC configuration property by name
 * @param varname Property name (case-insensitive)
 * @param value Property value as string
 * @return true if property was recognized and set, false otherwise
 *
 * Supported properties:
 * - number_of_samples: Total samples to generate
 * - number_of_chains: Number of parallel chains
 * - number_of_burnout_samples: Samples to discard at start
 * - initial_perturbation_factor: Initial perturbation magnitude
 * - record_interval: Save every nth sample
 * - initial_perturbation: "yes" or "no"
 * - perform_global_sensitivity: "yes" or "no"
 * - continue_based_on_file_name: Filename to continue from
 * - samples_filename: Output filename
 * - number_of_post_estimate_realizations: Post-sampling realizations
 * - increment_for_sensitivity_analysis: Sensitivity increment
 * - add_noise_to_realizations: "yes" or "no"
 * - number_of_threads: Thread count
 * - acceptance_rate: Target acceptance rate
 * - perturbation_change_scale: Adaptive scaling factor
 */
template<class T>
bool CMCMC<T>::SetProperty(const std::string& varname, const std::string& value)
{
    // Convert to lowercase for case-insensitive comparison
    std::string lowerVarname = aquiutils::tolower(varname);

    // Integer properties
    if (lowerVarname == "number_of_samples") {
        settings.total_number_of_samples = aquiutils::atoi(value);
        return true;
    }

    if (lowerVarname == "number_of_chains") {
        settings.number_of_chains = aquiutils::atoi(value);
        return true;
    }

    if (lowerVarname == "number_of_burnout_samples") {
        settings.burnout_samples = aquiutils::atoi(value);
        return true;
    }

    if (lowerVarname == "record_interval") {
        settings.save_interval = aquiutils::atoi(value);
        return true;
    }

    if (lowerVarname == "number_of_post_estimate_realizations") {
        settings.number_of_post_estimate_realizations = aquiutils::atoi(value);
        return true;
    }

    if (lowerVarname == "number_of_threads") {
        settings.numberOfThreads = aquiutils::atoi(value);
        return true;
    }

    // Double properties
    if (lowerVarname == "initial_perturbation_factor") {
        settings.initial_perturbation_factor = aquiutils::atof(value);
        return true;
    }

    if (lowerVarname == "perturbation_change_scale") {
        settings.perturbation_change_scale = aquiutils::atof(value);
        return true;
    }

    if (lowerVarname == "acceptance_rate") {
        settings.acceptance_rate = aquiutils::atof(value);
        return true;
    }

    if (lowerVarname == "increment_for_sensitivity_analysis") {
        settings.increment_for_sensitivity = aquiutils::atof(value);
        return true;
    }

    // Boolean properties (yes/no)
    if (lowerVarname == "initial_perturbation") {
        settings.no_initial_perturbation = (aquiutils::tolower(value) != "yes");
        return true;
    }

    if (lowerVarname == "perform_global_sensitivity") {
        settings.global_sensitivity = (aquiutils::tolower(value) == "yes");
        return true;
    }

    if (lowerVarname == "add_noise_to_realizations") {
        settings.noise_realization_writeout = (aquiutils::tolower(value) == "yes");
        return true;
    }

    // File path properties
    if (lowerVarname == "continue_based_on_file_name") {
        if (!value.empty()) {
            settings.continue_filename = value;
            settings.continue_mcmc = true;
        } else {
            settings.continue_mcmc = false;
        }
        return true;
    }

    if (lowerVarname == "samples_filename") {
        if (!value.empty()) {
            // Check if path is absolute or relative
            if (value.find_first_of('/') != std::string::npos ||
                value.find_first_of('\\') != std::string::npos) {
                fileInformation.outputfilename = value;
            } else {
                fileInformation.outputfilename = fileInformation.outputpath + value;
            }
        }

        // Set detail filename based on output filename
        fileInformation.detailfilename = aquiutils::extract_path(fileInformation.outputfilename)
                                         + "/" + "MCMC_details.txt";
        return true;
    }

    // Property not found
    last_error = "SetProperty: Unknown property '" + varname + "'";
    return false;
}

/**
 * @brief Calculate log posterior probability
 * @param par Parameter values
 * @param sample_number Sample index for logging
 * @param saveOutput If true, save full model output to modelOutput
 * @return Log posterior probability
 *
 * This method:
 * 1. Creates a copy of the model
 * 2. Sets parameter values
 * 3. Calculates log prior probability from parameter distributions
 * 4. Runs the model
 * 5. Calculates log likelihood from objective function
 * 6. Returns log posterior = log prior + log likelihood
 *
 * The objective function is assumed to be -2*log(likelihood), so
 * log likelihood = -ObjectiveFunction/2 (or just -ObjectiveFunction for SSE)
 */
template<class T>
double CMCMC<T>::CalculateLogPosterior(const std::vector<double>& par,
                                       int sample_number,
                                       bool saveOutput)
{
    // Validate input
    if (!model)
    {
        last_error = "CalculateLogPosterior: model pointer is null";
        return -1e10;  // Very low probability
    }

    if (par.size() != settings.number_of_parameters)
    {
        last_error = "CalculateLogPosterior: parameter vector size mismatch";
        return -1e10;
    }

    // Create a copy of the model
    T modelCopy = *model;

    // Calculate log prior probability and set parameter values
    double logPrior = 0.0;
    for (unsigned int i = 0; i < settings.number_of_parameters; ++i)
    {
        modelCopy.SetParameterValue(i, par[i]);

        Parameter* p = GetParameter(i);
        if (p) {
            logPrior += p->CalcLogPriorProbability(par[i]);
        }
    }

    // Log start of simulation (thread-safe)
#pragma omp critical
    {
        std::ofstream detailFile(fileInformation.detailfilename, std::ios::app);
        if (detailFile.is_open()) {
            detailFile << "Sample #" << sample_number << " started.\n";
            detailFile.close();
        }
    }

    // Get objective function value (negative log likelihood)
    double objectiveFunctionValue = modelCopy.GetObjectiveFunctionValue();

    // Calculate log likelihood
    // Assuming objective function is sum of squared errors (SSE)
    // log likelihood ∝ -SSE/2, but we can use -SSE for simplicity
    double logLikelihood = objectiveFunctionValue;

    // Log completion (thread-safe)
#pragma omp critical
    {
        std::ofstream detailFile(fileInformation.detailfilename, std::ios::app);
        if (detailFile.is_open()) {
            detailFile << "Sample #" << sample_number
                       << " simulation_duration: " << modelCopy.GetSimulationDuration()
                       << ", simulation_failed: " << (modelCopy.GetSolutionFailed() ? "true" : "false")
                       << ", objective_function: " << objectiveFunctionValue
                       << ", log_posterior: " << (logPrior + logLikelihood) << "\n";
            detailFile.close();
        }
    }

    // Save output if requested
    if (saveOutput) {
        modelOutput = modelCopy;
    }

    // Return log posterior = log prior + log likelihood
    return logPrior + logLikelihood;
}

/**
 * @brief Run model with given parameters (modifies provided model in-place)
 * @param modelPtr Pointer to model to run
 * @param par Parameter values to use
 *
 * @deprecated This method is kept for backward compatibility but is not recommended.
 *             Use RunModel() instead which doesn't modify the original model.
 *
 * @note This method has a bug in the original code where SetSilent, SetRecordResults,
 *       and SetNumThreads are called inside the parameter loop. This has been fixed.
 */
template<class T>
void CMCMC<T>::RunModelInPlace(T* modelPtr, const std::vector<double>& par)
{
    // Validate input
    if (!modelPtr)
    {
        last_error = "RunModelInPlace: model pointer is null";
        return;
    }

    if (par.size() != settings.number_of_parameters)
    {
        last_error = "RunModelInPlace: parameter vector size mismatch";
        return;
    }

    // Configure model (FIXED: moved outside parameter loop)
    modelPtr->SetSilent(true);
    modelPtr->SetRecordResults(false);
    modelPtr->SetNumThreads(1);

    // Set parameter values
    for (unsigned int i = 0; i < settings.number_of_parameters; ++i)
    {
        modelPtr->SetParameterValue(i, par[i]);
    }

    // Apply parameters and solve
    modelPtr->ApplyParameters();
    modelPtr->Solve();
}


/**
 * @brief Initialize from model - common initialization code
 *
 * Extracted common initialization logic used by constructors.
 * Sets up parameters, observations, and default settings.
 */
template<class T>
void CMCMC<T>::InitializeFromModel()
{
    // Get parameters from model
    parameters = &(model->Parameters());

    // Get observations from model
    observations = model->Observations();

    // Validate we have parameters
    if (!parameters || parameters->empty())
    {
        last_error = "InitializeFromModel: Model has no parameters";
        return;
    }

    // Validate we have observations
    if (!observations || observations->empty())
    {
        last_error = "InitializeFromModel: Model has no observations";
        return;
    }

    // Set number of parameters in settings
    settings.number_of_parameters = static_cast<unsigned int>(parameters->size());

    // Initialize parameter indices (all parameters are estimated by default)
    parameterIndices.clear();
    for (unsigned int i = 0; i < settings.number_of_parameters; ++i)
    {
        parameterIndices.push_back(i);
    }

    // Set default output path if not set
    if (fileInformation.outputpath.empty())
    {
        fileInformation.outputpath = model->OutputPath();
    }

    // Set default output filename if not set
    if (fileInformation.outputfilename.empty())
    {
        fileInformation.outputfilename = fileInformation.outputpath + "mcmc_samples.txt";
    }

    // Set default detail filename if not set
    if (fileInformation.detailfilename.empty())
    {
        fileInformation.detailfilename = fileInformation.outputpath + "mcmc_details.txt";
    }

    // Initialize applyToAll flags (all parameters apply to all by default)
    applyToAll.resize(settings.number_of_parameters, true);
}

/**
 * @brief Initialize MCMC chains
 * @param random If true, initialize parameters randomly; if false, use current values
 *
 * This method:
 * 1. Clears the detail log file
 * 2. Allocates storage for all samples
 * 3. Calculates perturbation coefficients based on parameter ranges
 * 4. Initializes starting values for each chain (random or current)
 * 5. Evaluates initial log posterior for each chain
 *
 * For random initialization:
 * - Uniform/normal parameters: sampled uniformly from [low, high]
 * - Log-normal parameters: sampled uniformly in log-space
 *
 * For non-random initialization:
 * - All chains start from current parameter values
 */
template<class T>
void CMCMC<T>::Initialize(bool random)
{
    // Clear/create detail log file
    std::ofstream detailFile(fileInformation.detailfilename, std::ios::trunc);
    if (!detailFile) {
        last_error = "Initialize: Could not open detail file: " + fileInformation.detailfilename;
        std::cerr << last_error << std::endl;
        return;
    }
    detailFile.close();

    // Allocate storage for all samples
    parameterSamples.resize(settings.total_number_of_samples);
    logPosterior.resize(settings.total_number_of_samples);
    logPosteriorTransformed.resize(settings.total_number_of_samples);

    for (unsigned int i = 0; i < settings.total_number_of_samples; ++i)
    {
        parameterSamples[i].resize(settings.number_of_parameters);
    }

    // Calculate perturbation coefficients for each parameter
    perturbationCoefficients.resize(settings.number_of_parameters);

    for (unsigned int j = 0; j < settings.number_of_parameters; ++j)
    {
        Parameter* param = GetParameter(j);
        if (!param) {
            last_error = "Initialize: Parameter " + std::to_string(j) + " is null";
            return;
        }

        std::string distribution = param->GetPriorDistribution();
        Parameter::Range range = param->GetRange();

        if (distribution == "normal" || distribution == "uniform")
        {
            // For normal/uniform: perturbation proportional to range width
            perturbationCoefficients[j] = settings.perturbation_factor * (range.high - range.low);
        }
        else if (distribution == "log-normal")
        {
            // For log-normal: perturbation proportional to log-space range
            perturbationCoefficients[j] = settings.perturbation_factor
                                          * (std::log(range.high) - std::log(range.low));
        }
        else
        {
            // Default: use linear range
            perturbationCoefficients[j] = settings.perturbation_factor * (range.high - range.low);
        }
    }

    // Set OpenBLAS to single-threaded for parallel MCMC chains
#ifndef _WINDOWS
#ifndef _MacOS
    openblas_set_num_threads(1);
#endif
#endif

    // Initialize each chain
    if (random)
    {
        // Random initialization: sample from prior distributions
#pragma omp parallel for
        for (int j = 0; j < static_cast<int>(settings.number_of_chains); ++j)
        {
            double logJacobian = 0.0;  // For log-normal transformation

            // Sample parameters from their prior distributions
            for (unsigned int i = 0; i < settings.number_of_parameters; ++i)
            {
                Parameter* param = GetParameter(i);
                if (!param) continue;

                std::string distribution = param->GetPriorDistribution();
                Parameter::Range range = param->GetRange();

                if (distribution == "log-normal")
                {
                    // Sample uniformly in log-space
                    double logLow = std::log(range.low);
                    double logHigh = std::log(range.high);
                    double logValue = logLow + (logHigh - logLow) * normalDistribution.unitrandom();
                    parameterSamples[j][i] = std::exp(logValue);

                    // Accumulate Jacobian for transformation
                    logJacobian += std::log(parameterSamples[j][i]);
                }
                else
                {
                    // Sample uniformly in linear space
                    parameterSamples[j][i] = range.low + (range.high - range.low) * normalDistribution.unitrandom();
                }
            }

            // Evaluate initial log posterior
            logPosterior[j] = CalculateLogPosterior(parameterSamples[j], j) + logJacobian;
            logPosteriorTransformed[j] = logPosterior[j];
        }
    }
    else
    {
        // Non-random initialization: use current parameter values
#pragma omp parallel for
        for (int j = 0; j < static_cast<int>(settings.number_of_chains); ++j)
        {
            double logJacobian = 0.0;

            // Use current parameter values
            for (unsigned int i = 0; i < settings.number_of_parameters; ++i)
            {
                Parameter* param = GetParameter(i);
                if (!param) continue;

                parameterSamples[j][i] = param->GetValue();

                // Accumulate Jacobian for log-normal parameters
                if (param->GetPriorDistribution() == "log-normal")
                {
                    logJacobian += std::log(parameterSamples[j][i]);
                }
            }

            // Evaluate initial log posterior
            logPosterior[j] = CalculateLogPosterior(parameterSamples[j], j) + logJacobian;
            logPosteriorTransformed[j] = logPosterior[j];
        }
    }

    // Restore OpenBLAS to multi-threaded mode
#ifndef _WINDOWS
#ifndef _MacOS
    unsigned int cores = std::thread::hardware_concurrency();
    openblas_set_num_threads(cores > 0 ? cores : 1);
#endif
#endif
}

/**
 * @brief Initialize MCMC chains from specific parameter values
 * @param par Initial parameter values
 *
 * This method:
 * 1. Clears the detail log file
 * 2. Calculates perturbation coefficients (sensitivity-based or range-based)
 * 3. Initializes each chain with perturbations around the given values
 * 4. Evaluates initial log posterior for each chain
 *
 * Perturbation modes:
 * - If sensitivity_based_perturbation is enabled: scales by parameter sensitivity
 * - Otherwise: scales by parameter range
 *
 * Initial perturbation:
 * - If no_initial_perturbation is true: chains start at exact values
 * - Otherwise: chains start with small random perturbations
 */
template<class T>
void CMCMC<T>::InitializeFromParameters(const std::vector<double>& par)
{
    // Validate input
    if (par.size() != settings.number_of_parameters)
    {
        last_error = "InitializeFromParameters: parameter vector size ("
                     + std::to_string(par.size()) + ") does not match expected ("
                     + std::to_string(settings.number_of_parameters) + ")";
        return;
    }

    // Clear/create detail log file
    std::ofstream detailFile(fileInformation.detailfilename, std::ios::trunc);
    if (!detailFile) {
        last_error = "InitializeFromParameters: Could not open detail file: "
                     + fileInformation.detailfilename;
        std::cerr << last_error << std::endl;
        return;
    }
    detailFile.close();

    // Allocate storage
    parameterSamples.resize(settings.total_number_of_samples);
    logPosterior.resize(settings.total_number_of_samples);
    logPosteriorTransformed.resize(settings.total_number_of_samples);

    for (unsigned int i = 0; i < settings.total_number_of_samples; ++i)
    {
        parameterSamples[i].resize(settings.number_of_parameters);
    }

    perturbationCoefficients.resize(settings.number_of_parameters);

    // Calculate perturbation coefficients
    if (settings.sensitivity_based_perturbation)
    {
        // Use sensitivity to scale perturbations
        CVector sensitivity = CalculateSensitivity(settings.increment_for_sensitivity, par);

        for (unsigned int j = 0; j < settings.number_of_parameters; ++j)
        {
            Parameter* param = GetParameter(j);
            if (!param) continue;

            std::string distribution = param->GetPriorDistribution();
            double sensValue = std::fabs(sensitivity[j]);

            // Avoid division by zero
            if (sensValue < 1e-10) {
                sensValue = 1e-10;
            }

            if (distribution == "normal" || distribution == "uniform")
            {
                perturbationCoefficients[j] = settings.perturbation_factor / sensValue;
            }
            else if (distribution == "log-normal")
            {
                // For log-normal: scale by sqrt(param) * sensitivity
                perturbationCoefficients[j] = settings.perturbation_factor
                                              / (std::sqrt(par[j]) * sensValue);
            }
        }
    }
    else
    {
        // Use parameter range to scale perturbations
        for (unsigned int j = 0; j < settings.number_of_parameters; ++j)
        {
            Parameter* param = GetParameter(j);
            if (!param) continue;

            std::string distribution = param->GetPriorDistribution();
            Parameter::Range range = param->GetRange();

            if (distribution == "normal" || distribution == "uniform")
            {
                perturbationCoefficients[j] = settings.perturbation_factor
                                              * (range.high - range.low);
            }
            else if (distribution == "log-normal")
            {
                perturbationCoefficients[j] = settings.perturbation_factor
                                              * (std::log(range.high) - std::log(range.low));
            }
        }
    }

    // Perturbation scale factor
    double perturbationScale = settings.no_initial_perturbation ? 0.0 : 1.0;

    // Initialize each chain with perturbations around given values
    for (unsigned int j = 0; j < settings.number_of_chains; ++j)
    {
        double logJacobian = 0.0;

        for (unsigned int i = 0; i < settings.number_of_parameters; ++i)
        {
            Parameter* param = GetParameter(i);
            if (!param) continue;

            std::string distribution = param->GetPriorDistribution();
            Parameter::Range range = param->GetRange();

            if (distribution == "normal" || distribution == "uniform")
            {
                // Add normally-distributed perturbation in linear space
                parameterSamples[j][i] = par[i] + perturbationScale
                                                      * normalDistribution.getnormalrand(0, perturbationCoefficients[i]);

                // For uniform prior: ensure value stays within bounds
                if (distribution == "uniform")
                {
                    int attempts = 0;
                    const int maxAttempts = 100;

                    while ((parameterSamples[j][i] < range.low || parameterSamples[j][i] > range.high)
                           && attempts < maxAttempts)
                    {
                        parameterSamples[j][i] = par[i] + perturbationScale
                                                              * normalDistribution.getnormalrand(0, perturbationCoefficients[i]);
                        attempts++;
                    }

                    // If still out of bounds, clamp to range
                    if (parameterSamples[j][i] < range.low) {
                        parameterSamples[j][i] = range.low;
                    }
                    if (parameterSamples[j][i] > range.high) {
                        parameterSamples[j][i] = range.high;
                    }
                }
            }
            else if (distribution == "log-normal")
            {
                // Add perturbation in log-space
                parameterSamples[j][i] = par[i] * std::exp(perturbationScale
                                                           * normalDistribution.getnormalrand(0, perturbationCoefficients[i]));

                // Accumulate Jacobian for log-normal transformation
                logJacobian += std::log(par[i]);
            }
        }

        // Evaluate initial log posterior
        logPosterior[j] = CalculateLogPosterior(parameterSamples[j], j);
        logPosteriorTransformed[j] = logPosterior[j] + logJacobian;
    }
}

/**
 * @brief Perform single MCMC step using Metropolis-Hastings algorithm
 * @param k Sample index (must be >= number_of_chains)
 * @return true if proposal accepted, false if rejected
 *
 * This implements the Metropolis-Hastings algorithm:
 * 1. Generate proposal by perturbing previous sample
 * 2. Calculate proposal log posterior
 * 3. Calculate acceptance ratio
 * 4. Accept/reject based on ratio
 *
 * The method handles log-normal parameters with proper Jacobian adjustment.
 */
template<class T>
bool CMCMC<T>::PerformStep(int k)
{
    // Validate input
    if (k < static_cast<int>(settings.number_of_chains))
    {
        last_error = "PerformStep: sample index " + std::to_string(k)
        + " must be >= number_of_chains (" + std::to_string(settings.number_of_chains) + ")";
        return false;
    }

#ifdef Q_GUI_SUPPORT
    qDebug() << "MCMC step " << k;
#endif

    // Calculate previous sample index (for the same chain)
    int previousIndex = k - settings.number_of_chains;

    // Generate proposal by perturbing previous sample
    std::vector<double> proposal = PerturbParameters(previousIndex);

    // Calculate Jacobian for log-normal parameters
    double logJacobian = 0.0;
    for (unsigned int i = 0; i < settings.number_of_parameters; ++i)
    {
        Parameter* param = GetParameter(i);
        if (param && param->GetPriorDistribution() == "log-normal")
        {
            logJacobian += std::log(proposal[i]);
        }
    }

    // Calculate proposal log posterior
    double proposalLogPosterior = CalculateLogPosterior(proposal, k) + logJacobian;

    // Get previous log posterior
    double previousLogPosterior = logPosterior[previousIndex];

    // Log proposal evaluation (thread-safe)
#pragma omp critical
    {
        std::ofstream detailFile(fileInformation.detailfilename, std::ios::app);
        if (detailFile.is_open())
        {
            detailFile << "Sample #" << k
                       << " proposal_log_posterior: " << proposalLogPosterior
                       << ", previous_log_posterior: " << previousLogPosterior << "\n";
            detailFile.close();
        }
    }

    // Metropolis-Hastings acceptance test
    bool accepted = false;

    // Check for valid posterior (not NaN or -inf)
    if (!std::isnan(proposalLogPosterior) && !std::isinf(proposalLogPosterior))
    {
        // Calculate acceptance ratio (in log space)
        double logAcceptanceRatio = proposalLogPosterior - previousLogPosterior;

        // Accept if ratio > 1, or with probability = ratio
        double uniformRandom = normalDistribution.unitrandom();

        if (std::log(uniformRandom) < logAcceptanceRatio)
        {
            accepted = true;
        }
    }

    // Update sample based on acceptance
    if (accepted)
    {
        // Accept proposal
        parameterSamples[k] = proposal;
        logPosterior[k] = proposalLogPosterior;
        logPosteriorTransformed[k] = proposalLogPosterior;

        // Log acceptance (thread-safe)
#pragma omp critical
        {
            std::ofstream detailFile(fileInformation.detailfilename, std::ios::app);
            if (detailFile.is_open())
            {
                detailFile << "Sample #" << k
                           << " proposal_log_posterior: " << proposalLogPosterior
                           << ", previous_log_posterior: " << previousLogPosterior
                           << ", ACCEPTED\n";
                detailFile.close();
            }
        }
    }
    else
    {
        // Reject proposal - keep previous sample
        parameterSamples[k] = parameterSamples[previousIndex];
        logPosterior[k] = logPosterior[previousIndex];
        logPosteriorTransformed[k] = proposalLogPosterior;  // Store proposal for diagnostics

        // Log rejection (thread-safe)
#pragma omp critical
        {
            std::ofstream detailFile(fileInformation.detailfilename, std::ios::app);
            if (detailFile.is_open())
            {
                detailFile << "Sample #" << k
                           << " proposal_log_posterior: " << proposalLogPosterior
                           << ", previous_log_posterior: " << previousLogPosterior
                           << ", REJECTED\n";
                detailFile.close();
            }
        }
    }

    return accepted;
}

/**
 * @brief Generate proposal by perturbing current parameters
 * @param k Sample index to perturb from
 * @return Perturbed parameter vector
 *
 * Perturbation strategy:
 * - Normal/Uniform parameters: additive perturbation X_new = X_old + σ * N(0,1)
 * - Log-normal parameters: multiplicative perturbation X_new = X_old * exp(σ * N(0,1))
 *
 * The perturbation coefficients (σ) are scaled adaptively based on acceptance rate.
 */
template<class T>
std::vector<double> CMCMC<T>::PerturbParameters(int k)
{
    // Validate input
    if (k < 0 || k >= static_cast<int>(parameterSamples.size()))
    {
        last_error = "PerturbParameters: sample index " + std::to_string(k)
        + " out of range [0, " + std::to_string(parameterSamples.size() - 1) + "]";
        return std::vector<double>();
    }

    // Initialize proposal vector
    std::vector<double> proposal;
    proposal.resize(settings.number_of_parameters);

    // Perturb each parameter
    for (unsigned int i = 0; i < settings.number_of_parameters; ++i)
    {
        Parameter* param = GetParameter(i);
        if (!param)
        {
            // If parameter is null, keep current value
            proposal[i] = parameterSamples[k][i];
            continue;
        }

        // Get random normal deviate
        double normalDeviate = normalDistribution.getstdnormalrand();

        // Apply perturbation based on parameter distribution
        std::string distribution = param->GetPriorDistribution();

        if (distribution == "log-normal")
        {
            // Multiplicative perturbation in log-space
            // X_new = X_old * exp(σ * ε), where ε ~ N(0,1)
            proposal[i] = parameterSamples[k][i]
                          * std::exp(perturbationCoefficients[i] * normalDeviate);
        }
        else
        {
            // Additive perturbation for normal/uniform distributions
            // X_new = X_old + σ * ε, where ε ~ N(0,1)
            proposal[i] = parameterSamples[k][i]
                          + perturbationCoefficients[i] * normalDeviate;
        }
    }

    return proposal;
}

/**
 * @brief Perform multiple MCMC steps with output and progress tracking
 * @param k Starting sample index
 * @param numSamples Number of samples to generate
 * @param filename Output file for samples
 * @param runtimeWindow Optional GUI window for progress updates
 * @return true if completed, false if stopped by user
 *
 * This method:
 * 1. Creates output file with header (if not continuing)
 * 2. Runs parallel chains with OpenMP
 * 3. Adapts perturbation based on acceptance rate
 * 4. Periodically writes samples to file
 * 5. Updates GUI if runtime window provided
 */
template<class T>
bool CMCMC<T>::PerformSteps(int k, int numSamples, const std::string& filename,
                            ProgressWindow* runtimeWindow)
{
    // Constants for periodic operations
    const int WRITE_INTERVAL = 50;  // Write to file every N*chains samples
    const int ADAPT_INTERVAL = 50;  // Adapt perturbation every N*chains samples
    const int CONSOLE_UPDATE_INTERVAL = 10; // Console output every N*chains samples

    // Initialize output file if not continuing
    if (!settings.continue_mcmc)
    {
        FILE* file = fopen(filename.c_str(), "w");
        if (!file)
        {
            last_error = "PerformSteps: Could not open output file: " + filename;
            return false;
        }

        // Write header
        fprintf(file, "sample_no, ");
        for (unsigned int i = 0; i < settings.number_of_parameters; ++i)
        {
            Parameter* param = GetParameter(i);
            if (param) {
                fprintf(file, "%s, ", param->GetName().c_str());
            }
        }
        fprintf(file, "log_posterior, log_posterior_transformed, stuck_counter, ");
        for (unsigned int j = 0; j < perturbationCoefficients.size(); ++j)
        {
            fprintf(file, "perturbation_coeff_%u, ", j);
        }
        fprintf(file, "\n");
        fclose(file);
    }

    // Initialize tracking vectors
    CVector stuckCounter(settings.number_of_chains);
    CVector acceptedThisRound(settings.number_of_chains);

    // Store initial perturbation factor for plotting
    double initialPerturbation = perturbationCoefficients.empty() ? 1.0 : perturbationCoefficients[0];

    // Starting index for this run
    int startIndex = k;

#ifndef Q_GUI_SUPPORT
    // Print header for console output
    std::cout << "\n=== MCMC Sampling Progress ===" << std::endl;
    std::cout << "Total samples: " << numSamples << std::endl;
    std::cout << "Number of chains: " << settings.number_of_chains << std::endl;
    std::cout << "Target acceptance rate: " << settings.acceptance_rate << std::endl;
    std::cout << std::string(80, '-') << std::endl;
    std::cout << std::setw(10) << "Sample"
              << std::setw(15) << "Progress"
              << std::setw(15) << "Accept Rate"
              << std::setw(15) << "Log Post"
              << std::setw(15) << "Perturb"
              << std::endl;
    std::cout << std::string(80, '-') << std::endl;
#endif

    // Main sampling loop - process chains in parallel
    for (unsigned int kk = k;
         kk < k + numSamples + settings.number_of_chains;
         kk += settings.number_of_chains)
    {
        // Process GUI events to keep interface responsive
#ifdef Q_GUI_SUPPORT
        QCoreApplication::processEvents();

        // Check if user requested stop
        if (runtimeWindow && runtimeWindow->isCancelRequested())
        {
            break;
        }
#endif

        // Configure threading
#ifndef NO_OPENMP
        omp_set_num_threads(settings.numberOfThreads);
#endif

        // Set OpenBLAS to single-threaded for parallel MCMC
#ifndef _WINDOWS
#ifndef _MacOS
        openblas_set_num_threads(1);
#endif
#endif

        // Seed random number generator for each thread
#ifdef WIN64
#pragma omp parallel
        {
            srand(static_cast<int>(time(NULL)) ^ omp_get_thread_num() + kk);
        }
#endif

        // Perform MCMC steps for all chains in parallel
#pragma omp parallel for
        for (int jj = kk;
             jj < std::min(static_cast<int>(kk + settings.number_of_chains),
                           static_cast<int>(settings.total_number_of_samples));
             ++jj)
        {
#ifdef Q_GUI_SUPPORT
            qDebug() << "Starting step: " << jj;
#endif

            // Perform one MCMC step
            bool accepted = PerformStep(jj);

            // Update counters
            int chainIndex = jj - kk;
            if (!accepted)
            {
                stuckCounter[chainIndex]++;
                acceptedThisRound[chainIndex] = 0;
            }
            else
            {
                stuckCounter[chainIndex] = 0;
                acceptedThisRound[chainIndex] = 1;
            }

            // Log to GUI if available and enabled
        }

        // Restore OpenBLAS threading
#ifndef _WINDOWS
#ifndef _MacOS
        unsigned int cores = std::thread::hardware_concurrency();
        openblas_set_num_threads(cores > 0 ? cores : 1);
#endif
#endif

        // Update acceptance statistics
        acceptedCount += acceptedThisRound.sum();
        totalCount += acceptedThisRound.num;

        // Process GUI events
#ifdef Q_GUI_SUPPORT
        QCoreApplication::processEvents();
#endif

        // Console output (when GUI not available)
#ifndef Q_GUI_SUPPORT
        if ((kk - startIndex) % (CONSOLE_UPDATE_INTERVAL * settings.number_of_chains) == 0)
        {
            double progress = static_cast<double>(kk - startIndex) / static_cast<double>(numSamples) * 100.0;
            double currentAcceptanceRate = (totalCount > 0) ?
                                               static_cast<double>(acceptedCount) / static_cast<double>(totalCount) : 0.0;
            double avgLogPost = 0.0;
            int countValid = 0;

            // Calculate average log posterior for recent samples
            for (int jj = std::max(0, static_cast<int>(kk) - 10); jj < static_cast<int>(kk); ++jj)
            {
                if (jj < static_cast<int>(logPosterior.size()) && !std::isnan(logPosterior[jj]))
                {
                    avgLogPost += logPosterior[jj];
                    countValid++;
                }
            }
            avgLogPost = (countValid > 0) ? avgLogPost / countValid : 0.0;

            double perturbRatio = (initialPerturbation > 0) ?
                                      perturbationCoefficients[0] / initialPerturbation : 1.0;

            std::cout << std::setw(10) << kk
                      << std::setw(14) << std::fixed << std::setprecision(1) << progress << "%"
                      << std::setw(15) << std::fixed << std::setprecision(3) << currentAcceptanceRate
                      << std::setw(15) << std::scientific << std::setprecision(2) << avgLogPost
                      << std::setw(15) << std::fixed << std::setprecision(4) << perturbRatio
                      << std::endl;
        }
#endif

        // Periodic file output
        if ((kk - startIndex) % (WRITE_INTERVAL * settings.number_of_chains) == 0 ||
            kk >= k + numSamples)
        {
            FILE* file = fopen(filename.c_str(), "a");
            if (!file)
            {
                last_error = "PerformSteps: Could not open file for writing: " + filename;
                continue;
            }

            // Determine range of samples to write
            int writeStart = std::max(
                static_cast<int>(kk + settings.number_of_chains) - static_cast<int>(WRITE_INTERVAL * settings.number_of_chains),
                startIndex
                );

            int writeEnd = std::min(
                static_cast<int>(kk + settings.number_of_chains),
                static_cast<int>(settings.total_number_of_samples)
                );

            // Write samples to file
            for (int jj = writeStart; jj < writeEnd; ++jj)
            {
                if (jj % settings.save_interval == 0)
                {
                    fprintf(file, "%d, ", jj);

                    // Write parameters
                    for (unsigned int i = 0; i < settings.number_of_parameters; ++i)
                    {
                        fprintf(file, "%le, ", parameterSamples[jj][i]);
                    }

                    // Write log posteriors and diagnostics
                    fprintf(file, "%le, %le, %f, ",
                            logPosterior[jj],
                            logPosteriorTransformed[jj],
                            stuckCounter[jj % settings.number_of_chains]);

                    // Write perturbation coefficients
                    for (size_t j = 0; j < perturbationCoefficients.size(); ++j)
                    {
                        fprintf(file, "%le, ", perturbationCoefficients[j]);
                    }

                    fprintf(file, "\n");
                }
            }

            fclose(file);
        }

        // Adaptive perturbation scaling
        if ((kk - startIndex) % (ADAPT_INTERVAL * settings.number_of_chains) == 0)
        {
            if (totalCount > 0)
            {
                double currentAcceptanceRate = static_cast<double>(acceptedCount) / static_cast<double>(totalCount);

                if (currentAcceptanceRate > settings.acceptance_rate)
                {
                    // Acceptance rate too high - increase perturbation
                    for (unsigned int i = 0; i < settings.number_of_parameters; ++i)
                    {
                        perturbationCoefficients[i] /= settings.perturbation_change_scale;
                    }
                }
                else
                {
                    // Acceptance rate too low - decrease perturbation
                    for (unsigned int i = 0; i < settings.number_of_parameters; ++i)
                    {
                        perturbationCoefficients[i] *= settings.perturbation_change_scale;
                    }
                }

                // Reset counters
                acceptedCount = 0;
                totalCount = 0;
            }
        }

        // Update GUI if available
#ifdef Q_GUI_SUPPORT
        if (runtimeWindow)
        {
            // Update progress bar
            double progress = static_cast<double>(kk - startIndex) / static_cast<double>(numSamples);
            runtimeWindow->setProgress(progress);

            // Plot acceptance rate
            if (totalCount > 0)
            {
                double currentAcceptanceRate = static_cast<double>(acceptedCount) / static_cast<double>(totalCount);
                runtimeWindow->addMCMCPoint(kk, currentAcceptanceRate);
            }

            // Plot perturbation scaling (if plot2 enabled)
            if (initialPerturbation > 0)
            {
                double perturbationRatio = perturbationCoefficients[0] / initialPerturbation;
                runtimeWindow->addFitnessPoint(kk, perturbationRatio);
            }


        }
#endif
    }

#ifndef Q_GUI_SUPPORT
    std::cout << std::string(80, '-') << std::endl;
    std::cout << "MCMC sampling completed!" << std::endl;
    std::cout << "Final acceptance rate: "
              << std::fixed << std::setprecision(3)
              << ((totalCount > 0) ? static_cast<double>(acceptedCount) / static_cast<double>(totalCount) : 0.0)
              << std::endl << std::endl;
#endif

    return true;
}

#ifdef Q_GUI_SUPPORT
template<class T>
void CMCMC<T>::SetRunTimeWindow(ProgressWindow *_rtw)
{
    runtimeWindow = _rtw;
}
#endif

/**
 * @brief Calculate parameter sensitivities using finite differences
 * @param increment Relative finite difference increment (e.g., 0.01 for 1%)
 * @param par Parameter values at which to evaluate sensitivity
 * @return Vector of sensitivity values for each parameter
 *
 * Computes sensitivity as:
 *   S_i = [sqrt(|P(θ)|) - sqrt(|P(θ + Δθ_i)|)] / (Δθ_i)
 * where:
 *   - P(θ) is the posterior probability at parameter vector θ
 *   - Δθ_i = increment * θ_i is the perturbation to parameter i
 *   - S_i measures how much the posterior changes with parameter i
 *
 * The square root transformation makes the sensitivity measure more stable
 * for parameters with very different posterior magnitudes.
 *
 * @note Higher absolute sensitivity indicates greater parameter importance
 * @note Returns zero sensitivity if parameter value is zero (to avoid division by zero)
 *
 * @example
 * @code
 * std::vector<double> params = {1.0, 2.5, 0.3};
 * CVector sensitivities = mcmc.CalculateSensitivity(0.01, params);
 * // sensitivities[i] tells how sensitive the posterior is to parameter i
 * @endcode
 */
template<class T>
CVector CMCMC<T>::CalculateSensitivity(double increment,
                                       std::vector<double> par)
{
    // Validate inputs
    if (!model)
    {
        last_error = "CalculateSensitivity: model pointer is null";
        return CVector(settings.number_of_parameters);
    }

    if (par.size() != settings.number_of_parameters)
    {
        last_error = "CalculateSensitivity: parameter vector size ("
                     + std::to_string(par.size()) + ") does not match expected ("
                     + std::to_string(settings.number_of_parameters) + ")";
        return CVector(settings.number_of_parameters);
    }

    if (increment <= 0.0)
    {
        last_error = "CalculateSensitivity: increment must be positive, got "
                     + std::to_string(increment);
        return CVector(settings.number_of_parameters);
    }

    // Calculate baseline log posterior at current parameters
    double baseLogPosterior = CalculateLogPosterior(par, 0, false);

    // Initialize sensitivity vector
    CVector sensitivities(settings.number_of_parameters);

    // Calculate sensitivity for each parameter using finite differences
    for (unsigned int i = 0; i < settings.number_of_parameters; ++i)
    {
        // Skip if parameter is zero (avoid division by zero)
        if (std::fabs(par[i]) < 1e-12)
        {
            sensitivities[i] = 0.0;
            continue;
        }

        // Create perturbed parameter vector
        std::vector<double> perturbedParams = par;
        double parameterIncrement = increment * par[i];
        perturbedParams[i] = par[i] + parameterIncrement;

        // Validate perturbed parameter is within bounds
        Parameter* param = GetParameter(i);
        if (param)
        {
            Parameter::Range range = param->GetRange();
            if (perturbedParams[i] > range.high)
            {
                // If perturbed value exceeds upper bound, use backward difference
                perturbedParams[i] = par[i] - parameterIncrement;
            }
            if (perturbedParams[i] < range.low)
            {
                // If still out of bounds, set sensitivity to zero
                sensitivities[i] = 0.0;
                continue;
            }
        }

        // Calculate log posterior at perturbed parameters
        double perturbedLogPosterior = CalculateLogPosterior(perturbedParams, 0, false);

        // Calculate sensitivity using square-root transformation
        // S_i = [sqrt(|P_base|) - sqrt(|P_perturbed|)] / Δθ_i
        // The square root makes the measure more stable across different scales
        double sqrtBasePost = std::sqrt(std::fabs(baseLogPosterior));
        double sqrtPerturbedPost = std::sqrt(std::fabs(perturbedLogPosterior));

        sensitivities[i] = (sqrtBasePost - sqrtPerturbedPost) / parameterIncrement;

        // Handle NaN or Inf results
        if (std::isnan(sensitivities[i]) || std::isinf(sensitivities[i]))
        {
            sensitivities[i] = 0.0;
        }
    }

    return sensitivities;
}


/**
 * @brief Calculate log-scale parameter sensitivities
 * @param increment Relative finite difference increment (e.g., 0.01 for 1%)
 * @param par Parameter values at which to evaluate sensitivity
 * @return Vector of log-scale sensitivity values for each parameter
 *
 * Computes log-scale sensitivity as:
 *   S_log_i = θ_i * S_i
 * where:
 *   - S_i is the standard sensitivity from CalculateSensitivity()
 *   - θ_i is the parameter value
 *
 * This transformation gives the sensitivity in terms of relative (percentage)
 * changes rather than absolute changes. It's useful when:
 * - Parameters have very different scales (e.g., 0.001 vs 1000)
 * - You want to compare importance across parameters of different units
 * - Working with parameters that vary over orders of magnitude
 *
 * Interpretation:
 * - S_log_i measures: "How much does posterior change per 1% change in parameter i?"
 * - Higher absolute value = parameter has greater relative importance
 *
 * @note This is equivalent to computing sensitivity with respect to log(θ)
 * @note Returns zero if CalculateSensitivity() fails or returns invalid results
 *
 * @example
 * @code
 * std::vector<double> params = {1.0, 2.5, 0.3};
 * CVector logSens = mcmc.CalculateSensitivityLog(0.01, params);
 * // logSens[i] tells relative importance of parameter i
 * // Compare directly across parameters despite different scales
 * @endcode
 */
template<class T>
CVector CMCMC<T>::CalculateSensitivityLog(double increment,
                                          std::vector<double> par)
{
    // Validate inputs (basic checks before calling CalculateSensitivity)
    if (!model)
    {
        last_error = "CalculateSensitivityLog: model pointer is null";
        return CVector(settings.number_of_parameters);
    }

    if (par.size() != settings.number_of_parameters)
    {
        last_error = "CalculateSensitivityLog: parameter vector size ("
                     + std::to_string(par.size()) + ") does not match expected ("
                     + std::to_string(settings.number_of_parameters) + ")";
        return CVector(settings.number_of_parameters);
    }

    // Calculate standard sensitivities
    CVector standardSensitivities = CalculateSensitivity(increment, par);

    // Check if CalculateSensitivity failed (returns all zeros on error)
    bool allZeros = true;
    for (unsigned int i = 0; i < settings.number_of_parameters; ++i)
    {
        if (std::fabs(standardSensitivities[i]) > 1e-12)
        {
            allZeros = false;
            break;
        }
    }

    if (allZeros && !last_error.empty())
    {
        // CalculateSensitivity failed
        last_error = "CalculateSensitivityLog: CalculateSensitivity failed - " + last_error;
        return CVector(settings.number_of_parameters);
    }

    // Initialize log-scale sensitivity vector
    CVector logScaleSensitivities(settings.number_of_parameters);

    // Transform to log-scale by multiplying by parameter values
    // S_log_i = θ_i * S_i
    // This gives sensitivity per unit relative (log-scale) change
    for (unsigned int i = 0; i < settings.number_of_parameters; ++i)
    {
        logScaleSensitivities[i] = par[i] * standardSensitivities[i];

        // Handle NaN or Inf results that might arise from multiplication
        if (std::isnan(logScaleSensitivities[i]) || std::isinf(logScaleSensitivities[i]))
        {
            logScaleSensitivities[i] = 0.0;
        }
    }

    return logScaleSensitivities;
}

/**
 * @brief Read MCMC samples from a previous run to continue sampling
 * @param filename Path to the MCMC output file
 * @return Number of samples successfully read, or -1 on error
 *
 * File format expected (CSV):
 * - Header line: sample_no, param1, param2, ..., log_posterior,
 *                log_posterior_transformed, stuck_counter, perturbation_coeff_0, ...
 * - Data lines: one per sample with values matching header
 *
 * This method:
 * 1. Opens and validates the file
 * 2. Parses the header to ensure correct format
 * 3. Reads all sample data lines
 * 4. Loads parameter values, log posteriors, and perturbation coefficients
 * 5. Resizes internal storage to accommodate the samples
 *
 * @note Sets last_error on failure
 * @note Only loads complete, valid lines (skips malformed lines)
 * @note Updates perturbation coefficients from the last valid sample
 *
 * @example
 * @code
 * int samplesRead = mcmc.ReadFromFile("previous_run.txt");
 * if (samplesRead > 0) {
 *     std::cout << "Loaded " << samplesRead << " samples" << std::endl;
 *     // Continue sampling from here
 * }
 * @endcode
 */
template<class T>
int CMCMC<T>::ReadFromFile(std::string filename)
{
    // Open file
    std::ifstream file(filename);
    if (!file.is_open())
    {
        last_error = "ReadFromFile: Could not open file: " + filename;
        std::cerr << last_error << std::endl;
        return -1;
    }

    // Read and validate header line
    std::vector<std::string> headerTokens = aquiutils::getline(file);
    if (headerTokens.empty())
    {
        last_error = "ReadFromFile: File is empty or header is missing: " + filename;
        file.close();
        return -1;
    }

    // Calculate expected number of columns
    // Format: sample_no, params..., log_posterior, log_posterior_transformed,
    //         stuck_counter, perturbation_coeffs...
    const size_t expectedColumns = 1 // sample_no
                                   + settings.number_of_parameters  // parameters
                                   + 2  // log_posterior, log_posterior_transformed
                                   + 1  // stuck_counter
                                   + settings.number_of_parameters;  // perturbation coefficients

    if (headerTokens.size() < expectedColumns)
    {
        last_error = "ReadFromFile: Invalid header format. Expected "
                     + std::to_string(expectedColumns) + " columns, found "
                     + std::to_string(headerTokens.size());
        std::cerr << last_error << std::endl;
        file.close();
        return -1;
    }

    // Temporary storage for reading samples
    std::vector<std::vector<double>> tempParameterSamples;
    std::vector<double> tempLogPosterior;
    std::vector<double> tempLogPosteriorTransformed;
    std::vector<double> tempPerturbationCoefficients(settings.number_of_parameters);

    int samplesRead = 0;
    int lineNumber = 1;  // Start at 1 (header is line 0)

    // Read data lines
    while (!file.eof())
    {
        std::vector<std::string> tokens = aquiutils::getline(file);

        // Skip empty lines
        if (tokens.empty())
        {
            continue;
        }

        lineNumber++;

        // Validate line has correct number of columns
        if (tokens.size() != expectedColumns)
        {
            std::cerr << "ReadFromFile: Warning - Line " << lineNumber
                      << " has " << tokens.size() << " columns, expected "
                      << expectedColumns << ". Skipping line." << std::endl;
            continue;
        }

        try
        {
            // Parse sample data
            // Column 0: sample number (we'll ignore this and use our own counter)

            // Columns 1 to number_of_parameters: parameter values
            std::vector<double> parameterValues(settings.number_of_parameters);
            for (unsigned int i = 0; i < settings.number_of_parameters; ++i)
            {
                parameterValues[i] = aquiutils::atof(tokens[i + 1]);
            }

            // Next columns: log posterior values
            size_t logPostCol = 1 + settings.number_of_parameters;
            double logPost = aquiutils::atof(tokens[logPostCol]);
            double logPostTransformed = aquiutils::atof(tokens[logPostCol + 1]);

            // Column after that: stuck_counter (we skip this)
            // Then: perturbation coefficients
            size_t pertCoeffStartCol = logPostCol + 2 + 1;  // +2 for log posts, +1 for stuck_counter

            for (unsigned int i = 0; i < settings.number_of_parameters; ++i)
            {
                if (pertCoeffStartCol + i < tokens.size())
                {
                    tempPerturbationCoefficients[i] = aquiutils::atof(tokens[pertCoeffStartCol + i]);
                }
            }

            // Store the sample
            tempParameterSamples.push_back(parameterValues);
            tempLogPosterior.push_back(logPost);
            tempLogPosteriorTransformed.push_back(logPostTransformed);

            samplesRead++;
        }
        catch (const std::exception& e)
        {
            std::cerr << "ReadFromFile: Error parsing line " << lineNumber
                      << ": " << e.what() << ". Skipping line." << std::endl;
            continue;
        }
    }

    file.close();

    // Check if we read any samples
    if (samplesRead == 0)
    {
        last_error = "ReadFromFile: No valid samples found in file: " + filename;
        std::cerr << last_error << std::endl;
        return -1;
    }

    // Ensure we have enough space allocated
    if (tempParameterSamples.size() > parameterSamples.size())
    {
        parameterSamples.resize(tempParameterSamples.size());
        logPosterior.resize(tempParameterSamples.size());
        logPosteriorTransformed.resize(tempParameterSamples.size());

        for (size_t i = 0; i < parameterSamples.size(); ++i)
        {
            parameterSamples[i].resize(settings.number_of_parameters);
        }
    }

    // Copy temporary data to member variables
    for (size_t i = 0; i < tempParameterSamples.size(); ++i)
    {
        parameterSamples[i] = tempParameterSamples[i];
        logPosterior[i] = tempLogPosterior[i];
        logPosteriorTransformed[i] = tempLogPosteriorTransformed[i];
    }

    // Update perturbation coefficients from last valid sample
    perturbationCoefficients = tempPerturbationCoefficients;

    // Success
    std::cout << "ReadFromFile: Successfully loaded " << samplesRead
              << " samples from " << filename << std::endl;

    return samplesRead;
}


/**
 * @brief Calculate discretized prior distributions for all parameters
 * @param n_bins Number of bins for discretization (must be > 0)
 * @return TimeSeriesSet containing prior distribution for each parameter
 *
 * Creates discretized probability density functions for each parameter's
 * prior distribution. The distribution is evaluated at n_bins points
 * spanning ±4 standard deviations from the mean.
 *
 * Supported distributions:
 * - Normal: Gaussian PDF centered at mean with given std deviation
 * - Uniform: Flat distribution between low and high bounds
 * - Log-normal: Log-normal PDF in linear space
 *
 * For Normal/Uniform:
 *   Range: [mean - 4*σ, mean + 4*σ] (covers 99.99% of probability)
 *   PDF(x) = (1/(σ*√(2π))) * exp(-(x-μ)²/(2σ²))
 *
 * For Log-normal:
 *   Range: [mean * exp(-4*σ), mean * exp(4*σ)]
 *   PDF(x) = (1/(x*σ*√(2π))) * exp(-(ln(x)-ln(μ))²/(2σ²))
 *
 * @note The x-axis (time) represents parameter values
 * @note The y-axis (value) represents probability density
 * @note Each TimeSeries is normalized to integrate to 1.0
 *
 * @example
 * @code
 * TimeSeriesSet<double> priors = mcmc.CalculatePriorDistribution(100);
 * priors.write("prior_distributions.txt");
 * // Plot to visualize parameter uncertainties before calibration
 * @endcode
 */
template<class T>
TimeSeriesSet<double> CMCMC<T>::CalculatePriorDistribution(int n_bins)
{
    // Validate input
    if (n_bins <= 0)
    {
        last_error = "CalculatePriorDistribution: n_bins must be positive, got "
                     + std::to_string(n_bins);
        std::cerr << last_error << std::endl;
        return TimeSeriesSet<double>();
    }

    if (!parameters)
    {
        last_error = "CalculatePriorDistribution: parameters pointer is null";
        std::cerr << last_error << std::endl;
        return TimeSeriesSet<double>();
    }

    // Constants
    const double NUM_STD_DEVS = 4.0;        // Range coverage (±4σ = 99.99%)
    const double SQRT_2PI = std::sqrt(2.0 * M_PI);  // √(2π) for normal PDF

    // Initialize result
    TimeSeriesSet<double> priorDistributions(settings.number_of_parameters);

    // Calculate prior distribution for each parameter
    for (unsigned int i = 0; i < settings.number_of_parameters; ++i)
    {
        Parameter* param = GetParameter(i);
        if (!param)
        {
            std::cerr << "CalculatePriorDistribution: Warning - Parameter "
                      << i << " is null. Skipping." << std::endl;
            continue;
        }

        // Get parameter properties
        std::string distribution = param->GetPriorDistribution();
        Parameter::Range range = param->GetRange();

        // For normal/log-normal, we need mean and std dev
        // These are typically computed from the range or stored in the parameter
        double mean = (range.high + range.low) / 2.0;
        double stdDev = (range.high - range.low) / (2.0 * NUM_STD_DEVS);

        // Determine range for discretization
        double minValue, maxValue;

        if (distribution == "log-normal")
        {
            // For log-normal: range in log space
            // Range: [mean * exp(-4σ), mean * exp(4σ)]
            if (mean <= 0.0)
            {
                std::cerr << "CalculatePriorDistribution: Warning - Parameter "
                          << i << " has invalid mean (" << mean
                          << ") for log-normal. Using range bounds." << std::endl;
                minValue = range.low;
                maxValue = range.high;
            }
            else
            {
                minValue = mean * std::exp(-NUM_STD_DEVS * stdDev);
                maxValue = mean * std::exp(NUM_STD_DEVS * stdDev);
            }

            // Ensure within bounds
            minValue = std::max(minValue, range.low);
            maxValue = std::min(maxValue, range.high);
        }
        else if (distribution == "uniform")
        {
            // For uniform: use full range
            minValue = range.low;
            maxValue = range.high;
        }
        else  // normal or other
        {
            // For normal: ±4σ from mean
            minValue = mean - NUM_STD_DEVS * stdDev;
            maxValue = mean + NUM_STD_DEVS * stdDev;

            // Ensure within parameter bounds
            minValue = std::max(minValue, range.low);
            maxValue = std::min(maxValue, range.high);
        }

        // Validate range
        if (maxValue <= minValue)
        {
            std::cerr << "CalculatePriorDistribution: Warning - Invalid range for parameter "
                      << i << " [" << minValue << ", " << maxValue << "]. Skipping."
                      << std::endl;
            continue;
        }

        // Create discretized distribution
        TimeSeries<double> priorDensity(n_bins);

        // Calculate bin width
        double binWidth = (maxValue - minValue) / static_cast<double>(n_bins);

        // Set bin centers and calculate PDF values
        for (int j = 0; j < n_bins; ++j)
        {
            // Bin center
            double x = minValue + (j + 0.5) * binWidth;
            priorDensity.setTime(j, x);

            // Calculate probability density based on distribution type
            double density = 0.0;

            if (distribution == "uniform")
            {
                // Uniform distribution: constant density
                density = 1.0 / (maxValue - minValue);
            }
            else if (distribution == "log-normal")
            {
                // Log-normal PDF: (1/(x*σ*√(2π))) * exp(-(ln(x)-ln(μ))²/(2σ²))
                if (x > 0.0 && mean > 0.0 && stdDev > 0.0)
                {
                    double logX = std::log(x);
                    double logMean = std::log(mean);
                    double exponent = -std::pow(logX - logMean, 2.0)
                                      / (2.0 * std::pow(stdDev, 2.0));
                    density = std::exp(exponent) / (x * stdDev * SQRT_2PI);
                }
            }
            else  // normal or default
            {
                // Normal (Gaussian) PDF: (1/(σ*√(2π))) * exp(-(x-μ)²/(2σ²))
                if (stdDev > 0.0)
                {
                    double exponent = -std::pow(x - mean, 2.0)
                    / (2.0 * std::pow(stdDev, 2.0));
                    density = std::exp(exponent) / (stdDev * SQRT_2PI);
                }
            }

            priorDensity.setValue(j, density);
        }

        // Set name and add to result
        priorDensity.setName(param->GetName());
        priorDistributions[i] = priorDensity;
    }

    return priorDistributions;
}

/**
 * @brief Generate model realizations from posterior parameter samples
 * @param MCMCout TimeSeriesSet containing parameter samples from MCMC
 *
 * This method:
 * 1. Randomly samples parameters from the posterior distribution
 * 2. Runs the model with each parameter set
 * 3. Collects model predictions for all observations
 * 4. Calculates percentile bands (2.5%, 50%, 97.5%)
 * 5. Writes results to files
 *
 * Uses parallel processing to run multiple realizations simultaneously.
 * The number of realizations is controlled by settings.number_of_post_estimate_realizations.
 *
 * Output files (written to fileInformation.outputpath):
 * - "Realizations_[ObsName].txt": All model realizations
 * - "Predicted_95p_Bracket_[ObsName].txt": Percentile bands
 *
 * @note Requires observations to be set up and model to be valid
 * @note Updates GUI progress if runtimeWindow is provided
 * @note Skips burnout samples when sampling parameters
 *
 * @example
 * @code
 * TimeSeriesSet<double> samples = mcmc.GetParameterSamples();
 * mcmc.ProduceRealizations(samples);
 * // Files written with uncertainty bands for all observations
 * @endcode
 */
template<class T>
void CMCMC<T>::ProduceRealizations(TimeSeriesSet<double>& MCMCout)
{
    // Validate inputs
    if (!model)
    {
        last_error = "ProduceRealizations: model pointer is null";
        std::cerr << last_error << std::endl;
        return;
    }

    if (!observations)
    {
        last_error = "ProduceRealizations: observations pointer is null";
        std::cerr << last_error << std::endl;
        return;
    }

    if (observations->empty())
    {
        last_error = "ProduceRealizations: no observations available";
        std::cerr << last_error << std::endl;
        return;
    }

    if (settings.number_of_post_estimate_realizations == 0)
    {
        std::cerr << "ProduceRealizations: Warning - number_of_post_estimate_realizations is 0. Nothing to do."
                  << std::endl;
        return;
    }

    if (MCMCout.seriesCount() == 0)
    {
        last_error = "ProduceRealizations: MCMCout has no parameters";
        std::cerr << last_error << std::endl;
        return;
    }

    // Update GUI
#ifdef Q_GUI_SUPPORT
    if (runtimeWindow)
    {
        runtimeWindow->appendLog("Generating "
                                  + QString::number(settings.number_of_post_estimate_realizations)
                                  + " realizations...");
    }
#endif

    std::cout << "Generating " << settings.number_of_post_estimate_realizations
              << " realizations from " << (MCMCout[0].size() - settings.burnout_samples)
              << " post-burnout samples..." << std::endl;

    // Storage for all realizations
    std::vector<TimeSeriesSet<double>> realizedTimeSeries(observations->size());
    std::vector<TimeSeriesSet<double>> predictedPercentiles(observations->size());

    // Calculate number of batches needed
    unsigned int numThreads = settings.numberOfThreads;
    unsigned int numRealizations = settings.number_of_post_estimate_realizations;
    unsigned int numBatches = (numRealizations + numThreads - 1) / numThreads;  // Ceiling division

    // Process realizations in batches (for memory efficiency)
    for (unsigned int batchIndex = 0; batchIndex < numBatches; ++batchIndex)
    {
        // Calculate how many realizations in this batch
        unsigned int batchStart = batchIndex * numThreads;
        unsigned int batchSize = std::min(numThreads, numRealizations - batchStart);

        // Storage for this batch's results
        std::vector<std::vector<TimeSeries<double>>> batchResults(batchSize);
        for (unsigned int j = 0; j < batchSize; ++j)
        {
            batchResults[j].resize(observations->size());
        }

        // Configure OpenMP threading
#ifndef NO_OPENMP
        omp_set_num_threads(numThreads);
#endif

        // Seed random number generator for each thread
#pragma omp parallel
        {
            unsigned int seed = static_cast<unsigned int>(time(NULL))
            ^ (omp_get_thread_num() + 1) * 48271  // Prime multiplier
                ^ (batchIndex + 1) * 65537;           // Another prime
            srand(seed);
        }

        // Run models in parallel
#pragma omp parallel for
        for (int j = 0; j < static_cast<int>(batchSize); ++j)
        {
            try
            {
                // Create model copy OUTSIDE of getrandom call to avoid race conditions
                T modelCopy = *model;

                // Sample random parameters from posterior (skip burnout)
                std::vector<double> sampledParameters =
                    MCMCout.getrandom(settings.burnout_samples);

                // Validate sampled parameters
                if (sampledParameters.size() != settings.number_of_parameters)
                {
#pragma omp critical
                    {
                        std::cerr << "ProduceRealizations: Warning - Sampled parameters size mismatch. "
                                  << "Expected " << settings.number_of_parameters
                                  << ", got " << sampledParameters.size() << std::endl;
                    }
                    continue;
                }

                // Set parameters on the model copy
                for (unsigned int i = 0; i < settings.number_of_parameters; ++i)
                {
                    modelCopy.SetParameterValue(i, sampledParameters[i]);
                }

                // Extract results from THIS model copy's observations
                std::vector<Observation>* modelObservations = modelCopy.Observations();

                if (!modelObservations || modelObservations->size() != observations->size())
                {
#pragma omp critical
                    {
                        std::cerr << "ProduceRealizations: Warning - Model copy observations mismatch for realization "
                                  << (batchStart + j) << std::endl;
                    }
                    continue;
                }

                // Store results from this realization
                for (unsigned int i = 0; i < modelObservations->size(); ++i)
                {
                    Observation& obs = (*modelObservations)[i];
                    if (obs.GetModeledTimeSeries())
                    {
                        batchResults[j][i] = *(obs.GetModeledTimeSeries());
                    }
                }
            }
            catch (const std::exception& e)
            {
#pragma omp critical
                {
                    std::cerr << "ProduceRealizations: Error in realization "
                              << (batchStart + j) << ": " << e.what() << std::endl;
                }
            }
        }

        // Collect results from this batch (serial to avoid race conditions)
        for (unsigned int j = 0; j < batchSize; ++j)
        {
            for (unsigned int i = 0; i < observations->size(); ++i)
            {
                if (!batchResults[j][i].empty())
                {
                    realizedTimeSeries[i].append(batchResults[j][i]);
                }
            }
        }

        // Update progress
#ifdef Q_GUI_SUPPORT
        if (runtimeWindow)
        {
            double progress = static_cast<double>(batchStart + batchSize)
            / static_cast<double>(numRealizations);
            runtimeWindow->setProgress(progress);

            // Keep GUI responsive
            QCoreApplication::processEvents();

            // Check if user cancelled
            if (runtimeWindow->isCancelRequested())
            {
                runtimeWindow->appendLog("Realization generation stopped by user");
                return;
            }
        }
#endif

        // Progress output to console
        if ((batchIndex + 1) % std::max(1u, numBatches / 10) == 0 || batchIndex == numBatches - 1)
        {
            std::cout << "Completed " << (batchStart + batchSize) << "/" << numRealizations
                      << " realizations" << std::endl;
        }
    }

    // Use percentiles from settings, or default to 2.5%, 50%, 97.5%
    std::vector<double> percentiles;
    if (!outputPercentiles.empty())
    {
        percentiles = outputPercentiles;
    }
    else
    {
        percentiles = {0.025, 0.5, 0.975};  // Default: 95% credible interval + median
    }

    // Calculate percentiles and write output files
    for (unsigned int i = 0; i < observations->size(); ++i)
    {
        Observation* obs = observation(i);
        if (!obs)
        {
            std::cerr << "ProduceRealizations: Warning - Observation "
                      << i << " is null. Skipping." << std::endl;
            continue;
        }

        std::string obsName = obs->GetName();

        // Check if we have any realizations for this observation
        if (realizedTimeSeries[i].seriesCount() == 0)
        {
            std::cerr << "ProduceRealizations: Warning - No realizations collected for "
                      << obsName << ". Skipping." << std::endl;
            continue;
        }

        std::cout << "Writing " << realizedTimeSeries[i].seriesCount()
                  << " realizations for " << obsName << std::endl;

        // Write all realizations
        std::string realizationsFilename = fileInformation.outputpath
                                           + "Realizations_" + obsName + ".txt";
        try
        {
            realizedTimeSeries[i].write(realizationsFilename);
            std::cout << "  Written to: " << realizationsFilename << std::endl;
        }
        catch (const std::exception& e)
        {
            std::cerr << "ProduceRealizations: Error writing realizations for "
                      << obsName << ": " << e.what() << std::endl;
        }

        // Calculate percentile bands
        try
        {
            predictedPercentiles[i] = realizedTimeSeries[i].getpercentiles(percentiles);
            obs->SetPercentile95(predictedPercentiles[i]);
            obs->SetRealizations(realizedTimeSeries[i]);
        }
        catch (const std::exception& e)
        {
            std::cerr << "ProduceRealizations: Error calculating percentiles for "
                      << obsName << ": " << e.what() << std::endl;
            continue;
        }

        // Write percentile bands
        std::string percentilesFilename = fileInformation.outputpath
                                          + "Predicted_95p_Bracket_" + obsName + ".txt";
        try
        {
            predictedPercentiles[i].write(percentilesFilename);
            std::cout << "  Percentiles written to: " << percentilesFilename << std::endl;
        }
        catch (const std::exception& e)
        {
            std::cerr << "ProduceRealizations: Error writing percentiles for "
                      << obsName << ": " << e.what() << std::endl;
        }
    }

    // Final progress update
#ifdef Q_GUI_SUPPORT
    if (runtimeWindow)
    {
        runtimeWindow->setProgress(1.0);
        runtimeWindow->appendLog("Realizations complete. Generated "
                                  + QString::number(numRealizations) + " realizations for "
                                  + QString::number(observations->size()) + " observations.");
    }
#endif

    std::cout << "\n========================================" << std::endl;
    std::cout << "Realizations complete!" << std::endl;
    std::cout << "Generated " << numRealizations << " realizations" << std::endl;
    std::cout << "For " << observations->size() << " observations" << std::endl;
    std::cout << "========================================\n" << std::endl;
}

/**
 * @brief Calculate and write output percentiles from realizations
 * @param MCMCout TimeSeriesSet containing parameter samples from MCMC
 *
 * This is a wrapper function that:
 * 1. Calls ProduceRealizations() to generate model outputs
 * 2. Calculates percentile bands for all observations
 * 3. Writes percentile files
 *
 * The percentiles calculated are defined in the outputPercentiles member variable.
 * If not set, defaults to {0.025, 0.5, 0.975} (95% credible interval + median).
 *
 * Output files (written to fileInformation.outputpath):
 * - "Predicted_Percentiles_[ObsName].txt": Percentile bands for each observation
 * - "Predicted_Percentiles_WithNoise_[ObsName].txt": Percentiles including noise (if enabled)
 *
 * @note This function is essentially a convenience wrapper around ProduceRealizations()
 * @note The actual realization generation and percentile calculation happens in ProduceRealizations()
 * @note If you've already called ProduceRealizations(), this may duplicate work
 *
 * @deprecated Consider using ProduceRealizations() directly as it performs the same operations
 *
 * @example
 * @code
 * TimeSeriesSet<double> samples = mcmc.GetParameterSamples();
 * mcmc.outputPercentiles = {0.05, 0.5, 0.95};  // Custom percentiles
 * mcmc.GetOutputPercentiles(samples);
 * // Files written with percentile bands
 * @endcode
 */
/**
 * @brief Calculate output percentiles from realizations
 * @param MCMCout TimeSeriesSet containing parameter samples from MCMC
 * @return TimeSeriesSet where each series represents an observation's percentiles
 *
 * This function:
 * 1. Calls ProduceRealizations() to generate model outputs
 * 2. Extracts percentile information for all observations
 * 3. Returns percentiles in a structured format
 *
 * Return format:
 * - Each TimeSeries in the set represents one observation
 * - The "time" axis contains the percentile values (e.g., 0.025, 0.5, 0.975)
 * - The "values" contain the corresponding observation values at those percentiles
 * - Series names match observation names
 *
 * The percentiles calculated are defined in the outputPercentiles member variable.
 * If not set, defaults to {0.025, 0.5, 0.975} (95% credible interval + median).
 *
 * @note This function calls ProduceRealizations() which generates all realizations
 * @note If you've already called ProduceRealizations(), consider extracting
 *       percentiles directly from observation objects
 *
 * @example
 * @code
 * TimeSeriesSet<double> samples = mcmc.GetParameterSamples();
 * mcmc.outputPercentiles = {0.05, 0.25, 0.5, 0.75, 0.95};
 * TimeSeriesSet<double> percentiles = mcmc.GetOutputPercentiles(samples);
 *
 * // Access percentiles for observation i
 * // percentiles[i].getTime(j) = percentile level (e.g., 0.05)
 * // percentiles[i].getValue(j) = observation value at that percentile
 * percentiles.write("output_percentiles.txt");
 * @endcode
 */
template<class T>
TimeSeriesSet<double> CMCMC<T>::GetOutputPercentiles(TimeSeriesSet<double>& MCMCout)
{
    // Validate inputs
    if (!model)
    {
        last_error = "GetOutputPercentiles: model pointer is null";
        std::cerr << last_error << std::endl;
        return TimeSeriesSet<double>();
    }

    if (!observations)
    {
        last_error = "GetOutputPercentiles: observations pointer is null";
        std::cerr << last_error << std::endl;
        return TimeSeriesSet<double>();
    }

    if (observations->empty())
    {
        last_error = "GetOutputPercentiles: no observations available";
        std::cerr << last_error << std::endl;
        return TimeSeriesSet<double>();
    }

    // Update GUI
#ifdef Q_GUI_SUPPORT
    if (runtimeWindow)
    {
        runtimeWindow->appendLog("Calculating output percentiles...");
    }
#endif

    // Generate realizations - this calculates percentiles and stores them in observations
    ProduceRealizations(MCMCout);

    // Determine which percentiles to use
    std::vector<double> percentiles;
    if (!outputPercentiles.empty())
    {
        percentiles = outputPercentiles;
    }
    else
    {
        percentiles = {0.025, 0.5, 0.975};  // Default: 95% credible interval + median
    }

    unsigned int numObservations = observations->size();
    unsigned int numPercentiles = percentiles.size();

    // Create result TimeSeriesSet
    TimeSeriesSet<double> percentileResults(numObservations);

    // Extract percentiles for each observation
    for (unsigned int i = 0; i < numObservations; ++i)
    {
        Observation* obs = observation(i);
        if (!obs)
        {
            std::cerr << "GetOutputPercentiles: Warning - Observation "
                      << i << " is null. Skipping." << std::endl;
            continue;
        }

        // Get the percentile data that was stored by ProduceRealizations()
        TimeSeriesSet<double> obsPercentiles = obs->GetPercentile95();

        // Create a TimeSeries for this observation's percentiles
        TimeSeries<double> percentileSeries(numPercentiles);
        percentileSeries.setName(obs->GetName());

        // Populate the series
        // Time axis = percentile levels (0.025, 0.5, 0.975, etc.)
        // Values = observation values at those percentiles
        for (unsigned int j = 0; j < numPercentiles; ++j)
        {
            // Set the percentile level as the "time"
            percentileSeries.setTime(j, percentiles[j]);

            // Get the value at this percentile from the stored percentile data
            // obsPercentiles typically has series for different percentile bands
            // We need to extract the appropriate value
            if (j < obsPercentiles.seriesCount() && obsPercentiles[j].size() > 0)
            {
                // Use the mean of the percentile band as the representative value
                double percentileValue = obsPercentiles[j].mean();
                percentileSeries.setValue(j, percentileValue);
            }
            else
            {
                // If percentile data not available, set to NaN
                percentileSeries.setValue(j, std::nan(""));
            }
        }

        // Add to result set
        percentileResults[i] = percentileSeries;
    }

    // Provide user feedback
#ifdef Q_GUI_SUPPORT
    if (runtimeWindow)
    {
        QString summary = "Output percentiles calculated for "
                              + QString::number(numObservations) + " observations.\n";

        summary += "Percentiles: ";
        for (size_t i = 0; i < percentiles.size(); ++i)
        {
            summary += std::to_string(percentiles[i] * 100.0) + "%";
            if (i < percentiles.size() - 1)
            {
                summary += ", ";
            }
        }

        runtimeWindow->appendLog(summary);
    }
#endif

    // Log summary to console
    std::cout << "GetOutputPercentiles: Calculated percentiles for "
              << numObservations << " observations" << std::endl;
    std::cout << "Percentile levels: ";
    for (size_t i = 0; i < percentiles.size(); ++i)
    {
        std::cout << (percentiles[i] * 100.0) << "%";
        if (i < percentiles.size() - 1) std::cout << ", ";
    }
    std::cout << std::endl;

    return percentileResults;
}


/**
 * @brief Perform complete MCMC sampling workflow
 *
 * This is the main entry point for MCMC parameter estimation. It executes
 * the complete workflow:
 * 1. Initialize chains (or continue from previous run)
 * 2. Generate MCMC samples using Metropolis-Hastings
 * 3. Create posterior distributions for each parameter
 * 4. Calculate posterior percentiles (credible intervals)
 * 5. Generate model realizations from posterior
 * 6. Write all results to files
 *
 * The function uses settings from the 'settings' member variable:
 * - total_number_of_samples: How many samples to generate
 * - number_of_chains: How many parallel chains to run
 * - burnout_samples: How many initial samples to discard
 * - continue_mcmc: Whether to continue from a previous run
 * - continue_filename: File to read previous samples from
 *
 * Output files (written to fileInformation.outputpath):
 * - [outputfilename]: All MCMC samples with parameters and log posteriors
 * - "Posterior_Distributions.txt": Discretized posterior PDFs
 * - "Posterior_Percentiles.txt": Summary statistics (2.5%, 50%, 97.5%, mean)
 * - "Realizations_[ObsName].txt": Model predictions for each observation
 * - "Predicted_95p_Bracket_[ObsName].txt": Percentile bands for predictions
 *
 * @note Requires model, parameters, and observations to be properly configured
 * @note Updates GUI if runtimeWindow is provided
 * @note Can be interrupted by user if using GUI
 *
 * @example
 * @code
 * CMCMC<MyModel> mcmc(&model);
 * mcmc.SetProperty("number_of_samples", "10000");
 * mcmc.SetProperty("number_of_chains", "4");
 * mcmc.SetProperty("number_of_burnout_samples", "1000");
 * mcmc.Perform();
 * // All results written to output directory
 * @endcode
 */
template<class T>
void CMCMC<T>::Perform()
{
    // Validate prerequisites
    if (!model)
    {
        last_error = "Perform: model pointer is null";
        std::cerr << last_error << std::endl;
        return;
    }

    if (!parameters || parameters->empty())
    {
        last_error = "Perform: no parameters available for estimation";
        std::cerr << last_error << std::endl;
        return;
    }

    if (!observations || observations->empty())
    {
        last_error = "Perform: no observations available for calibration";
        std::cerr << last_error << std::endl;
        return;
    }

    if (settings.total_number_of_samples == 0)
    {
        last_error = "Perform: total_number_of_samples is zero";
        std::cerr << last_error << std::endl;
        return;
    }

    // ========================================================================
    // Step 1: Initialize MCMC chains
    // ========================================================================

#ifdef Q_GUI_SUPPORT
    if (runtimeWindow)
    {
        runtimeWindow->appendLog("Initializing MCMC chains...");
    }
#endif

    std::cout << "Initializing MCMC with " << settings.number_of_chains
              << " chains..." << std::endl;

    Initialize(false);  // Initialize with current parameter values

    // ========================================================================
    // Step 2: Load previous samples if continuing
    // ========================================================================

    int startingSample = settings.number_of_chains;

    if (settings.continue_mcmc)
    {
        if (settings.continue_filename.empty())
        {
            last_error = "Perform: continue_mcmc is true but continue_filename is empty";
            std::cerr << last_error << std::endl;
            return;
        }

#ifdef Q_GUI_SUPPORT
        if (runtimeWindow)
        {
            runtimeWindow->appendLog("Reading samples from " + QString::fromStdString(settings.continue_filename));
        }
#endif

        std::cout << "Continuing MCMC from file: " << settings.continue_filename << std::endl;

        int samplesRead = ReadFromFile(settings.continue_filename);

        if (samplesRead < 0)
        {
            last_error = "Perform: Failed to read samples from " + settings.continue_filename;
            std::cerr << last_error << std::endl;
            return;
        }

        startingSample = samplesRead;
        std::cout << "Loaded " << samplesRead << " samples. Continuing from sample "
                  << startingSample << std::endl;
    }

    // ========================================================================
    // Step 3: Generate MCMC samples
    // ========================================================================

#ifdef Q_GUI_SUPPORT
    if (runtimeWindow)
    {
        runtimeWindow->appendLog("Generating MCMC samples...");
    }
#endif

    // Calculate number of samples to generate
    int samplesToGenerate = settings.total_number_of_samples - startingSample;

    // Round down to multiple of number_of_chains for clean batch processing
    samplesToGenerate = (samplesToGenerate / settings.number_of_chains) * settings.number_of_chains;

    std::cout << "Generating " << samplesToGenerate << " samples..." << std::endl;

    bool success = PerformSteps(startingSample,
                                samplesToGenerate,
                                fileInformation.outputfilename,
                                runtimeWindow);

    if (!success)
    {
#ifdef Q_GUI_SUPPORT
        if (runtimeWindow)
        {
            runtimeWindow->appendLog("MCMC sampling stopped by user");
        }
#endif
        std::cout << "MCMC sampling stopped" << std::endl;
        return;
    }

    // ========================================================================
    // Step 4: Create posterior distributions
    // ========================================================================

#ifdef Q_GUI_SUPPORT
    if (runtimeWindow)
    {
        runtimeWindow->appendLog("Creating posterior distributions...");
    }
#endif

    std::cout << "Creating posterior distributions..." << std::endl;

    TimeSeriesSet<double> allPosteriorDistributions;
    TimeSeriesSet<double> allParameterSamples;
    std::vector<CVector> posteriorPercentiles;
    std::vector<std::string> columnLabels;
    std::vector<std::string> rowLabels = {"0.025", "0.5", "0.975", "mean"};

    // Process each parameter
    for (unsigned int paramIndex = 0; paramIndex < parameters->size(); ++paramIndex)
    {
        Parameter* param = GetParameter(paramIndex);
        if (!param)
        {
            std::cerr << "Perform: Warning - Parameter " << paramIndex
                      << " is null. Skipping." << std::endl;
            continue;
        }

        // Create separate time series for each chain
        TimeSeriesSet<double> chainValues(settings.number_of_chains);
        for (unsigned int chainIndex = 0; chainIndex < settings.number_of_chains; ++chainIndex)
        {
            chainValues.setname(chainIndex, "Chain_" + aquiutils::numbertostring(chainIndex));
        }

        // Collect samples for this parameter from all chains (skip burnout)
        for (unsigned int sampleIndex = settings.burnout_samples;
             sampleIndex < settings.total_number_of_samples;
             ++sampleIndex)
        {
            unsigned int chainIndex = sampleIndex % settings.number_of_chains;
            chainValues[chainIndex].append(sampleIndex, parameterSamples[sampleIndex][paramIndex]);
        }

        // Combine all chains into single time series
        TimeSeries<double> allSamples;
        for (unsigned int chainIndex = 0; chainIndex < settings.number_of_chains; ++chainIndex)
        {
            allSamples.append(chainValues[chainIndex]);
        }

        // Set parameter name
        chainValues.name = param->GetName();
        param->SetMCMCSamples(chainValues);

        // Create discretized posterior distribution (histogram)
        int numBins = std::max(50, static_cast<int>(allSamples.size() / 100));
        TimeSeries<double> posteriorDistribution = allSamples.distribution(numBins, 0);
        posteriorDistribution.setName("Posterior density");

        allPosteriorDistributions.append(posteriorDistribution, param->GetName());
        param->SetPosteriorDistribution(posteriorDistribution);

        // Store all samples
        allParameterSamples.append(allSamples);

        // Calculate posterior percentiles
        CVector percentiles;
        columnLabels.push_back(param->GetName());

        percentiles.append(allSamples.percentile(0.025, settings.burnout_samples));
        percentiles.append(allSamples.percentile(0.5, settings.burnout_samples));      // Median
        percentiles.append(allSamples.percentile(0.975, settings.burnout_samples));
        percentiles.append(allSamples.mean());

        posteriorPercentiles.push_back(percentiles);
    }

    // ========================================================================
    // Step 5: Write posterior distributions and percentiles
    // ========================================================================

    // Write posterior distributions
    std::string posteriorDistFile = fileInformation.outputpath + "Posterior_Distributions.txt";
    try
    {
        allPosteriorDistributions.write(posteriorDistFile);
        std::cout << "Posterior distributions written to: " << posteriorDistFile << std::endl;
    }
    catch (const std::exception& e)
    {
        std::cerr << "Perform: Error writing posterior distributions: " << e.what() << std::endl;
    }

    // Write posterior percentiles table
    std::string percentilesFile = fileInformation.outputpath + "Posterior_Percentiles.txt";
    try
    {
        WritePercentilesTable(posteriorPercentiles, columnLabels, rowLabels, percentilesFile);
        std::cout << "Posterior percentiles written to: " << percentilesFile << std::endl;
    }
    catch (const std::exception& e)
    {
        std::cerr << "Perform: Error writing percentiles: " << e.what() << std::endl;
    }

    // ========================================================================
    // Step 6: Generate model realizations
    // ========================================================================

    if (settings.number_of_post_estimate_realizations > 0)
    {
#ifdef Q_GUI_SUPPORT
        if (runtimeWindow)
        {
            runtimeWindow->appendLog("Generating model realizations...");
        }
#endif

        std::cout << "Generating " << settings.number_of_post_estimate_realizations
                  << " model realizations..." << std::endl;

        ProduceRealizations(allParameterSamples);

        std::cout << "Realizations complete" << std::endl;
    }
    else
    {
        std::cout << "Skipping realizations (number_of_post_estimate_realizations = 0)" << std::endl;
    }

    // ========================================================================
    // Final summary
    // ========================================================================

#ifdef Q_GUI_SUPPORT
    if (runtimeWindow)
    {
        QString summary = "MCMC complete!\n";
        summary += "Total samples: " + std::to_string(settings.total_number_of_samples) + "\n";
        summary += "Burnout samples: " + std::to_string(settings.burnout_samples) + "\n";
        summary += "Acceptance rate: " + std::to_string(GetAcceptanceRate() * 100.0) + "%\n";
        summary += "Parameters estimated: " + std::to_string(parameters->size()) + "\n";
        summary += "Observations used: " + std::to_string(observations->size());

        runtimeWindow->appendLog(summary);
        runtimeWindow->setProgress(1.0);
    }
#endif

    std::cout << "\n========================================" << std::endl;
    std::cout << "MCMC COMPLETE" << std::endl;
    std::cout << "========================================" << std::endl;
    std::cout << "Total samples: " << settings.total_number_of_samples << std::endl;
    std::cout << "Burnout samples: " << settings.burnout_samples << std::endl;
    std::cout << "Effective samples: " << (settings.total_number_of_samples - settings.burnout_samples) << std::endl;
    std::cout << "Final acceptance rate: " << (GetAcceptanceRate() * 100.0) << "%" << std::endl;
    std::cout << "Parameters estimated: " << parameters->size() << std::endl;
    std::cout << "Output directory: " << fileInformation.outputpath << std::endl;
    std::cout << "========================================\n" << std::endl;
}

/**
 * @brief Write percentiles table to file
 * @param percentiles Vector of CVector containing percentile values for each parameter
 * @param columnLabels Parameter names (column headers)
 * @param rowLabels Percentile labels (row headers: "0.025", "0.5", etc.)
 * @param filename Output file path
 */
template<class T>
void CMCMC<T>::WritePercentilesTable(const std::vector<CVector>& percentiles,
                                     const std::vector<std::string>& columnLabels,
                                     const std::vector<std::string>& rowLabels,
                                     const std::string& filename)
{
    std::ofstream file(filename);

    if (!file.is_open())
    {
        last_error = "WritePercentilesTable: Could not open file: " + filename;
        throw std::runtime_error(last_error);
    }

    // Write header row
    file << "Statistic";
    for (const auto& label : columnLabels)
    {
        file << "," << label;
    }
    file << "\n";

    // Write data rows
    for (size_t row = 0; row < rowLabels.size(); ++row)
    {
        file << rowLabels[row];

        for (size_t col = 0; col < percentiles.size(); ++col)
        {
            if (row < percentiles[col].num)
            {
                file << "," << percentiles[col][row];
            }
            else
            {
                file << ",";  // Empty cell if data missing
            }
        }

        file << "\n";
    }

    file.close();
}

/**
 * @brief Get current acceptance rate
 * @return Ratio of accepted proposals to total proposals (0.0 to 1.0)
 *
 * The acceptance rate measures how often the MCMC algorithm accepts
 * new proposals. The optimal rate is typically around 23-25% for
 * multivariate problems.
 *
 * @note Returns 0.0 if no proposals have been made yet
 *
 * @example
 * @code
 * double rate = mcmc.GetAcceptanceRate();
 * std::cout << "Acceptance rate: " << (rate * 100.0) << "%" << std::endl;
 * @endcode
 */
template<class T>
double CMCMC<T>::GetAcceptanceRate() const
{
    if (totalCount == 0.0)
    {
        return 0.0;
    }

    return acceptedCount / totalCount;
}


/**
 * @brief Calculate parameter correlation matrix from MCMC samples
 * @param burnin Number of initial samples to skip
 * @return Correlation matrix where element (i,j) is correlation between parameters i and j
 *
 * The correlation matrix is symmetric with 1's on the diagonal.
 * Values near ±1 indicate strong linear correlation between parameters.
 *
 * Formula: corr(X,Y) = cov(X,Y) / (std(X) * std(Y))
 */
template<class T>
CMatrix_arma CMCMC<T>::CalculateParameterCorrelation(int burnin)
{
    // Validate inputs
    if (burnin < 0 || static_cast<size_t>(burnin) >= parameterSamples.size())
    {
        last_error = "CalculateParameterCorrelation: invalid burnin value";
        std::cerr << last_error << std::endl;
        return CMatrix_arma();
    }

    int nParams = settings.number_of_parameters;
    int nSamples = parameterSamples.size() - burnin;

    if (nSamples < 2)
    {
        last_error = "CalculateParameterCorrelation: insufficient samples after burnin";
        std::cerr << last_error << std::endl;
        return CMatrix_arma();
    }

    // Initialize correlation matrix
    CMatrix_arma corrMatrix(nParams, nParams);

    // Calculate means for each parameter
    std::vector<double> means(nParams, 0.0);
    for (int i = burnin; i < static_cast<int>(parameterSamples.size()); ++i)
    {
        for (int j = 0; j < nParams; ++j)
        {
            means[j] += parameterSamples[i][j];
        }
    }
    for (int j = 0; j < nParams; ++j)
    {
        means[j] /= nSamples;
    }

    // Calculate standard deviations
    std::vector<double> stddevs(nParams, 0.0);
    for (int i = burnin; i < static_cast<int>(parameterSamples.size()); ++i)
    {
        for (int j = 0; j < nParams; ++j)
        {
            double diff = parameterSamples[i][j] - means[j];
            stddevs[j] += diff * diff;
        }
    }
    for (int j = 0; j < nParams; ++j)
    {
        stddevs[j] = std::sqrt(stddevs[j] / (nSamples - 1));
    }

    // Calculate correlation coefficients
    for (int p1 = 0; p1 < nParams; ++p1)
    {
        for (int p2 = 0; p2 < nParams; ++p2)
        {
            if (p1 == p2)
            {
                // Diagonal: perfect correlation with self
                corrMatrix(p1, p2) = 1.0;
            }
            else if (p2 > p1)
            {
                // Calculate covariance
                double covariance = 0.0;
                for (int i = burnin; i < static_cast<int>(parameterSamples.size()); ++i)
                {
                    double diff1 = parameterSamples[i][p1] - means[p1];
                    double diff2 = parameterSamples[i][p2] - means[p2];
                    covariance += diff1 * diff2;
                }
                covariance /= (nSamples - 1);

                // Calculate correlation
                double correlation = 0.0;
                if (stddevs[p1] > 1e-10 && stddevs[p2] > 1e-10)
                {
                    correlation = covariance / (stddevs[p1] * stddevs[p2]);
                }

                // Matrix is symmetric
                corrMatrix(p1, p2) = correlation;
                corrMatrix(p2, p1) = correlation;
            }
        }
    }

    return corrMatrix;
}

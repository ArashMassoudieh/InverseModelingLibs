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

#ifndef MCMC_H
#define MCMC_H

#include <vector>
#include <string>
#include "math.h"
#include <iostream>
#include "NormalDist.h"
#include "GA.h"
#include "Vector.h"
#include "TimeSeriesSet.h"

class Observation;
class ParameterSet;
class Object;

using namespace std;

// Forward declarations
class RunTimeWindow;
class Parameter_Set;
class Parameter;

/**
 * @brief File path configuration for MCMC output
 */
struct MCMCFileNames
{
    string outputpath;           ///< Directory path for output files
    string outputfilename;       ///< Main output file with samples
    string detailfilename;       ///< Detailed log file
};

/**
 * @brief Configuration settings for MCMC algorithm
 *
 * Contains all parameters that control the behavior of the
 * Markov Chain Monte Carlo sampling algorithm.
 */
struct MCMCSettings
{
    // Sample configuration
    unsigned int total_number_of_samples = 0;        ///< Total samples to generate
    unsigned int number_of_chains = 1;               ///< Number of parallel chains
    unsigned int burnout_samples = 0;                ///< Samples to discard at start

    // Perturbation settings
    double initial_perturbation_factor = 1.0;        ///< Initial perturbation magnitude
    double perturbation_factor = 0.05;               ///< Standard perturbation magnitude
    double perturbation_change_scale = 0.75;         ///< Scale factor for adaptive perturbation

    // Parameter configuration
    unsigned int number_of_parameters = 0;           ///< Number of parameters to estimate

    // Output configuration
    int save_interval = 1;                           ///< Save every nth sample
    string continue_filename;                        ///< File to continue from

    // Algorithm options
    bool no_initial_perturbation = false;            ///< Skip initial parameter perturbation
    bool sensitivity_based_perturbation = false;     ///< Use sensitivity for perturbation scaling
    bool global_sensitivity = false;                 ///< Calculate global sensitivity
    bool continue_mcmc = false;                      ///< Continue from previous run

    // Post-processing
    unsigned int number_of_post_estimate_realizations = 0;  ///< Realizations after sampling
    double increment_for_sensitivity = 0.01;         ///< Increment for sensitivity calculation
    bool noise_realization_writeout = false;         ///< Include noise in realizations

    // Performance
    unsigned int numberOfThreads = 8;                ///< Number of parallel threads
    double acceptance_rate = 0.234;                  ///< Target acceptance rate
};

/**
 * @brief Markov Chain Monte Carlo parameter estimation class
 *
 * @tparam T Model type that supports parameter estimation interface
 *
 * Template Requirements for T:
 * - SetParameterValue(int index, double value)
 * - ApplyParameters()
 * - Solve()
 * - GetObjectiveFunctionValue()
 * - SetSilent(bool)
 * - SetRecordResults(bool)
 * - SetNumThreads(int)
 *
 * This class implements Metropolis-Hastings MCMC for Bayesian parameter
 * estimation with adaptive proposal distributions and parallel chain execution.
 *
 * @example
 * @code
 * CMCMC<MyModel> mcmc(&model);
 * mcmc.SetProperty("number_of_samples", "10000");
 * mcmc.SetProperty("number_of_chains", "4");
 * mcmc.initialize(false);
 * mcmc.Perform();
 * @endcode
 */
template<class T>
class CMCMC
{
public:
    // ============================================================================
    // Constructors and Destructor
    // ============================================================================

    /**
     * @brief Default constructor
     */
    CMCMC();

    /**
     * @brief Constructor with model pointer
     * @param system Pointer to the model to be calibrated
     *
     * Initializes MCMC with the given model, sets up parameter references,
     * and configures default settings.
     */
    explicit CMCMC(T *system);

    /**
     * @brief Destructor
     *
     * Cleans up allocated memory for parameter samples and likelihoods.
     */
    ~CMCMC();

    // Copy constructor and assignment operator deleted to prevent issues with pointers
    CMCMC(const CMCMC&) = delete;
    CMCMC& operator=(const CMCMC&) = delete;

    // ============================================================================
    // Configuration
    // ============================================================================

    /**
     * @brief Set MCMC configuration from object properties
     * @param obj Object containing property-value pairs
     */
    void SetParameters(Object *obj);

    /**
     * @brief Set a single configuration property
     * @param varname Property name (case-insensitive)
     * @param value Property value as string
     * @return true if property was recognized and set, false otherwise
     *
     * Supported properties:
     * - number_of_samples: Total samples to generate
     * - number_of_chains: Parallel chains
     * - number_of_burnout_samples: Samples to discard
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
     *
     * If property not found, sets last_error message.
     */
    bool SetProperty(const string &varname, const string &value);

    /**
     * @brief Set the runtime window for progress updates
     * @param _rtw Pointer to runtime window (can be nullptr)
     */
    void SetRunTimeWindow(RunTimeWindow *_rtw);

    // ============================================================================
    // Initialization
    // ============================================================================

    /**
     * @brief Initialize MCMC chains with random or nominal parameters
     * @param random If true, randomize initial parameters within prior ranges
     *
     * Sets up parameter vectors, perturbation coefficients, and evaluates
     * initial posterior probabilities for all chains.
     */
    void initialize(bool random = false);

    /**
     * @brief Initialize MCMC chains from specific parameter values
     * @param par Initial parameter values to use
     *
     * Similar to initialize(bool) but starts from specified values.
     */
    void initialize(vector<double> par);

    // ============================================================================
    // Core MCMC Operations
    // ============================================================================

    /**
     * @brief Perform complete MCMC sampling
     *
     * Executes the full MCMC workflow:
     * 1. Initialize chains
     * 2. Generate samples with adaptive perturbation
     * 3. Create posterior distributions
     * 4. Generate realizations
     *
     * Uses settings from MCMCSettings and outputs to files specified
     * in FileInformation.
     */
    void Perform();

    // ============================================================================
    // Accessors
    // ============================================================================

    /**
     * @brief Get the last error message
     * @return Error message string (empty if no error)
     */
    string GetLastError() const { return last_error; }

    /**
     * @brief Get current acceptance rate
     * @return Ratio of accepted to total proposals
     */
    double GetAcceptanceRate() const;

    /**
     * @brief Get parameter samples
     * @return Reference to sample matrix [sample][parameter]
     */
    const vector<vector<double>>& GetParameterSamples() const { return parameterSamples; }

    /**
     * @brief Get log posterior values
     * @return Vector of log posterior probabilities
     */
    const vector<double>& GetLogPosterior() const { return logPosterior; }

    /**
     * @brief Get MCMC settings
     * @return Reference to current settings
     */
    const MCMCSettings& GetSettings() const { return settings; }

    /**
     * @brief Get file information
     * @return Reference to file paths
     */
    const MCMCFileNames& GetFileInformation() const { return fileInformation; }

    /**
     * @brief Access parameter by index
     * @param i Parameter index
     * @return Pointer to Parameter object
     */
    Parameter* GetParameter(int i);

    /**
     * @brief Access observation by index
     * @param i Observation index
     * @return Pointer to Observation object
     */
    Observation* GetObservation(int i);

    // ============================================================================
    // File I/O
    // ============================================================================

    /**
     * @brief Write current samples to output file
     * @param filename Output file path
     */
    void WriteOutput(string filename);

    /**
     * @brief Read previous MCMC run from file
     * @param filename Input file path
     * @return Number of samples read
     *
     * Loads parameter samples, likelihoods, and perturbation coefficients
     * from a previous run to continue sampling.
     */
    int ReadFromFile(string filename);

    // ============================================================================
    // Post-Processing
    // ============================================================================

    /**
     * @brief Generate model realizations from posterior samples
     * @param MCMCout TimeSeriesSet containing parameter samples
     *
     * Runs the model multiple times with randomly sampled parameters
     * from the posterior distribution to generate prediction intervals.
     */
    void ProduceRealizations(TimeSeriesSet<double> &MCMCout);

    /**
     * @brief Calculate output percentiles from realizations
     * @param MCMCout TimeSeriesSet containing parameter samples
     *
     * Computes percentile bands (e.g., 2.5%, 50%, 97.5%) for model outputs
     * based on posterior parameter distribution.
     */
    void GetOutputPercentiles(TimeSeriesSet<double> &MCMCout);

    /**
     * @brief Calculate prior distribution for visualization
     * @param n_bins Number of bins for discretization
     * @return TimeSeriesSet with prior distribution for each parameter
     */
    TimeSeriesSet<double> CalculatePriorDistribution(int n_bins);

    // ============================================================================
    // Sensitivity Analysis
    // ============================================================================

    /**
     * @brief Calculate parameter sensitivities
     * @param increment Finite difference increment (relative)
     * @param par Parameter values at which to evaluate
     * @return Vector of sensitivity values
     *
     * Uses finite differences to compute d(sqrt|posterior|)/d(parameter).
     */
    CVector CalculateSensitivity(double increment, vector<double> par);

    /**
     * @brief Calculate log-scale parameter sensitivities
     * @param increment Finite difference increment
     * @param par Parameter values at which to evaluate
     * @return Vector of log-scale sensitivities
     *
     * Computes parameter * d(posterior)/d(parameter) for log-scale interpretation.
     */
    CVector CalculateSensitivityLog(double increment, vector<double> par);

    // ============================================================================
    // Public Configuration (maintained for backward compatibility)
    // ============================================================================

    MCMCSettings settings;                   ///< MCMC algorithm settings
    MCMCFileNames fileInformation;           ///< File path configuration
    vector<double> outputPercentiles;        ///< Percentiles to calculate for outputs

    // Model output storage
    vector<vector<TimeSeriesSet<double>>> modelOutputObserved;
    vector<vector<TimeSeriesSet<double>>> modelOutputObservedNoise;
    vector<vector<TimeSeriesSet<double>>> modelOutputPercentilesObserved;
    vector<vector<TimeSeriesSet<double>>> modelOutputPercentilesObservedNoise;
    vector<CMatrix> globalSensitivityLumped;

    TimeSeriesSet<double> parameterList;
    TimeSeriesSet<double> realizedParameterList;

private:
    // ============================================================================
    // Private Core Methods
    // ============================================================================

    /**
     * @brief Perform single MCMC step for chain k
     * @param k Chain index
     * @return true if proposal was accepted, false if rejected
     *
     * Generates proposal, evaluates posterior, and performs Metropolis-Hastings
     * acceptance test.
     */
    bool PerformStep(int k);

    /**
     * @brief Perform MCMC steps with file output and progress updates
     * @param k Starting sample index
     * @param nsamps Number of samples to generate
     * @param filename Output file path
     * @param _rtw Runtime window for progress (can be nullptr)
     * @return Success status
     *
     * Main sampling loop with adaptive perturbation, periodic file writes,
     * and GUI updates if runtime window provided.
     */
    bool PerformSteps(int k, int nsamps, string filename, RunTimeWindow* _rtw = nullptr);

    /**
     * @brief Perturb parameters to generate proposal
     * @param k Current sample index
     * @return Perturbed parameter vector
     *
     * Adds normally distributed noise scaled by perturbation coefficients.
     */
    vector<double> PerturbParameters(int k);

    /**
     * @brief Calculate log posterior probability
     * @param par Parameter values
     * @param sample_number Sample index for logging
     * @param out If true, store full model output
     * @return Log posterior probability (log prior + log likelihood)
     *
     * Evaluates model at given parameters, computes log prior from parameter
     * distributions, and log likelihood from objective function.
     */
    double CalculateLogPosterior(vector<double> par, int sample_number, bool out = false);

    /**
     * @brief Run model with given parameters
     * @param par Parameter values
     * @return Model outputs as TimeSeriesSet
     */
    TimeSeriesSet<double> RunModel(vector<double> par);

    /**
     * @brief Run model and store in provided model object
     * @param Model1 Model object to run
     * @param par Parameter values to use
     */
    void RunModel(T *Model1, vector<double> par);

    // ============================================================================
    // Private Helper Methods
    // ============================================================================

    /**
     * @brief Initialize from model - common initialization code
     *
     * Extracted common initialization logic used by constructors.
     */
    void InitializeFromModel();

    // ============================================================================
    // Private Data Members
    // ============================================================================

    // Model and parameters
    T* model;                                ///< Pointer to model being calibrated
    T modelOutput;                           ///< Model copy for output storage
    Parameter_Set* parameters;               ///< Pointer to parameter set
    vector<Observation>* observations;       ///< Pointer to observations
    vector<int> parameterIndices;            ///< Indices of parameters to estimate

    // MCMC state
    vector<vector<double>> parameterSamples; ///< All parameter samples [sample][param]
    vector<double> perturbationCoefficients; ///< Current perturbation scales [param]
    vector<double> logPosterior;             ///< Log posterior for each sample
    vector<double> logPosteriorTransformed;  ///< Transformed log posterior
    vector<double> uniformRandoms;           ///< Pre-generated uniform randoms
    vector<bool> applyToAll;                 ///< Parameter application flags

    // Statistics
    double acceptedCount;                    ///< Number of accepted proposals
    double totalCount;                       ///< Total number of proposals

    // Utilities
    CNormalDist normalDistribution;          ///< Random number generator
    RunTimeWindow* runtimeWindow;            ///< Optional GUI window
    string last_error;                       ///< Last error message

    // Legacy compatibility data
    TimeSeriesSet<double> measuredData;      ///< Measured data storage
};

#include "MCMC.hpp"

#endif // MCMC_H

#include <iostream>
#include <vector>
#include <cmath>
#include <random>
#include "MCMC.h"
#include "parameter.h"
#include "observation.h"

/**
 * @brief Simple model to estimate mean and std deviation from data
 *
 * This model assumes data is normally distributed: x ~ N(μ, σ)
 * We'll estimate μ (mean) and σ (standard deviation) using MCMC
 */
class NormalDistributionModel
{
public:
    NormalDistributionModel()
    {
        SetupParameters();
        SetupObservations();
    }

    void SetupParameters()
    {
        // Parameter 1: Mean (μ)
        Parameter mu;
        mu.SetName("mu");
        mu.SetValue(0.0);  // Initial guess
        mu.SetRange(-10.0, 10.0);
        mu.SetPriorDistribution("uniform");
        parameters.AddParameter(mu);

        // Parameter 2: Standard Deviation (σ)
        Parameter sigma;
        sigma.SetName("sigma");
        sigma.SetValue(1.0);  // Initial guess
        sigma.SetRange(0.1, 10.0);
        sigma.SetPriorDistribution("uniform");
        parameters.AddParameter(sigma);
    }

    void SetupObservations()
    {
        // We'll add observations when we generate synthetic data
    }

    void GenerateSyntheticData(double true_mean, double true_std, int n_samples, unsigned int seed = 12345)
    {
        std::cout << "\nGenerating synthetic data..." << std::endl;
        std::cout << "  True mean: " << true_mean << std::endl;
        std::cout << "  True std: " << true_std << std::endl;
        std::cout << "  Number of samples: " << n_samples << std::endl;

        // Store true values for later comparison
        this->true_mean = true_mean;
        this->true_std = true_std;

        // Generate data from normal distribution
        std::mt19937 gen(seed);
        std::normal_distribution<double> dist(true_mean, true_std);

        data.clear();
        data.reserve(n_samples);

        for (int i = 0; i < n_samples; ++i)
        {
            data.push_back(dist(gen));
        }

        // Create one "observation" that represents all the data
        Observation obs;
        obs.SetName("data_likelihood");
        observations.clear();
        observations.push_back(obs);

        std::cout << "  Data generated successfully" << std::endl;
        std::cout << "  Sample mean: " << CalculateSampleMean() << std::endl;
        std::cout << "  Sample std: " << CalculateSampleStd() << std::endl;
    }

    double CalculateSampleMean() const
    {
        double sum = 0.0;
        for (double x : data) sum += x;
        return sum / data.size();
    }

    double CalculateSampleStd() const
    {
        double mean = CalculateSampleMean();
        double sum_sq = 0.0;
        for (double x : data) sum_sq += (x - mean) * (x - mean);
        return std::sqrt(sum_sq / (data.size() - 1));
    }

    void SetParameterValue(int index, double value)
    {
        if (index == 0) current_mu = value;
        else if (index == 1) current_sigma = value;
    }

    void ApplyParameters()
    {
        // Nothing to do - parameters are already set
    }

    void Solve()
    {
        // Calculate negative log-likelihood
        // For normal distribution: -log L = n/2 * log(2π) + n * log(σ) + Σ(x_i - μ)²/(2σ²)

        double n = data.size();
        double sigma_sq = current_sigma * current_sigma;

        // Sum of squared deviations
        double sum_sq_dev = 0.0;
        for (double x : data)
        {
            double dev = x - current_mu;
            sum_sq_dev += dev * dev;
        }

        // Negative log-likelihood (this is our "objective function")
        objective_value = n * std::log(current_sigma) + sum_sq_dev / (2.0 * sigma_sq);

        // Note: We omit the constant term n/2 * log(2π) as it doesn't affect optimization
    }

    double GetObjectiveFunctionValue() const
    {
        return objective_value;
    }

    void SetSilent(bool silent) { this->silent = silent; }
    void SetRecordResults(bool record) { this->record_results = record; }
    void SetNumThreads(int n) { this->num_threads = n; }
    bool GetSolutionFailed() const { return false; }
    double GetSimulationDuration() const { return 0.001; }

    Parameter_Set& Parameters() { return parameters; }
    std::vector<Observation>* Observations() { return &observations; }
    std::string OutputPath() const { return "./output/"; }

    TimeSeriesSet<double> GetPredictions() const
    {
        // For this simple model, we don't have time series predictions
        // Return empty set
        return TimeSeriesSet<double>();
    }

    // Getters for true values
    double GetTrueMean() const { return true_mean; }
    double GetTrueStd() const { return true_std; }

private:
    Parameter_Set parameters;
    std::vector<Observation> observations;
    std::vector<double> data;

    // Current parameter values
    double current_mu = 0.0;
    double current_sigma = 1.0;

    // True values (for comparison)
    double true_mean = 0.0;
    double true_std = 1.0;

    // Objective function value
    double objective_value = 0.0;

    // Settings
    bool silent = false;
    bool record_results = false;
    int num_threads = 1;
};



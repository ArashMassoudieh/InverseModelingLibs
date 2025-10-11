/*
 * Dummy model for testing MCMC and GA refactoring
 * Implementation file
 */

#include "polynomialmodel.h"
#include <cmath>
#include <chrono>
#include <stdexcept>

#ifdef GSL
#include <gsl/gsl_randist.h>
#include <gsl/gsl_rng.h>
#endif

// ============================================================================
// Constructor
// ============================================================================

PolynomialModel::PolynomialModel(int degree, const std::vector<double>& xValues,
                                 const std::vector<double>& yValues)
    : polynomialDegree(degree)
    , xData(xValues)
    , silent(false)
    , recordResults(true)
    , numThreads(1)
    , solutionFailed(false)
    , simulationDuration(0.0)
{
    // Create parameters (polynomial coefficients)
    for (int i = 0; i <= degree; ++i) {
        std::string name = "a" + std::to_string(i);
        Parameter param(name, -10.0, 10.0, "uniform", 0.0);
        parameters.AddParameter(param);
    }

    // Create observation
    Observation obs("y_values");
    TimeSeries<double> observedTS;
    for (size_t i = 0; i < xData.size(); ++i) {
        observedTS.append(xData[i], yValues[i]);
    }
    obs.SetObservedTimeSeries(observedTS);
    observations.push_back(obs);

    // Initialize outputs
    ObservedOutputs = TimeSeriesSet<double>(1);
}

// ============================================================================
// MCMC Interface Methods
// ============================================================================

void PolynomialModel::SetParameterValue(int index, double value) {
    if (index >= 0 && index < parameters.size()) {
        parameters[index]->SetValue(value);
    }
}

void PolynomialModel::ApplyParameters() {
    // Nothing special needed for this simple model
    // Parameters are already set via SetParameterValue
}

void PolynomialModel::Solve() {
    auto start = std::chrono::high_resolution_clock::now();

    solutionFailed = false;

    // Evaluate polynomial at each x point
    TimeSeries<double> modeledTS;
    for (double x : xData) {
        double y = EvaluatePolynomial(x);
        modeledTS.append(x, y);
    }

    // Store in observation
    observations[0].SetModeledTimeSeries(modeledTS);

    // Store in outputs
    ObservedOutputs = TimeSeriesSet<double>(1);
    ObservedOutputs.append(modeledTS, "y_predicted");

    auto end = std::chrono::high_resolution_clock::now();
    simulationDuration = std::chrono::duration<double>(end - start).count();
}

double PolynomialModel::GetObjectiveFunctionValue() const {
    return observations[0].CalculateSSE();
}

void PolynomialModel::SetSilent(bool s) {
    silent = s;
}

void PolynomialModel::SetRecordResults(bool r) {
    recordResults = r;
}

void PolynomialModel::SetNumThreads(int n) {
    numThreads = n;
}

bool PolynomialModel::GetSolutionFailed() const {
    return solutionFailed;
}

double PolynomialModel::GetSimulationDuration() const {
    return simulationDuration;
}

std::string PolynomialModel::OutputPath() const {
    return "./output/";
}

Parameter_Set& PolynomialModel::Parameters() {
    return parameters;
}

const Parameter_Set& PolynomialModel::Parameters() const {
    return parameters;
}

std::vector<Observation>* PolynomialModel::Observations() {
    return &observations;
}

const std::vector<Observation>* PolynomialModel::Observations() const {
    return &observations;
}

int PolynomialModel::ObservationsCount() const {
    return static_cast<int>(observations.size());
}

Observation* PolynomialModel::observation(int i) {
    if (i >= 0 && i < static_cast<int>(observations.size())) {
        return &observations[i];
    }
    return nullptr;
}

// ============================================================================
// GA Interface Methods
// ============================================================================

std::vector<std::string> PolynomialModel::GetParameterNames() const {
    std::vector<std::string> names;
    for (int i = 0; i < parameters.size(); ++i) {
        names.push_back(parameters[i]->GetName());
    }
    return names;
}

std::string PolynomialModel::GetParameterName(int i) const {
    if (i >= 0 && i < parameters.size()) {
        return parameters[i]->GetName();
    }
    return "";
}

double PolynomialModel::GetParameterValue(int i) const {
    if (i >= 0 && i < parameters.size()) {
        return parameters[i]->GetValue();
    }
    return 0.0;
}

bool PolynomialModel::IsParameterLogged(int i) const {
    if (i >= 0 && i < parameters.size()) {
        return parameters[i]->GetPriorDistribution() == "log-normal";
    }
    return false;
}

double PolynomialModel::GetParameterLow(int i) const {
    if (i >= 0 && i < parameters.size()) {
        return parameters[i]->GetRange().low;
    }
    return 0.0;
}

double PolynomialModel::GetParameterHigh(int i) const {
    if (i >= 0 && i < parameters.size()) {
        return parameters[i]->GetRange().high;
    }
    return 0.0;
}

int PolynomialModel::GetNumberOfParameters() const {
    return parameters.size();
}

// ============================================================================
// Private Methods
// ============================================================================

double PolynomialModel::EvaluatePolynomial(double x) const {
    double result = 0.0;
    double xPower = 1.0;  // x^0

    for (int i = 0; i <= polynomialDegree; ++i) {
        double coeff = parameters[i]->GetValue();
        result += coeff * xPower;
        xPower *= x;  // Next power of x
    }

    return result;
}

// ============================================================================
// Synthetic Data Generation
// ============================================================================

void PolynomialModel::GenerateSyntheticObservation(const std::vector<double>& trueCoefficients,
                                                   double x_start,
                                                   double x_end,
                                                   double interval,
                                                   double noiseStdDev) {
#ifdef GSL
    // Validate input
    if (trueCoefficients.size() != static_cast<size_t>(polynomialDegree + 1)) {
        throw std::invalid_argument("Number of coefficients must match polynomial degree + 1");
    }

    if (noiseStdDev < 0.0) {
        throw std::invalid_argument("Noise standard deviation must be non-negative");
    }

    if (x_end <= x_start) {
        throw std::invalid_argument("x_end must be greater than x_start");
    }

    if (interval <= 0.0) {
        throw std::invalid_argument("Interval must be positive");
    }

    // Initialize GSL random number generator
    const gsl_rng_type* T = gsl_rng_default;
    gsl_rng* r = gsl_rng_alloc(T);

    // Seed with current time
    gsl_rng_set(r, static_cast<unsigned long>(std::chrono::system_clock::now().time_since_epoch().count()));

    // Generate x values and synthetic observations
    TimeSeries<double> syntheticObserved;
    xData.clear();  // Clear old x data

    for (double x = x_start; x <= x_end; x += interval) {
        // Store x value
        xData.push_back(x);

        // Evaluate true polynomial
        double y_true = 0.0;
        double xPower = 1.0;

        for (size_t i = 0; i < trueCoefficients.size(); ++i) {
            y_true += trueCoefficients[i] * xPower;
            xPower *= x;
        }

        // Add Gaussian noise
        double noise = gsl_ran_gaussian(r, noiseStdDev);
        double y_observed = y_true + noise;

        syntheticObserved.append(x, y_observed);
    }

    // Update observation with synthetic data
    if (!observations.empty()) {
        observations[0].SetObservedTimeSeries(syntheticObserved);
    } else {
        // Create new observation if none exists
        Observation obs("y_values");
        obs.SetObservedTimeSeries(syntheticObserved);
        observations.push_back(obs);
    }

    // Clean up GSL
    gsl_rng_free(r);

#else
    throw std::runtime_error("GenerateSyntheticObservation requires GSL library. "
                             "Compile with -DGSL flag and link with -lgsl -lgslcblas");
#endif
}

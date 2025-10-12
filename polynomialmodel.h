/*
 * Dummy model for testing MCMC and GA refactoring
 * Implements a simple polynomial regression model
 */

#ifndef POLYNOMIAL_MODEL_H
#define POLYNOMIAL_MODEL_H

#include <vector>
#include <string>
#include "TimeSeries.h"
#include "TimeSeriesSet.h"
#include "parameter.h"
#include "parameter_set.h"
#include "observation.h"

/**
 * @brief Polynomial regression model for testing MCMC and GA
 *
 * Model: y = a0 + a1*x + a2*x^2 + ... + an*x^n
 *
 * This is a simple test model that implements all interfaces required
 * by both MCMC and GA optimization algorithms.
 */
class PolynomialModel
{
public:
    // ========================================================================
    // Constructors
    // ========================================================================

    /**
     * @brief Constructor
     * @param degree Polynomial degree (e.g., 2 for quadratic)
     * @param xValues X coordinates for observations
     * @param yValues Y coordinates (observed data)
     */
    PolynomialModel(int degree, const std::vector<double>& xValues,
                    const std::vector<double>& yValues);

    // ========================================================================
    // Required by MCMC
    // ========================================================================

    /**
     * @brief Set parameter value by index
     * @param index Parameter index
     * @param value New parameter value
     */
    void SetParameterValue(int index, double value);

    /**
     * @brief Apply parameters to model (called after setting parameters)
     */
    void ApplyParameters();

    /**
     * @brief Run model simulation
     */
    void Solve();

    /**
     * @brief Get objective function value (SSE)
     * @return Sum of squared errors
     */
    double GetObjectiveFunctionValue();

    /**
     * @brief Set silent mode (suppress output)
     */
    void SetSilent(bool s);

    /**
     * @brief Set whether to record results
     */
    void SetRecordResults(bool r);

    /**
     * @brief Set number of threads for parallel execution
     */
    void SetNumThreads(int n);

    /**
     * @brief Check if solution failed
     */
    bool GetSolutionFailed() const;

    /**
     * @brief Get simulation duration in seconds
     */
    double GetSimulationDuration() const;

    /**
     * @brief Get output path for results
     */
    std::string OutputPath() const;

    /**
     * @brief Get parameter set (mutable)
     */
    Parameter_Set& Parameters();

    /**
     * @brief Get parameter set (const)
     */
    const Parameter_Set& Parameters() const;

    /**
     * @brief Get observations vector (mutable)
     */
    std::vector<Observation>* Observations();

    /**
     * @brief Get observations vector (const)
     */
    const std::vector<Observation>* Observations() const;

    /**
     * @brief Get number of observations
     */
    int ObservationsCount() const;

    /**
     * @brief Get observation by index
     */
    Observation* observation(int i);

    // ========================================================================
    // Required by GA
    // ========================================================================

    /**
     * @brief Get all parameter names
     */
    std::vector<std::string> GetParameterNames() const;

    /**
     * @brief Get parameter name by index
     */
    std::string GetParameterName(int i) const;

    /**
     * @brief Get parameter value by index
     */
    double GetParameterValue(int i) const;

    /**
     * @brief Check if parameter uses log scale
     */
    bool IsParameterLogged(int i) const;

    /**
     * @brief Get parameter lower bound
     */
    double GetParameterLow(int i) const;

    /**
     * @brief Get parameter upper bound
     */
    double GetParameterHigh(int i) const;

    /**
     * @brief Get total number of parameters
     */
    int GetNumberOfParameters() const;

    // ========================================================================
    // Synthetic Data Generation
    // ========================================================================

    /**
     * @brief Generate synthetic observation with noise
     * @param trueCoefficients True polynomial coefficients
     * @param x_start Starting x value
     * @param x_end Ending x value
     * @param interval Spacing between x values
     * @param noiseStdDev Standard deviation of Gaussian noise to add
     *
     * Generates x values from x_start to x_end with given interval,
     * evaluates the polynomial with given true coefficients at each x point,
     * adds Gaussian noise with specified standard deviation, and stores as
     * the observed data. Useful for testing calibration algorithms.
     *
     * @note Requires GSL (GNU Scientific Library) to be installed
     *
     * @example
     * @code
     * std::vector<double> trueCoeffs = {2.0, 3.0, -0.5};  // y = 2 + 3x - 0.5x^2
     * model.GenerateSyntheticObservation(trueCoeffs, 0.0, 10.0, 0.5, 0.3);
     * // Generates data from x=0 to x=10 with dx=0.5, noise σ=0.3
     * @endcode
     */
    void GenerateSyntheticObservation(const std::vector<double>& trueCoefficients,
                                      double x_start,
                                      double x_end,
                                      double interval,
                                      double noiseStdDev);

    // ========================================================================
    // Public Data Members (for output)
    // ========================================================================

    TimeSeriesSet<double> Outputs;          ///< General model outputs
    TimeSeriesSet<double> ObservedOutputs;  ///< Outputs at observation points

private:
    // ========================================================================
    // Private Methods
    // ========================================================================

    /**
     * @brief Evaluate polynomial at given x value
     * @param x Input value
     * @return Polynomial value at x
     */
    double EvaluatePolynomial(double x) const;

    // ========================================================================
    // Private Data Members
    // ========================================================================

    int polynomialDegree;                   ///< Degree of polynomial
    std::vector<double> xData;              ///< X coordinates of observations

    Parameter_Set parameters;               ///< Model parameters (coefficients)
    std::vector<Observation> observations;  ///< Observations for calibration

    bool silent;                            ///< Suppress output
    bool recordResults;                     ///< Record detailed results
    int numThreads;                         ///< Number of parallel threads
    bool solutionFailed;                    ///< Did simulation fail
    double simulationDuration;              ///< Time to run simulation (seconds)
};

#endif // POLYNOMIAL_MODEL_H

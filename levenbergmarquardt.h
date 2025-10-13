// levenberg_marquardt.h

#ifndef LEVENBERG_MARQUARDT_H
#define LEVENBERG_MARQUARDT_H

#include <vector>
#include <string>
#include <functional>
#include "Matrix_arma.h"
#include "Vector_arma.h"

/**
 * @struct LMParameters
 * @brief Configuration parameters for Levenberg-Marquardt optimization
 */
struct LMParameters
{
    int maxIterations = 100;           ///< Maximum number of iterations
    double tolerance = 1e-6;           ///< Convergence tolerance on gradient
    double paramTolerance = 1e-6;      ///< Convergence tolerance on parameters
    double objectiveTolerance = 1e-6;  ///< Convergence tolerance on objective
    double lambda = 0.01;              ///< Initial damping parameter
    double lambdaIncrease = 10.0;      ///< Factor to increase lambda on failure
    double lambdaDecrease = 10.0;      ///< Factor to decrease lambda on success
    double minLambda = 1e-10;          ///< Minimum lambda value
    double maxLambda = 1e10;           ///< Maximum lambda value
    double finiteDiffStep = 1e-6;      ///< Step size for finite differences
    bool verbose = true;               ///< Print progress information
};

/**
 * @struct LMResult
 * @brief Results from Levenberg-Marquardt optimization
 */
struct LMResult
{
    std::vector<double> parameters;    ///< Optimized parameters
    double objectiveValue;             ///< Final objective function value
    int iterations;                    ///< Number of iterations performed
    bool converged;                    ///< Did the algorithm converge?
    std::string message;               ///< Convergence/termination message
    std::vector<double> residuals;     ///< Final residuals
    CMatrix_arma covariance;           ///< Parameter covariance matrix
    std::vector<double> standardErrors;///< Parameter standard errors
};

/**
 * @class LevenbergMarquardt
 * @brief Levenberg-Marquardt nonlinear least squares optimizer
 * @tparam T Model type that provides residual evaluation
 *
 * Implements the Levenberg-Marquardt algorithm for nonlinear least squares
 * optimization. The algorithm interpolates between gradient descent (for
 * far-from-optimum) and Gauss-Newton (near optimum).
 *
 * The template parameter T must provide:
 * - Parameters() method returning parameter list
 * - SetParameterValue(int, double) method
 * - ApplyParameters() method
 * - Observations() or similar method to get observations
 * - GetObjectiveFunctionValue() method (optional, for monitoring)
 *
 * The algorithm minimizes: sum((y_observed - y_model)^2)
 */
template<class T>
class LevenbergMarquardt
{
public:
    // ========================================================================
    // Constructors
    // ========================================================================

    /**
     * @brief Default constructor
     */
    LevenbergMarquardt();

    /**
     * @brief Construct with model pointer
     * @param model Pointer to model instance
     */
    explicit LevenbergMarquardt(T* model);

    /**
     * @brief Destructor
     */
    ~LevenbergMarquardt();

    // ========================================================================
    // Main API
    // ========================================================================

    /**
     * @brief Run the Levenberg-Marquardt optimization
     * @return LMResult structure with optimization results
     */
    LMResult Optimize();

    /**
     * @brief Set optimization parameter by name
     * @param name Parameter name
     * @param value Parameter value as string
     * @return true if parameter was set, false if not recognized
     */
    bool SetParameter(const std::string& name, const std::string& value);

    /**
     * @brief Get last error message
     * @return Error message string
     */
    std::string GetLastError() const { return lastError; }

    /**
     * @brief Get optimization parameters structure
     * @return Reference to LMParameters
     */
    LMParameters& GetParameters() { return params; }

private:
    // ========================================================================
    // Private Members
    // ========================================================================

    T* model;                          ///< Pointer to model
    LMParameters params;               ///< Optimization parameters
    std::string lastError;             ///< Last error message

    int numParameters;                 ///< Number of parameters to optimize
    int numResiduals;                  ///< Number of residuals (observations)

    // ========================================================================
    // Private Methods - Core Algorithm
    // ========================================================================

    /**
     * @brief Compute residuals at current parameter values
     * @return Vector of residuals (observed - modeled)
     */
    CVector_arma ComputeResiduals();

    /**
     * @brief Compute Jacobian matrix using finite differences
     * @param currentResiduals Current residual vector
     * @return Jacobian matrix (numResiduals × numParameters)
     */
    CMatrix_arma ComputeJacobian(const CVector_arma& currentResiduals);

    /**
     * @brief Solve the Levenberg-Marquardt linear system
     * @param jacobian Jacobian matrix
     * @param residuals Residual vector
     * @param lambda Damping parameter
     * @return Parameter update vector Δp
     */
    CVector_arma SolveLinearSystem(const CMatrix_arma& jacobian,
                                   const CVector_arma& residuals,
                                   double lambda);

    /**
     * @brief Apply parameter update to model
     * @param update Parameter update vector
     */
    void ApplyParameterUpdate(const CVector_arma& update);

    /**
     * @brief Get current parameter values from model
     * @return Vector of current parameters
     */
    CVector_arma GetCurrentParameters();

    /**
     * @brief Set model parameters
     * @param parameters Parameter vector
     */
    void SetModelParameters(const CVector_arma& parameters);

    /**
     * @brief Calculate sum of squared residuals
     * @param residuals Residual vector
     * @return SSE = sum(residuals^2)
     */
    double CalculateSSE(const CVector_arma& residuals);

    /**
     * @brief Compute parameter covariance matrix
     * @param jacobian Jacobian at solution
     * @param sse Sum of squared errors at solution
     * @return Covariance matrix
     */
    CMatrix_arma ComputeCovariance(const CMatrix_arma& jacobian, double sse);

    /**
     * @brief Check convergence criteria
     * @param gradientNorm Norm of gradient
     * @param paramChange Norm of parameter change
     * @param objectiveChange Change in objective function
     * @param message Output: convergence message
     * @return true if converged
     */
    bool CheckConvergence(double gradientNorm,
                          double paramChange,
                          double objectiveChange,
                          std::string& message);
};

#include "levenbergmarquardt.hpp"

#endif // LEVENBERG_MARQUARDT_H

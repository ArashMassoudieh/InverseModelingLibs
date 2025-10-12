// levenberg_marquardt.hpp

#include <cmath>
#include <iostream>
#include <iomanip>
#include <limits>
#include "Utilities.h"
#include "observation.h"

// ============================================================================
// Constructors
// ============================================================================

template<class T>
LevenbergMarquardt<T>::LevenbergMarquardt()
    : model(nullptr)
    , numParameters(0)
    , numResiduals(0)
{
}

template<class T>
LevenbergMarquardt<T>::LevenbergMarquardt(T* model)
    : model(model)
    , numParameters(0)
    , numResiduals(0)
{
    if (!model)
    {
        lastError = "Model pointer is null";
        return;
    }

    // Get number of parameters
    numParameters = static_cast<int>(model->Parameters().size());

    // Get number of residuals (observations)
    if (model->Observations() && !model->Observations()->empty())
    {
        // Count total number of observation points
        numResiduals = 0;
        for (size_t i = 0; i < model->Observations()->size(); ++i)
        {
            const Observation* obs = &(*model->Observations())[i];
            if (obs)
            {
                numResiduals += obs->GetObservedData().size();
            }
        }
    }
}

template<class T>
LevenbergMarquardt<T>::~LevenbergMarquardt()
{
}

// ============================================================================
// Main Optimization Method
// ============================================================================

template<class T>
LMResult LevenbergMarquardt<T>::Optimize()
{
    LMResult result;
    result.converged = false;
    result.iterations = 0;
    result.objectiveValue = std::numeric_limits<double>::max();

    // Validate model
    if (!model)
    {
        lastError = "Model pointer is null";
        result.message = lastError;
        return result;
    }

    if (numParameters == 0)
    {
        lastError = "No parameters to optimize";
        result.message = lastError;
        return result;
    }

    if (numResiduals == 0)
    {
        lastError = "No observations available";
        result.message = lastError;
        return result;
    }

    if (params.verbose)
    {
        std::cout << "\n=== Levenberg-Marquardt Optimization ===" << std::endl;
        std::cout << "Parameters: " << numParameters << std::endl;
        std::cout << "Observations: " << numResiduals << std::endl;
        std::cout << "Max iterations: " << params.maxIterations << std::endl;
        std::cout << std::endl;
        std::cout << std::setw(6) << "Iter"
                  << std::setw(15) << "SSE"
                  << std::setw(15) << "Lambda"
                  << std::setw(15) << "||grad||"
                  << std::setw(15) << "||Δp||"
                  << std::endl;
        std::cout << std::string(66, '-') << std::endl;
    }

    // Get initial parameters
    CVector_arma currentParams = GetCurrentParameters();

    // Compute initial residuals and objective
    CVector_arma currentResiduals = ComputeResiduals();
    double currentSSE = CalculateSSE(currentResiduals);

    // Initialize damping parameter
    double lambda = params.lambda;

    // Main optimization loop
    for (int iter = 0; iter < params.maxIterations; ++iter)
    {
        result.iterations = iter + 1;

        // Compute Jacobian
        CMatrix_arma jacobian = ComputeJacobian(currentResiduals);

        // Compute gradient: g = -J^T * r
        CMatrix_arma JT = Transpose(jacobian);
        CVector_arma gradient = JT * currentResiduals * (-1.0);
        double gradientNorm = gradient.norm2();

        // Solve for parameter update
        CVector_arma paramUpdate = SolveLinearSystem(jacobian, currentResiduals, lambda);
        double paramUpdateNorm = paramUpdate.norm2();

        // Try the parameter update
        CVector_arma trialParams = currentParams + paramUpdate;
        SetModelParameters(trialParams);
        CVector_arma trialResiduals = ComputeResiduals();
        double trialSSE = CalculateSSE(trialResiduals);

        double objectiveChange = std::abs(currentSSE - trialSSE);

        // Check if we improved
        if (trialSSE < currentSSE)
        {
            // Accept the update
            currentParams = trialParams;
            currentResiduals = trialResiduals;
            currentSSE = trialSSE;

            // Decrease damping parameter (move toward Gauss-Newton)
            lambda = std::max(lambda / params.lambdaDecrease, params.minLambda);

            if (params.verbose)
            {
                std::cout << std::setw(6) << iter
                          << std::setw(15) << std::scientific << std::setprecision(6) << currentSSE
                          << std::setw(15) << std::scientific << lambda
                          << std::setw(15) << std::scientific << gradientNorm
                          << std::setw(15) << std::scientific << paramUpdateNorm
                          << " ✓" << std::endl;
            }

            // Check convergence
            if (CheckConvergence(gradientNorm, paramUpdateNorm, objectiveChange, result.message))
            {
                result.converged = true;
                break;
            }
        }
        else
        {
            // Reject the update, increase damping (move toward gradient descent)
            lambda = std::min(lambda * params.lambdaIncrease, params.maxLambda);
            SetModelParameters(currentParams); // Restore previous parameters

            if (params.verbose)
            {
                std::cout << std::setw(6) << iter
                          << std::setw(15) << std::scientific << currentSSE
                          << std::setw(15) << std::scientific << lambda
                          << std::setw(15) << std::scientific << gradientNorm
                          << std::setw(15) << std::scientific << paramUpdateNorm
                          << " ✗" << std::endl;
            }

            // Check if lambda is too large
            if (lambda >= params.maxLambda)
            {
                result.message = "Lambda reached maximum value - unable to make progress";
                break;
            }
        }
    }

    // Finalize results
    result.parameters.resize(numParameters);
    for (int i = 0; i < numParameters; ++i)
    {
        result.parameters[i] = currentParams[i];
    }

    result.residuals.resize(numResiduals);
    for (int i = 0; i < numResiduals; ++i)
    {
        result.residuals[i] = currentResiduals[i];
    }

    result.objectiveValue = currentSSE;

    // Compute covariance and standard errors
    CMatrix_arma finalJacobian = ComputeJacobian(currentResiduals);
    result.covariance = ComputeCovariance(finalJacobian, currentSSE);

    result.standardErrors.resize(numParameters);
    for (int i = 0; i < numParameters; ++i)
    {
        result.standardErrors[i] = std::sqrt(result.covariance(i, i));
    }

    if (!result.converged && result.iterations >= params.maxIterations)
    {
        result.message = "Maximum iterations reached without convergence";
    }

    if (params.verbose)
    {
        std::cout << std::string(66, '-') << std::endl;
        std::cout << result.message << std::endl;
        std::cout << "Final SSE: " << result.objectiveValue << std::endl;
        std::cout << "\nOptimized parameters:" << std::endl;
        for (int i = 0; i < numParameters; ++i)
        {
            std::cout << "  " << model->Parameters()[i]->GetName()
            << " = " << result.parameters[i]
            << " ± " << result.standardErrors[i] << std::endl;
        }
        std::cout << std::endl;
    }

    return result;
}

// ============================================================================
// Private Methods - Core Algorithm
// ============================================================================

template<class T>
CVector_arma LevenbergMarquardt<T>::ComputeResiduals()
{
    CVector_arma residuals(numResiduals);

    int residualIndex = 0;

    // Apply current parameters and solve model
    model->ApplyParameters();
    model->Solve();

    // Collect residuals from all observations
    for (size_t obsIdx = 0; obsIdx < model->Observations()->size(); ++obsIdx)
    {
        const Observation* obs = &(*model->Observations())[obsIdx];
        if (!obs) continue;

        const TimeSeries<double>& observed = obs->GetObservedData();
        const TimeSeries<double>* modeled = obs->GetModeledTimeSeries();

        if (!modeled) continue;

        int nPoints = std::min(observed.size(), modeled->size());

        for (int i = 0; i < nPoints; ++i)
        {
            double residual = observed.getValue(i) - modeled->getValue(i);
            residuals[residualIndex++] = residual;
        }
    }

    return residuals;
}

template<class T>
CMatrix_arma LevenbergMarquardt<T>::ComputeJacobian(const CVector_arma& currentResiduals)
{
    CMatrix_arma jacobian(numResiduals, numParameters);

    // Get current parameter values
    CVector_arma currentParams = GetCurrentParameters();

    // Compute Jacobian using forward finite differences
    for (int j = 0; j < numParameters; ++j)
    {
        // Perturb parameter j
        double originalValue = currentParams[j];
        double h = params.finiteDiffStep * std::max(std::abs(originalValue), 1.0);

        currentParams[j] = originalValue + h;
        SetModelParameters(currentParams);

        // Compute perturbed residuals
        CVector_arma perturbedResiduals = ComputeResiduals();

        // Compute derivative: ∂r/∂p_j ≈ (r(p+h) - r(p)) / h
        for (int i = 0; i < numResiduals; ++i)
        {
            jacobian(i, j) = (perturbedResiduals[i] - currentResiduals[i]) / h;
        }

        // Restore parameter
        currentParams[j] = originalValue;
    }

    // Restore original parameters
    SetModelParameters(currentParams);

    return jacobian;
}

template<class T>
CVector_arma LevenbergMarquardt<T>::SolveLinearSystem(const CMatrix_arma& jacobian,
                                                      const CVector_arma& residuals,
                                                      double lambda)
{
    // Compute J^T * J
    CMatrix_arma JT = Transpose(const_cast<CMatrix_arma&>(jacobian));
    CMatrix_arma JTJ = JT * jacobian;

    // Add damping: (J^T*J + λ*diag(J^T*J))
    for (int i = 0; i < numParameters; ++i)
    {
        JTJ(i, i) *= (1.0 + lambda);
    }

    // Compute right-hand side: -J^T * r
    CVector_arma rhs = JT * residuals * (-1.0);

    // Solve: (J^T*J + λ*diag(J^T*J)) * Δp = -J^T * r
    CVector_arma paramUpdate = rhs / JTJ;

    return paramUpdate;
}

template<class T>
CVector_arma LevenbergMarquardt<T>::GetCurrentParameters()
{
    CVector_arma params(numParameters);

    for (int i = 0; i < numParameters; ++i)
    {
        params[i] = model->Parameters()[i]->GetValue();
    }

    return params;
}

template<class T>
void LevenbergMarquardt<T>::SetModelParameters(const CVector_arma& parameters)
{
    for (int i = 0; i < numParameters; ++i)
    {
        model->SetParameterValue(i, parameters[i]);
    }
}

template<class T>
void LevenbergMarquardt<T>::ApplyParameterUpdate(const CVector_arma& update)
{
    for (int i = 0; i < numParameters; ++i)
    {
        double currentValue = model->Parameters()[i]->GetValue();
        model->SetParameterValue(i, currentValue + update[i]);
    }
}

template<class T>
double LevenbergMarquardt<T>::CalculateSSE(const CVector_arma& residuals)
{
    double sse = 0.0;
    for (int i = 0; i < residuals.num(); ++i)
    {
        sse += residuals[i] * residuals[i];
    }
    return sse;
}

template<class T>
CMatrix_arma LevenbergMarquardt<T>::ComputeCovariance(const CMatrix_arma& jacobian, double sse)
{
    // Compute J^T * J
    CMatrix_arma JT = Transpose(const_cast<CMatrix_arma&>(jacobian));
    CMatrix_arma JTJ = JT * jacobian;

    // Compute (J^T*J)^(-1)
    CMatrix_arma JTJ_inv = inv(JTJ);

    // Scale by residual variance: s^2 = SSE / (n - p)
    int degreesOfFreedom = numResiduals - numParameters;
    if (degreesOfFreedom <= 0)
        degreesOfFreedom = 1;

    double variance = sse / static_cast<double>(degreesOfFreedom);

    // Covariance = variance * (J^T*J)^(-1)
    CMatrix_arma covariance = variance*JTJ_inv;

    return covariance;
}

template<class T>
bool LevenbergMarquardt<T>::CheckConvergence(double gradientNorm,
                                             double paramChange,
                                             double objectiveChange,
                                             std::string& message)
{
    if (gradientNorm < params.tolerance)
    {
        message = "Converged: gradient norm below tolerance";
        return true;
    }

    if (paramChange < params.paramTolerance)
    {
        message = "Converged: parameter change below tolerance";
        return true;
    }

    if (objectiveChange < params.objectiveTolerance)
    {
        message = "Converged: objective function change below tolerance";
        return true;
    }

    return false;
}

// ============================================================================
// Parameter Setting
// ============================================================================

template<class T>
bool LevenbergMarquardt<T>::SetParameter(const std::string& name, const std::string& value)
{
    std::string lowerName = aquiutils::tolower(name);

    if (lowerName == "max_iterations")
    {
        params.maxIterations = aquiutils::atoi(value);
        return true;
    }
    else if (lowerName == "tolerance")
    {
        params.tolerance = aquiutils::atof(value);
        return true;
    }
    else if (lowerName == "param_tolerance")
    {
        params.paramTolerance = aquiutils::atof(value);
        return true;
    }
    else if (lowerName == "objective_tolerance")
    {
        params.objectiveTolerance = aquiutils::atof(value);
        return true;
    }
    else if (lowerName == "lambda")
    {
        params.lambda = aquiutils::atof(value);
        return true;
    }
    else if (lowerName == "lambda_increase")
    {
        params.lambdaIncrease = aquiutils::atof(value);
        return true;
    }
    else if (lowerName == "lambda_decrease")
    {
        params.lambdaDecrease = aquiutils::atof(value);
        return true;
    }
    else if (lowerName == "finite_diff_step")
    {
        params.finiteDiffStep = aquiutils::atof(value);
        return true;
    }
    else if (lowerName == "verbose")
    {
        std::string lowerValue = aquiutils::tolower(value);
        params.verbose = (lowerValue == "true" || lowerValue == "1" || lowerValue == "yes");
        return true;
    }

    lastError = "Unknown parameter: " + name;
    return false;
}

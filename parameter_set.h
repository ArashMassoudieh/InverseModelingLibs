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

#ifndef PARAMETER_SET_H
#define PARAMETER_SET_H

#include <vector>
#include "parameter.h"

/**
 * @brief Collection of Parameter objects
 *
 * Manages a set of parameters for model calibration. Provides array-like
 * access to individual parameters and basic collection management operations.
 *
 * @example
 * @code
 * Parameter_Set params;
 * params.AddParameter(Parameter("k", 0.001, 1.0, "log-normal"));
 * params.AddParameter(Parameter("n", 0.1, 0.5, "uniform"));
 *
 * // Access by index
 * Parameter* p = params[0];
 * std::cout << p->GetName() << ": " << p->GetValue() << std::endl;
 * @endcode
 */
class Parameter_Set
{
public:
    // ========================================================================
    // Constructors
    // ========================================================================

    /**
     * @brief Default constructor
     *
     * Creates an empty parameter set.
     */
    Parameter_Set();

    /**
     * @brief Constructor with initial size
     * @param n Number of default parameters to create
     *
     * Creates n default-constructed parameters.
     */
    explicit Parameter_Set(int n);

    // ========================================================================
    // Parameter Management
    // ========================================================================

    /**
     * @brief Add a parameter to the set
     * @param param Parameter to add
     *
     * Appends the parameter to the end of the collection.
     */
    void AddParameter(const Parameter& param);

    /**
     * @brief Get number of parameters in the set
     * @return Number of parameters
     */
    int size() const;

    /**
     * @brief Check if parameter set is empty
     * @return true if no parameters, false otherwise
     */
    bool empty() const;

    /**
     * @brief Clear all parameters from the set
     */
    void clear();

    /**
     * @brief Reserve space for n parameters
     * @param n Number of parameters to reserve space for
     *
     * Useful for efficiency when adding many parameters.
     */
    void reserve(int n);

    // ========================================================================
    // Access Operators
    // ========================================================================

    /**
     * @brief Access parameter by index (mutable)
     * @param i Parameter index (0-based)
     * @return Pointer to Parameter, or nullptr if index out of range
     *
     * @note Returns nullptr instead of throwing for out-of-range access
     *
     * @example
     * @code
     * Parameter* p = params[0];
     * if (p) {
     *     p->SetValue(1.5);
     * }
     * @endcode
     */
    Parameter* operator[](int i);

    /**
     * @brief Access parameter by index (const)
     * @param i Parameter index (0-based)
     * @return Const pointer to Parameter, or nullptr if index out of range
     *
     * @note Returns nullptr instead of throwing for out-of-range access
     */
    const Parameter* operator[](int i) const;

    /**
     * @brief Get iterator to beginning
     * @return Iterator to first parameter
     */
    std::vector<Parameter>::iterator begin();

    /**
     * @brief Get iterator to end
     * @return Iterator past last parameter
     */
    std::vector<Parameter>::iterator end();

    /**
     * @brief Get const iterator to beginning
     * @return Const iterator to first parameter
     */
    std::vector<Parameter>::const_iterator begin() const;

    /**
     * @brief Get const iterator to end
     * @return Const iterator past last parameter
     */
    std::vector<Parameter>::const_iterator end() const;

    // ========================================================================
    // Utility Methods
    // ========================================================================

    /**
     * @brief Find parameter by name
     * @param name Parameter name to search for
     * @return Pointer to Parameter if found, nullptr otherwise
     */
    Parameter* FindByName(const std::string& name);

    /**
     * @brief Find parameter by name (const)
     * @param name Parameter name to search for
     * @return Const pointer to Parameter if found, nullptr otherwise
     */
    const Parameter* FindByName(const std::string& name) const;

    /**
     * @brief Get index of parameter by name
     * @param name Parameter name to search for
     * @return Index of parameter, or -1 if not found
     */
    int GetIndexByName(const std::string& name) const;

    /**
     * @brief Access parameter by name (mutable)
     * @param name Parameter name to search for
     * @return Pointer to Parameter, or nullptr if not found
     *
     * @example
     * @code
     * Parameter* k = params["conductivity"];
     * if (k) {
     *     k->SetValue(1.5);
     * }
     * @endcode
     */
    Parameter* operator[](const std::string& name);

    /**
     * @brief Access parameter by name (const)
     * @param name Parameter name to search for
     * @return Const pointer to Parameter, or nullptr if not found
     */
    const Parameter* operator[](const std::string& name) const;

private:
    // ========================================================================
    // Private Data Members
    // ========================================================================

    std::vector<Parameter> parameters;  ///< Storage for parameters
};

#endif // PARAMETER_SET_H

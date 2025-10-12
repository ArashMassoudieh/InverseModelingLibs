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

#include "parameter_set.h"

// ============================================================================
// Constructors
// ============================================================================

Parameter_Set::Parameter_Set() {
    // Empty parameter set
}

Parameter_Set::Parameter_Set(int n) {
    parameters.resize(n);
}

// ============================================================================
// Parameter Management
// ============================================================================

void Parameter_Set::AddParameter(const Parameter& param) {
    parameters.push_back(param);
}

int Parameter_Set::size() const {
    return static_cast<int>(parameters.size());
}

bool Parameter_Set::empty() const {
    return parameters.empty();
}

void Parameter_Set::clear() {
    parameters.clear();
}

void Parameter_Set::reserve(int n) {
    if (n > 0) {
        parameters.reserve(static_cast<size_t>(n));
    }
}

// ============================================================================
// Access Operators
// ============================================================================

Parameter* Parameter_Set::operator[](int i) {
    if (i >= 0 && i < static_cast<int>(parameters.size())) {
        return &parameters[i];
    }
    return nullptr;
}

const Parameter* Parameter_Set::operator[](int i) const {
    if (i >= 0 && i < static_cast<int>(parameters.size())) {
        return &parameters[i];
    }
    return nullptr;
}

std::vector<Parameter>::iterator Parameter_Set::begin() {
    return parameters.begin();
}

std::vector<Parameter>::iterator Parameter_Set::end() {
    return parameters.end();
}

std::vector<Parameter>::const_iterator Parameter_Set::begin() const {
    return parameters.begin();
}

std::vector<Parameter>::const_iterator Parameter_Set::end() const {
    return parameters.end();
}

// ============================================================================
// Utility Methods
// ============================================================================

Parameter* Parameter_Set::FindByName(const std::string& name) {
    for (auto& param : parameters) {
        if (param.GetName() == name) {
            return &param;
        }
    }
    return nullptr;
}

const Parameter* Parameter_Set::FindByName(const std::string& name) const {
    for (const auto& param : parameters) {
        if (param.GetName() == name) {
            return &param;
        }
    }
    return nullptr;
}

int Parameter_Set::GetIndexByName(const std::string& name) const {
    for (int i = 0; i < static_cast<int>(parameters.size()); ++i) {
        if (parameters[i].GetName() == name) {
            return i;
        }
    }
    return -1;  // Not found
}

Parameter* Parameter_Set::operator[](const std::string& name) {
    for (auto& param : parameters) {
        if (param.GetName() == name) {
            return &param;
        }
    }
    return nullptr;  // Not found
}

const Parameter* Parameter_Set::operator[](const std::string& name) const {
    for (const auto& param : parameters) {
        if (param.GetName() == name) {
            return &param;
        }
    }
    return nullptr;  // Not found
}

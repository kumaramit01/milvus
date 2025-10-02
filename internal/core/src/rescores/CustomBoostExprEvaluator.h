// Licensed to the LF AI & Data foundation under one
// or more contributor license agreements. See the NOTICE file
// distributed with this work for additional information
// regarding copyright ownership. The ASF licenses this file
// to you under the Apache License, Version 2.0 (the
// "License"); you may not use this file except in compliance
// with the License. You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#pragma once

#include <cmath>
#include <string>
#include <unordered_map>
#include <memory>
#include <stdexcept>
#include "common/EasyAssert.h"
#include "fmt/core.h"

namespace milvus::rescores {

// Simple expression evaluator for custom boost mode
// Supports variables: original_score, boost_score
// Supports operators: +, -, *, /, %, **, >, <, >=, <=, ==, !=, &&, ||, !
// Supports functions: min, max, abs, sqrt, log, exp, pow, ceil, floor, round, clamp
// Supports ternary: condition ? true_val : false_val
class CustomBoostExprEvaluator {
 public:
    explicit CustomBoostExprEvaluator(const std::string& expression)
        : expr_(expression), pos_(0) {
        // Precompile and validate expression
        // For now, evaluate it at runtime
    }

    float
    evaluate(float original_score, float boost_score) const {
        // Create a mutable copy for evaluation
        CustomBoostExprEvaluator evaluator(expr_);
        evaluator.variables_["original_score"] = original_score;
        evaluator.variables_["boost_score"] = boost_score;
        return evaluator.parse_ternary();
    }

 private:
    std::string expr_;
    mutable size_t pos_;
    mutable std::unordered_map<std::string, float> variables_;

    void
    skip_whitespace() const {
        while (pos_ < expr_.length() && std::isspace(expr_[pos_])) {
            pos_++;
        }
    }

    char
    peek() const {
        skip_whitespace();
        return pos_ < expr_.length() ? expr_[pos_] : '\0';
    }

    char
    get() const {
        skip_whitespace();
        return pos_ < expr_.length() ? expr_[pos_++] : '\0';
    }

    bool
    match(const std::string& str) const {
        skip_whitespace();
        if (expr_.substr(pos_, str.length()) == str) {
            pos_ += str.length();
            return true;
        }
        return false;
    }

    float
    parse_ternary() const {
        float condition = parse_logical_or();
        skip_whitespace();
        if (peek() == '?') {
            get();  // consume '?'
            float true_val = parse_ternary();
            skip_whitespace();
            if (get() != ':') {
                throw std::runtime_error("Expected ':' in ternary operator");
            }
            float false_val = parse_ternary();
            return (condition != 0.0f) ? true_val : false_val;
        }
        return condition;
    }

    float
    parse_logical_or() const {
        float left = parse_logical_and();
        while (match("||")) {
            float right = parse_logical_and();
            left = (left != 0.0f || right != 0.0f) ? 1.0f : 0.0f;
        }
        return left;
    }

    float
    parse_logical_and() const {
        float left = parse_comparison();
        while (match("&&")) {
            float right = parse_comparison();
            left = (left != 0.0f && right != 0.0f) ? 1.0f : 0.0f;
        }
        return left;
    }

    float
    parse_comparison() const {
        float left = parse_additive();
        skip_whitespace();

        if (match(">=")) {
            return (left >= parse_additive()) ? 1.0f : 0.0f;
        } else if (match("<=")) {
            return (left <= parse_additive()) ? 1.0f : 0.0f;
        } else if (match("==")) {
            return (std::abs(left - parse_additive()) < 1e-6f) ? 1.0f : 0.0f;
        } else if (match("!=")) {
            return (std::abs(left - parse_additive()) >= 1e-6f) ? 1.0f : 0.0f;
        } else if (match(">")) {
            return (left > parse_additive()) ? 1.0f : 0.0f;
        } else if (match("<")) {
            return (left < parse_additive()) ? 1.0f : 0.0f;
        }

        return left;
    }

    float
    parse_additive() const {
        float left = parse_multiplicative();
        while (true) {
            skip_whitespace();
            char op = peek();
            if (op == '+' || op == '-') {
                get();
                float right = parse_multiplicative();
                left = (op == '+') ? left + right : left - right;
            } else {
                break;
            }
        }
        return left;
    }

    float
    parse_multiplicative() const {
        float left = parse_unary();
        while (true) {
            skip_whitespace();
            char op = peek();
            if (op == '*' || op == '/') {
                get();
                float right = parse_unary();
                if (op == '*') {
                    left *= right;
                } else {
                    if (std::abs(right) < 1e-10f) {
                        throw std::runtime_error("Division by zero");
                    }
                    left /= right;
                }
            } else if (op == '%') {
                get();
                float right = parse_unary();
                left = std::fmod(left, right);
            } else {
                break;
            }
        }
        return left;
    }

    float
    parse_unary() const {
        skip_whitespace();
        char ch = peek();
        if (ch == '-') {
            get();
            return -parse_unary();
        } else if (ch == '+') {
            get();
            return parse_unary();
        } else if (ch == '!') {
            get();
            return (parse_unary() == 0.0f) ? 1.0f : 0.0f;
        }
        return parse_power();
    }

    float
    parse_power() const {
        float left = parse_primary();
        if (match("**")) {
            float right = parse_power();  // Right associative
            return std::pow(left, right);
        }
        return left;
    }

    float
    parse_primary() const {
        skip_whitespace();

        // Check for parentheses
        if (peek() == '(') {
            get();
            float result = parse_ternary();
            if (get() != ')') {
                throw std::runtime_error("Mismatched parentheses");
            }
            return result;
        }

        // Check for function calls or variables
        size_t start = pos_;
        while (pos_ < expr_.length() &&
               (std::isalnum(expr_[pos_]) || expr_[pos_] == '_')) {
            pos_++;
        }

        if (pos_ > start) {
            std::string name = expr_.substr(start, pos_ - start);

            skip_whitespace();
            if (peek() == '(') {
                // Function call
                return parse_function(name);
            } else {
                // Variable
                auto it = variables_.find(name);
                if (it != variables_.end()) {
                    return it->second;
                }
                throw std::runtime_error(
                    fmt::format("Unknown variable: {}", name));
            }
        }

        // Check for number
        if (std::isdigit(peek()) || peek() == '.') {
            return parse_number();
        }

        throw std::runtime_error(fmt::format(
            "Unexpected character at position {}: '{}'", pos_, peek()));
    }

    float
    parse_number() const {
        size_t start = pos_;
        bool has_dot = false;

        while (pos_ < expr_.length() &&
               (std::isdigit(expr_[pos_]) || expr_[pos_] == '.')) {
            if (expr_[pos_] == '.') {
                if (has_dot) {
                    break;
                }
                has_dot = true;
            }
            pos_++;
        }

        try {
            return std::stof(expr_.substr(start, pos_ - start));
        } catch (const std::exception& e) {
            throw std::runtime_error(
                fmt::format("Invalid number: {}", expr_.substr(start, pos_ - start)));
        }
    }

    float
    parse_function(const std::string& name) const {
        get();  // consume '('

        if (name == "min") {
            float a = parse_ternary();
            if (get() != ',') {
                throw std::runtime_error("min() requires two arguments");
            }
            float b = parse_ternary();
            if (get() != ')') {
                throw std::runtime_error("Expected closing ')'");
            }
            return std::min(a, b);
        } else if (name == "max") {
            float a = parse_ternary();
            if (get() != ',') {
                throw std::runtime_error("max() requires two arguments");
            }
            float b = parse_ternary();
            if (get() != ')') {
                throw std::runtime_error("Expected closing ')'");
            }
            return std::max(a, b);
        } else if (name == "abs") {
            float val = parse_ternary();
            if (get() != ')') {
                throw std::runtime_error("Expected closing ')'");
            }
            return std::abs(val);
        } else if (name == "sqrt") {
            float val = parse_ternary();
            if (get() != ')') {
                throw std::runtime_error("Expected closing ')'");
            }
            return std::sqrt(val);
        } else if (name == "log") {
            float val = parse_ternary();
            if (get() != ')') {
                throw std::runtime_error("Expected closing ')'");
            }
            return std::log(val);
        } else if (name == "exp") {
            float val = parse_ternary();
            if (get() != ')') {
                throw std::runtime_error("Expected closing ')'");
            }
            return std::exp(val);
        } else if (name == "pow") {
            float base = parse_ternary();
            if (get() != ',') {
                throw std::runtime_error("pow() requires two arguments");
            }
            float exponent = parse_ternary();
            if (get() != ')') {
                throw std::runtime_error("Expected closing ')'");
            }
            return std::pow(base, exponent);
        } else if (name == "ceil") {
            float val = parse_ternary();
            if (get() != ')') {
                throw std::runtime_error("Expected closing ')'");
            }
            return std::ceil(val);
        } else if (name == "floor") {
            float val = parse_ternary();
            if (get() != ')') {
                throw std::runtime_error("Expected closing ')'");
            }
            return std::floor(val);
        } else if (name == "round") {
            float val = parse_ternary();
            if (get() != ')') {
                throw std::runtime_error("Expected closing ')'");
            }
            return std::round(val);
        } else if (name == "clamp") {
            float val = parse_ternary();
            if (get() != ',') {
                throw std::runtime_error("clamp() requires three arguments");
            }
            float min_val = parse_ternary();
            if (get() != ',') {
                throw std::runtime_error("clamp() requires three arguments");
            }
            float max_val = parse_ternary();
            if (get() != ')') {
                throw std::runtime_error("Expected closing ')'");
            }
            return std::clamp(val, min_val, max_val);
        }

        throw std::runtime_error(fmt::format("Unknown function: {}", name));
    }
};

}  // namespace milvus::rescores


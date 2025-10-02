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

#include <gtest/gtest.h>
#include <cmath>
#include "rescores/CustomBoostExprEvaluator.h"

using namespace milvus::rescores;

class CustomBoostExprTest : public ::testing::Test {
 protected:
    void SetUp() override {
    }

    void TearDown() override {
    }

    void
    AssertNearlyEqual(float actual, float expected, float epsilon = 1e-5f) {
        ASSERT_NEAR(actual, expected, epsilon);
    }
};

TEST_F(CustomBoostExprTest, BasicArithmetic) {
    // Test addition
    {
        CustomBoostExprEvaluator eval("original_score + boost_score");
        ASSERT_FLOAT_EQ(eval.evaluate(0.7f, 0.3f), 1.0f);
    }

    // Test subtraction
    {
        CustomBoostExprEvaluator eval("original_score - boost_score");
        ASSERT_FLOAT_EQ(eval.evaluate(0.7f, 0.3f), 0.4f);
    }

    // Test multiplication
    {
        CustomBoostExprEvaluator eval("original_score * boost_score");
        ASSERT_FLOAT_EQ(eval.evaluate(0.5f, 2.0f), 1.0f);
    }

    // Test division
    {
        CustomBoostExprEvaluator eval("original_score / boost_score");
        ASSERT_FLOAT_EQ(eval.evaluate(1.0f, 2.0f), 0.5f);
    }
}

TEST_F(CustomBoostExprTest, WeightedAverage) {
    CustomBoostExprEvaluator eval("original_score * 0.7 + boost_score * 0.3");
    ASSERT_FLOAT_EQ(eval.evaluate(0.8f, 0.6f), 0.8f * 0.7f + 0.6f * 0.3f);
}

TEST_F(CustomBoostExprTest, Conditional) {
    CustomBoostExprEvaluator eval(
        "original_score > 0.5 ? original_score * boost_score : original_score");

    // When condition is true
    ASSERT_FLOAT_EQ(eval.evaluate(0.7f, 2.0f), 1.4f);

    // When condition is false
    ASSERT_FLOAT_EQ(eval.evaluate(0.3f, 2.0f), 0.3f);
}

TEST_F(CustomBoostExprTest, MinFunction) {
    CustomBoostExprEvaluator eval("min(original_score + boost_score, 1.0)");

    // Result less than 1.0
    ASSERT_FLOAT_EQ(eval.evaluate(0.3f, 0.5f), 0.8f);

    // Result capped at 1.0
    ASSERT_FLOAT_EQ(eval.evaluate(0.7f, 0.5f), 1.0f);
}

TEST_F(CustomBoostExprTest, MaxFunction) {
    CustomBoostExprEvaluator eval("max(original_score, boost_score)");
    ASSERT_FLOAT_EQ(eval.evaluate(0.7f, 0.3f), 0.7f);
    ASSERT_FLOAT_EQ(eval.evaluate(0.3f, 0.7f), 0.7f);
}

TEST_F(CustomBoostExprTest, AbsFunction) {
    CustomBoostExprEvaluator eval("abs(original_score - boost_score)");
    ASSERT_FLOAT_EQ(eval.evaluate(0.7f, 0.3f), 0.4f);
    ASSERT_FLOAT_EQ(eval.evaluate(0.3f, 0.7f), 0.4f);
}

TEST_F(CustomBoostExprTest, SqrtFunction) {
    CustomBoostExprEvaluator eval("sqrt(original_score)");
    AssertNearlyEqual(eval.evaluate(0.25f, 0.0f), 0.5f);
}

TEST_F(CustomBoostExprTest, LogFunction) {
    CustomBoostExprEvaluator eval("original_score + log(1 + boost_score)");
    float expected = 0.5f + std::log(1.0f + 0.5f);
    AssertNearlyEqual(eval.evaluate(0.5f, 0.5f), expected);
}

TEST_F(CustomBoostExprTest, ExpFunction) {
    CustomBoostExprEvaluator eval("original_score * exp(boost_score)");
    float expected = 0.5f * std::exp(0.1f);
    AssertNearlyEqual(eval.evaluate(0.5f, 0.1f), expected);
}

TEST_F(CustomBoostExprTest, PowFunction) {
    CustomBoostExprEvaluator eval("pow(original_score, boost_score)");
    AssertNearlyEqual(eval.evaluate(2.0f, 3.0f), 8.0f);
}

TEST_F(CustomBoostExprTest, PowerOperator) {
    CustomBoostExprEvaluator eval("original_score ** boost_score");
    AssertNearlyEqual(eval.evaluate(2.0f, 3.0f), 8.0f);
}

TEST_F(CustomBoostExprTest, CeilFunction) {
    CustomBoostExprEvaluator eval("ceil(original_score)");
    ASSERT_FLOAT_EQ(eval.evaluate(2.1f, 0.0f), 3.0f);
    ASSERT_FLOAT_EQ(eval.evaluate(2.0f, 0.0f), 2.0f);
    ASSERT_FLOAT_EQ(eval.evaluate(-2.1f, 0.0f), -2.0f);
}

TEST_F(CustomBoostExprTest, FloorFunction) {
    CustomBoostExprEvaluator eval("floor(original_score)");
    ASSERT_FLOAT_EQ(eval.evaluate(2.9f, 0.0f), 2.0f);
    ASSERT_FLOAT_EQ(eval.evaluate(2.0f, 0.0f), 2.0f);
    ASSERT_FLOAT_EQ(eval.evaluate(-2.1f, 0.0f), -3.0f);
}

TEST_F(CustomBoostExprTest, RoundFunction) {
    CustomBoostExprEvaluator eval("round(original_score)");
    ASSERT_FLOAT_EQ(eval.evaluate(2.4f, 0.0f), 2.0f);
    ASSERT_FLOAT_EQ(eval.evaluate(2.5f, 0.0f), 3.0f);
    ASSERT_FLOAT_EQ(eval.evaluate(2.6f, 0.0f), 3.0f);
    ASSERT_FLOAT_EQ(eval.evaluate(-2.5f, 0.0f), -3.0f);
}

TEST_F(CustomBoostExprTest, ClampFunction) {
    CustomBoostExprEvaluator eval("clamp(original_score, 0.2, 0.8)");
    
    // Value within range
    ASSERT_FLOAT_EQ(eval.evaluate(0.5f, 0.0f), 0.5f);
    
    // Value below minimum
    ASSERT_FLOAT_EQ(eval.evaluate(0.1f, 0.0f), 0.2f);
    
    // Value above maximum
    ASSERT_FLOAT_EQ(eval.evaluate(0.9f, 0.0f), 0.8f);
    
    // Edge cases
    ASSERT_FLOAT_EQ(eval.evaluate(0.2f, 0.0f), 0.2f);
    ASSERT_FLOAT_EQ(eval.evaluate(0.8f, 0.0f), 0.8f);
}

TEST_F(CustomBoostExprTest, ComplexExpression) {
    CustomBoostExprEvaluator eval(
        "max(min(original_score * 2.0, 1.0), 0.0) + boost_score * 0.1");

    // original_score * 2.0 = 0.8, clamped to [0.0, 1.0] = 0.8, + 0.5 * 0.1 = 0.85
    ASSERT_FLOAT_EQ(eval.evaluate(0.4f, 0.5f), 0.85f);

    // original_score * 2.0 = 1.4, clamped to [0.0, 1.0] = 1.0, + 0.5 * 0.1 = 1.05
    ASSERT_FLOAT_EQ(eval.evaluate(0.7f, 0.5f), 1.05f);
}

TEST_F(CustomBoostExprTest, Comparison) {
    // Greater than
    {
        CustomBoostExprEvaluator eval("original_score > boost_score");
        ASSERT_FLOAT_EQ(eval.evaluate(0.7f, 0.3f), 1.0f);
        ASSERT_FLOAT_EQ(eval.evaluate(0.3f, 0.7f), 0.0f);
    }

    // Less than
    {
        CustomBoostExprEvaluator eval("original_score < boost_score");
        ASSERT_FLOAT_EQ(eval.evaluate(0.3f, 0.7f), 1.0f);
        ASSERT_FLOAT_EQ(eval.evaluate(0.7f, 0.3f), 0.0f);
    }

    // Greater than or equal
    {
        CustomBoostExprEvaluator eval("original_score >= boost_score");
        ASSERT_FLOAT_EQ(eval.evaluate(0.5f, 0.5f), 1.0f);
        ASSERT_FLOAT_EQ(eval.evaluate(0.7f, 0.3f), 1.0f);
        ASSERT_FLOAT_EQ(eval.evaluate(0.3f, 0.7f), 0.0f);
    }

    // Less than or equal
    {
        CustomBoostExprEvaluator eval("original_score <= boost_score");
        ASSERT_FLOAT_EQ(eval.evaluate(0.5f, 0.5f), 1.0f);
        ASSERT_FLOAT_EQ(eval.evaluate(0.3f, 0.7f), 1.0f);
        ASSERT_FLOAT_EQ(eval.evaluate(0.7f, 0.3f), 0.0f);
    }
}

TEST_F(CustomBoostExprTest, LogicalOperators) {
    // Logical AND
    {
        CustomBoostExprEvaluator eval(
            "(original_score > 0.5) && (boost_score > 0.5)");
        ASSERT_FLOAT_EQ(eval.evaluate(0.7f, 0.7f), 1.0f);
        ASSERT_FLOAT_EQ(eval.evaluate(0.3f, 0.7f), 0.0f);
        ASSERT_FLOAT_EQ(eval.evaluate(0.7f, 0.3f), 0.0f);
    }

    // Logical OR
    {
        CustomBoostExprEvaluator eval(
            "(original_score > 0.5) || (boost_score > 0.5)");
        ASSERT_FLOAT_EQ(eval.evaluate(0.7f, 0.3f), 1.0f);
        ASSERT_FLOAT_EQ(eval.evaluate(0.3f, 0.7f), 1.0f);
        ASSERT_FLOAT_EQ(eval.evaluate(0.3f, 0.3f), 0.0f);
    }
}

TEST_F(CustomBoostExprTest, Parentheses) {
    CustomBoostExprEvaluator eval("(original_score + boost_score) * 2.0");
    ASSERT_FLOAT_EQ(eval.evaluate(0.3f, 0.2f), 1.0f);
}

TEST_F(CustomBoostExprTest, NestedTernary) {
    CustomBoostExprEvaluator eval(
        "original_score > 0.7 ? 1.0 : original_score > 0.3 ? 0.5 : 0.0");

    ASSERT_FLOAT_EQ(eval.evaluate(0.8f, 0.0f), 1.0f);
    ASSERT_FLOAT_EQ(eval.evaluate(0.5f, 0.0f), 0.5f);
    ASSERT_FLOAT_EQ(eval.evaluate(0.2f, 0.0f), 0.0f);
}

TEST_F(CustomBoostExprTest, InvalidExpressionThrows) {
    // Invalid variable
    {
        CustomBoostExprEvaluator eval("invalid_var");
        ASSERT_THROW(eval.evaluate(0.5f, 0.5f), std::runtime_error);
    }

    // Division by zero
    {
        CustomBoostExprEvaluator eval("original_score / 0.0");
        ASSERT_THROW(eval.evaluate(0.5f, 0.5f), std::runtime_error);
    }

    // Mismatched parentheses
    {
        CustomBoostExprEvaluator eval("(original_score + boost_score");
        ASSERT_THROW(eval.evaluate(0.5f, 0.5f), std::runtime_error);
    }

    // Invalid function
    {
        CustomBoostExprEvaluator eval("unknown_func(original_score)");
        ASSERT_THROW(eval.evaluate(0.5f, 0.5f), std::runtime_error);
    }
}

TEST_F(CustomBoostExprTest, Phase1CombinedExamples) {
    // Score normalization with ceiling
    {
        CustomBoostExprEvaluator eval("ceil(original_score * 10) / 10");
        ASSERT_FLOAT_EQ(eval.evaluate(0.73f, 0.0f), 0.8f);
        ASSERT_FLOAT_EQ(eval.evaluate(0.71f, 0.0f), 0.8f);
    }

    // Score clamping with boost
    {
        CustomBoostExprEvaluator eval("clamp(original_score + boost_score, 0.0, 1.0)");
        ASSERT_FLOAT_EQ(eval.evaluate(0.7f, 0.2f), 0.9f);  // Within range
        ASSERT_FLOAT_EQ(eval.evaluate(0.8f, 0.5f), 1.0f);  // Clamped to max
        ASSERT_FLOAT_EQ(eval.evaluate(-0.1f, 0.05f), 0.0f); // Clamped to min
    }

    // Rounded weighted average
    {
        CustomBoostExprEvaluator eval("round((original_score * 0.7 + boost_score * 0.3) * 100) / 100");
        float expected = std::round((0.73f * 0.7f + 0.84f * 0.3f) * 100) / 100;
        ASSERT_FLOAT_EQ(eval.evaluate(0.73f, 0.84f), expected);
    }

    // Floor-based scoring tiers
    {
        CustomBoostExprEvaluator eval("floor(original_score * 5) / 5 + boost_score * 0.1");
        float tier = std::floor(0.67f * 5) / 5;  // 0.6 (tier)
        float expected = tier + 0.8f * 0.1f;     // 0.68
        ASSERT_FLOAT_EQ(eval.evaluate(0.67f, 0.8f), expected);
    }
}

TEST_F(CustomBoostExprTest, RealWorldExamples) {
    // Example 1: Weighted combination with cap
    {
        CustomBoostExprEvaluator eval("min(original_score * 0.8 + boost_score * 0.2, 1.0)");
        float result = eval.evaluate(0.9f, 0.8f);
        ASSERT_FLOAT_EQ(result, std::min(0.9f * 0.8f + 0.8f * 0.2f, 1.0f));
    }

    // Example 2: Logarithmic boost
    {
        CustomBoostExprEvaluator eval("original_score + log(1.0 + boost_score)");
        float result = eval.evaluate(0.5f, 2.0f);
        AssertNearlyEqual(result, 0.5f + std::log(3.0f));
    }

    // Example 3: Conditional boost application
    {
        CustomBoostExprEvaluator eval(
            "original_score > 0.6 ? original_score * boost_score : original_score + boost_score * 0.1");

        // High score - multiply
        float result1 = eval.evaluate(0.8f, 1.5f);
        ASSERT_FLOAT_EQ(result1, 1.2f);

        // Low score - add small boost
        float result2 = eval.evaluate(0.4f, 1.5f);
        ASSERT_FLOAT_EQ(result2, 0.55f);
    }

    // Example 4: NEW - Score quantization with clamp
    {
        CustomBoostExprEvaluator eval("clamp(ceil(original_score * boost_score * 10) / 10, 0.1, 1.0)");
        float raw_score = 0.73f * 1.2f * 10;  // 8.76
        float quantized = std::ceil(raw_score) / 10;  // 0.9
        float clamped = std::clamp(quantized, 0.1f, 1.0f);  // 0.9
        ASSERT_FLOAT_EQ(eval.evaluate(0.73f, 1.2f), clamped);
    }
}


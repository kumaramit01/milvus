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

package rerank

import (
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/milvus-io/milvus-proto/go-api/v2/commonpb"
	"github.com/milvus-io/milvus-proto/go-api/v2/schemapb"
)

func TestToFloat64TypeConversion(t *testing.T) {
	testCases := []struct {
		name     string
		input    interface{}
		expected float64
	}{
		{"float64", float64(3.14159), 3.14159},
		{"float32", float32(2.71828), 2.71828},
		{"int", int(42), 42.0},
		{"int32", int32(100), 100.0},
		{"int64", int64(1000), 1000.0},
		{"negative int", int(-50), -50.0},
		{"zero", int(0), 0.0},
		{"unsupported type", "string", 0.0}, // fallback
		{"nil", nil, 0.0},                   // fallback
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			result := toFloat64(tc.input)
			assert.Equal(t, tc.expected, result, "toFloat64(%v) should return %f", tc.input, tc.expected)
		})
	}
}
func TestNewExprRerank(t *testing.T) {
	collSchema := &schemapb.CollectionSchema{
		Name: "test_collection",
		Fields: []*schemapb.FieldSchema{
			{FieldID: 1, Name: "id", DataType: schemapb.DataType_Int64, IsPrimaryKey: true},
			{FieldID: 2, Name: "quality", DataType: schemapb.DataType_Float},
			{FieldID: 3, Name: "popularity", DataType: schemapb.DataType_Int64},
		},
	}

	t.Run("valid expression", func(t *testing.T) {
		funcSchema := &schemapb.FunctionSchema{
			Name:            "test_expr_rerank",
			Type:            schemapb.FunctionType_Rerank,
			InputFieldNames: []string{"quality", "popularity"},
			Params: []*commonpb.KeyValuePair{
				{Key: "reranker", Value: "expr"},
				{Key: "expr_code", Value: "score * (fields[\"quality\"] / 100.0)"},
			},
		}

		reranker, err := newExprRerank(collSchema, funcSchema)
		assert.NoError(t, err)
		assert.NotNil(t, reranker)
	})

	t.Run("missing expr_code", func(t *testing.T) {
		funcSchema := &schemapb.FunctionSchema{
			Name:            "test_missing_code",
			Type:            schemapb.FunctionType_Rerank,
			InputFieldNames: []string{"quality"},
			Params: []*commonpb.KeyValuePair{
				{Key: "reranker", Value: "expr"},
				// Missing expr_code
			},
		}

		reranker, err := newExprRerank(collSchema, funcSchema)
		assert.Error(t, err)
		assert.Nil(t, reranker)
		assert.Contains(t, err.Error(), "expr_code")
	})

	t.Run("invalid expression syntax", func(t *testing.T) {
		funcSchema := &schemapb.FunctionSchema{
			Name:            "test_invalid_syntax",
			Type:            schemapb.FunctionType_Rerank,
			InputFieldNames: []string{"quality"},
			Params: []*commonpb.KeyValuePair{
				{Key: "reranker", Value: "expr"},
				{Key: "expr_code", Value: "score * * invalid"},
			},
		}

		reranker, err := newExprRerank(collSchema, funcSchema)
		assert.Error(t, err)
		assert.Nil(t, reranker)
	})

	t.Run("mathematical functions in expression", func(t *testing.T) {
		testCases := []struct {
			name     string
			exprCode string
		}{
			{"log function", "score * log(fields[\"popularity\"] + 1)"},
			{"exp function", "score * exp(-0.01 * fields[\"quality\"])"},
			{"sqrt function", "score * sqrt(fields[\"quality\"])"},
			{"pow function", "score * pow(fields[\"quality\"], 2.0)"},
			{"min function", "min(score * fields[\"quality\"], score * 5.0)"},
			{"max function", "max(score, fields[\"quality\"] / 100.0)"},
			{"abs function", "score * abs(fields[\"quality\"] - 50.0)"},
			{"complex expression", "score * exp(-0.01 * fields[\"quality\"]) * (1.0 + log(fields[\"popularity\"] + 1) / 10.0)"},
		}

		for _, tc := range testCases {
			t.Run(tc.name, func(t *testing.T) {
				funcSchema := &schemapb.FunctionSchema{
					Name:            "test_math_" + tc.name,
					Type:            schemapb.FunctionType_Rerank,
					InputFieldNames: []string{"quality", "popularity"},
					Params: []*commonpb.KeyValuePair{
						{Key: "reranker", Value: "expr"},
						{Key: "expr_code", Value: tc.exprCode},
					},
				}

				reranker, err := newExprRerank(collSchema, funcSchema)
				assert.NoError(t, err, "Expression should compile successfully: %s", tc.exprCode)
				assert.NotNil(t, reranker)
			})
		}
	})

	t.Run("normalize parameter", func(t *testing.T) {
		testCases := []struct {
			name         string
			normalizeVal string
			shouldError  bool
			expectedNorm bool
		}{
			{"true", "true", false, true},
			{"false", "false", false, false},
			{"1", "1", false, true},
			{"0", "0", false, false},
			{"yes", "yes", false, true},
			{"no", "no", false, false},
			{"invalid", "invalid", true, false},
		}

		for _, tc := range testCases {
			t.Run(tc.name, func(t *testing.T) {
				funcSchema := &schemapb.FunctionSchema{
					Name:            "test_normalize",
					Type:            schemapb.FunctionType_Rerank,
					InputFieldNames: []string{"quality"},
					Params: []*commonpb.KeyValuePair{
						{Key: "reranker", Value: "expr"},
						{Key: "expr_code", Value: "score"},
						{Key: "normalize", Value: tc.normalizeVal},
					},
				}

				reranker, err := newExprRerank(collSchema, funcSchema)
				if tc.shouldError {
					assert.Error(t, err)
					assert.Nil(t, reranker)
				} else {
					assert.NoError(t, err)
					assert.NotNil(t, reranker)
				}
			})
		}
	})
}

func TestExprRerankExpressionExamples(t *testing.T) {
	// Test real-world expression examples that should compile successfully
	collSchema := &schemapb.CollectionSchema{
		Name: "test_collection",
		Fields: []*schemapb.FieldSchema{
			{FieldID: 1, Name: "id", DataType: schemapb.DataType_Int64, IsPrimaryKey: true},
			{FieldID: 2, Name: "created_at", DataType: schemapb.DataType_Int64},
			{FieldID: 3, Name: "quality_score", DataType: schemapb.DataType_Float},
			{FieldID: 4, Name: "popularity", DataType: schemapb.DataType_Int64},
			{FieldID: 5, Name: "category", DataType: schemapb.DataType_VarChar},
		},
	}

	testCases := []struct {
		name        string
		exprCode    string
		description string
	}{
		{
			name:        "basic_score_passthrough",
			exprCode:    "score",
			description: "Basic expression that returns original score unchanged",
		},
		{
			name:        "quality_boost",
			exprCode:    "score * (fields[\"quality_score\"] / 100.0)",
			description: "Quality-based boosting",
		},
		{
			name:        "recency_boost",
			exprCode:    "let age_days = (1704067200000 - fields[\"created_at\"]) / 86400000; score * exp(-0.01 * age_days)",
			description: "Recency-based boosting with exponential decay",
		},
		{
			name:        "popularity_boost",
			exprCode:    "score * (1.0 + log(fields[\"popularity\"] + 1) / 10.0)",
			description: "Popularity-based boosting with logarithmic scaling",
		},
		{
			name:        "category_conditional",
			exprCode:    "let boost = fields[\"category\"] == \"featured\" ? 2.0 : fields[\"category\"] == \"premium\" ? 1.5 : 1.0; score * boost",
			description: "Conditional category boosting",
		},
		{
			name:        "multi_factor",
			exprCode:    "let quality = fields[\"quality_score\"] / 100.0; let pop_boost = 1.0 + log(fields[\"popularity\"] + 1) / 20.0; score * quality * pop_boost",
			description: "Multi-factor scoring combining quality and popularity",
		},
		{
			name:        "quality_filter",
			exprCode:    "fields[\"quality_score\"] > 80.0 ? score : score * 0.1",
			description: "Quality filtering with penalty for low quality",
		},
		{
			name:        "capped_boost",
			exprCode:    "min(score * (fields[\"popularity\"] / 100.0), score * 5.0)",
			description: "Capped boosting to prevent extreme score inflation",
		},
		{
			name:        "balanced_scoring",
			exprCode:    "let age_days = (1704067200000 - fields[\"created_at\"]) / 86400000; let recency = exp(-0.005 * age_days); let quality = fields[\"quality_score\"] / 100.0; score * (recency * 0.6 + quality * 0.4)",
			description: "Balanced scoring between recency and quality",
		},
		{
			name:        "complex_conditional",
			exprCode:    "let age_days = (1704067200000 - fields[\"created_at\"]) / 86400000; let is_recent = age_days < 7; let is_high_quality = fields[\"quality_score\"] > 85; let is_popular = fields[\"popularity\"] > 500; score * (is_recent ? 1.5 : 1.0) * (is_high_quality ? 1.3 : 1.0) * (is_popular ? 1.2 : 1.0)",
			description: "Complex conditional logic with multiple boolean factors",
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			funcSchema := &schemapb.FunctionSchema{
				Name:            "test_" + tc.name,
				Type:            schemapb.FunctionType_Rerank,
				InputFieldNames: []string{"created_at", "quality_score", "popularity", "category"},
				Params: []*commonpb.KeyValuePair{
					{Key: "reranker", Value: "expr"},
					{Key: "expr_code", Value: tc.exprCode},
				},
			}

			reranker, err := newExprRerank(collSchema, funcSchema)
			require.NoError(t, err, "Expression should compile successfully: %s", tc.description)
			require.NotNil(t, reranker, "Reranker should be created for: %s", tc.description)

			t.Logf("✅ %s: %s", tc.description, tc.exprCode)
		})
	}
}

func TestParseBool(t *testing.T) {
	testCases := []struct {
		input       string
		expected    bool
		shouldError bool
	}{
		{"true", true, false},
		{"TRUE", true, false},
		{"True", true, false},
		{"1", true, false},
		{"yes", true, false},
		{"YES", true, false},
		{"false", false, false},
		{"FALSE", false, false},
		{"False", false, false},
		{"0", false, false},
		{"no", false, false},
		{"NO", false, false},
		{"invalid", false, true},
		{"", false, true},
		{"2", false, true},
		{"maybe", false, true},
	}

	for _, tc := range testCases {
		t.Run("input_"+tc.input, func(t *testing.T) {
			result, err := parseBool(tc.input)
			if tc.shouldError {
				assert.Error(t, err)
			} else {
				assert.NoError(t, err)
				assert.Equal(t, tc.expected, result)
			}
		})
	}
}
func TestGetExecutionLevel(t *testing.T) {
	tests := []struct {
		name           string
		rerankerName   string
		executionLevel string // empty if not set
		expectedLevel  string
		expectedIsQN   bool
	}{
		{
			name:           "ExprRerank with explicit QueryNode",
			rerankerName:   ExprRerankName,
			executionLevel: "querynode",
			expectedLevel:  "querynode",
			expectedIsQN:   true,
		},
		{
			name:           "ExprRerank with explicit Proxy",
			rerankerName:   ExprRerankName,
			executionLevel: "proxy",
			expectedLevel:  "proxy",
			expectedIsQN:   false,
		},
		{
			name:           "ExprRerank without execution_level (backward compat)",
			rerankerName:   ExprRerankName,
			executionLevel: "",
			expectedLevel:  "querynode", // Falls back to type-based logic
			expectedIsQN:   true,
		},
		{
			name:           "Wasm with explicit QueryNode",
			rerankerName:   WasmName,
			executionLevel: "querynode",
			expectedLevel:  "querynode",
			expectedIsQN:   true,
		},
		{
			name:           "Wasm with explicit Proxy",
			rerankerName:   WasmName,
			executionLevel: "proxy",
			expectedLevel:  "proxy",
			expectedIsQN:   false,
		},
		{
			name:           "Wasm without execution_level (backward compat)",
			rerankerName:   WasmName,
			executionLevel: "",
			expectedLevel:  "querynode", // Falls back to type-based logic
			expectedIsQN:   true,
		},
		{
			name:           "Decay with explicit QueryNode",
			rerankerName:   DecayFunctionName,
			executionLevel: "querynode",
			expectedLevel:  "querynode",
			expectedIsQN:   true,
		},
		{
			name:           "Decay with explicit Proxy",
			rerankerName:   DecayFunctionName,
			executionLevel: "proxy",
			expectedLevel:  "proxy",
			expectedIsQN:   false,
		},
		{
			name:           "Decay without execution_level (backward compat)",
			rerankerName:   DecayFunctionName,
			executionLevel: "",
			expectedLevel:  "proxy", // Falls back to type-based logic
			expectedIsQN:   false,
		},
		{
			name:           "RRF with explicit QueryNode",
			rerankerName:   RRFName,
			executionLevel: "querynode",
			expectedLevel:  "querynode",
			expectedIsQN:   true,
		},
		{
			name:           "RRF without execution_level (backward compat)",
			rerankerName:   RRFName,
			executionLevel: "",
			expectedLevel:  "proxy", // Falls back to type-based logic
			expectedIsQN:   false,
		},
		{
			name:           "Weighted with explicit QueryNode",
			rerankerName:   WeightedName,
			executionLevel: "querynode",
			expectedLevel:  "querynode",
			expectedIsQN:   true,
		},
		{
			name:           "Model with explicit QueryNode",
			rerankerName:   ModelFunctionName,
			executionLevel: "querynode",
			expectedLevel:  "querynode",
			expectedIsQN:   true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			funcSchema := &schemapb.FunctionSchema{
				Name: "test_reranker",
				Type: schemapb.FunctionType_Rerank,
				Params: []*commonpb.KeyValuePair{
					{Key: "reranker", Value: tt.rerankerName},
				},
			}

			// Add execution_level param if specified
			if tt.executionLevel != "" {
				funcSchema.Params = append(funcSchema.Params,
					&commonpb.KeyValuePair{Key: "execution_level", Value: tt.executionLevel})
			}

			// Test GetExecutionLevel
			actualLevel := GetExecutionLevel(funcSchema)
			assert.Equal(t, tt.expectedLevel, actualLevel,
				"GetExecutionLevel should return %s", tt.expectedLevel)

			// Test GetExecutionLevel
			actualIsQN := GetExecutionLevel(funcSchema) == "querynode"
			assert.Equal(t, tt.expectedIsQN, actualIsQN,
				"IsQueryNodeRanker should return %v", tt.expectedIsQN)
		})
	}
}

func TestGetExecutionLevelCaseInsensitive(t *testing.T) {
	tests := []struct {
		name           string
		executionLevel string
		expectedLevel  string
	}{
		{"QueryNode uppercase", "QUERYNODE", "querynode"},
		{"QueryNode mixed case", "QueryNode", "querynode"},
		{"Proxy uppercase", "PROXY", "proxy"},
		{"Proxy mixed case", "Proxy", "proxy"},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			funcSchema := &schemapb.FunctionSchema{
				Name: "test",
				Type: schemapb.FunctionType_Rerank,
				Params: []*commonpb.KeyValuePair{
					{Key: "reranker", Value: ExprRerankName},
					{Key: "execution_level", Value: tt.executionLevel},
				},
			}

			level := GetExecutionLevel(funcSchema)
			assert.Equal(t, tt.expectedLevel, level)
		})
	}
}

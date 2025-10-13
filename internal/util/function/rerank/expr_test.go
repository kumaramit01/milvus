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
	"context"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/milvus-io/milvus-proto/go-api/v2/commonpb"
	"github.com/milvus-io/milvus-proto/go-api/v2/schemapb"
)

func TestExprRerankWithRealData(t *testing.T) {
	// Create collection schema
	collSchema := &schemapb.CollectionSchema{
		Name: "test_collection",
		Fields: []*schemapb.FieldSchema{
			{FieldID: 1, Name: "id", DataType: schemapb.DataType_Int64, IsPrimaryKey: true},
			{FieldID: 2, Name: "quality_score", DataType: schemapb.DataType_Float},
			{FieldID: 3, Name: "popularity", DataType: schemapb.DataType_Int64},
		},
	}

	t.Run("quality_boost", func(t *testing.T) {
		funcSchema := &schemapb.FunctionSchema{
			Name:            "quality_boost",
			Type:            schemapb.FunctionType_Rerank,
			InputFieldNames: []string{"quality_score"},
			Params: []*commonpb.KeyValuePair{
				{Key: "reranker", Value: "expr"},
				{Key: "expr_code", Value: "score * (fields[\"quality_score\"] / 100.0)"},
			},
		}

		reranker, err := newExprRerank(collSchema, funcSchema)
		require.NoError(t, err)
		require.NotNil(t, reranker)

		// Create test data
		searchParams := NewSearchParams(1, 5, 0, -1, -1, 0, false, "", []string{})

		searchResultData := &schemapb.SearchResultData{
			NumQueries: 1,
			TopK:       5,
			Ids:        &schemapb.IDs{IdField: &schemapb.IDs_IntId{IntId: &schemapb.LongArray{Data: []int64{1, 2, 3, 4, 5}}}},
			Scores:     []float32{0.9, 0.8, 0.7, 0.6, 0.5}, // Original scores
			Topks:      []int64{5},
			FieldsData: []*schemapb.FieldData{
				{
					FieldId:   2,
					FieldName: "quality_score",
					Type:      schemapb.DataType_Float,
					Field: &schemapb.FieldData_Scalars{
						Scalars: &schemapb.ScalarField{
							Data: &schemapb.ScalarField_FloatData{
								FloatData: &schemapb.FloatArray{Data: []float32{90.0, 50.0, 80.0, 30.0, 100.0}},
							},
						},
					},
				},
			},
		}

		inputs, err := newRerankInputs([]*schemapb.SearchResultData{searchResultData}, reranker.GetInputFieldIDs(), false)
		require.NoError(t, err)

		// Process reranking
		outputs, err := reranker.Process(context.Background(), searchParams, inputs)
		require.NoError(t, err)
		require.NotNil(t, outputs)

		// Verify results
		// ID 1: 0.9 * (90/100) = 0.81
		// ID 2: 0.8 * (50/100) = 0.40
		// ID 3: 0.7 * (80/100) = 0.56
		// ID 4: 0.6 * (30/100) = 0.18
		// ID 5: 0.5 * (100/100) = 0.50
		assert.NotNil(t, outputs.searchResultData)
		assert.Equal(t, int64(5), outputs.searchResultData.Topks[0])

		// After reranking, order should be: 1 (0.81), 3 (0.56), 5 (0.50), 2 (0.40), 4 (0.18)
		t.Logf("Reranked scores: %v", outputs.searchResultData.Scores)
	})

	t.Run("popularity_log_boost", func(t *testing.T) {
		funcSchema := &schemapb.FunctionSchema{
			Name:            "popularity_log_boost",
			Type:            schemapb.FunctionType_Rerank,
			InputFieldNames: []string{"popularity"},
			Params: []*commonpb.KeyValuePair{
				{Key: "reranker", Value: "expr"},
				{Key: "expr_code", Value: "score * (1.0 + log(fields[\"popularity\"] + 1) / 10.0)"},
			},
		}

		reranker, err := newExprRerank(collSchema, funcSchema)
		require.NoError(t, err)

		searchParams := NewSearchParams(1, 3, 0, -1, -1, 0, false, "", []string{})

		searchResultData := &schemapb.SearchResultData{
			NumQueries: 1,
			TopK:       3,
			Ids:        &schemapb.IDs{IdField: &schemapb.IDs_IntId{IntId: &schemapb.LongArray{Data: []int64{1, 2, 3}}}},
			Scores:     []float32{0.5, 0.5, 0.5},
			Topks:      []int64{3},
			FieldsData: []*schemapb.FieldData{
				{
					FieldId:   3,
					FieldName: "popularity",
					Type:      schemapb.DataType_Int64,
					Field: &schemapb.FieldData_Scalars{
						Scalars: &schemapb.ScalarField{
							Data: &schemapb.ScalarField_LongData{
								LongData: &schemapb.LongArray{Data: []int64{1000, 100, 10}},
							},
						},
					},
				},
			},
		}

		inputs, err := newRerankInputs([]*schemapb.SearchResultData{searchResultData}, reranker.GetInputFieldIDs(), false)
		require.NoError(t, err)

		outputs, err := reranker.Process(context.Background(), searchParams, inputs)
		require.NoError(t, err)

		// ID 1 (popularity 1000) should have highest score after log boost
		// ID 3 (popularity 10) should have lowest
		scores := outputs.searchResultData.Scores
		t.Logf("Scores after popularity boost: %v", scores)

		// Verify scores are boosted proportionally
		assert.True(t, scores[0] > 0.5, "Most popular item should have boosted score")
	})

	t.Run("conditional_category_boost", func(t *testing.T) {
		// Test with string fields
		collSchema := &schemapb.CollectionSchema{
			Name: "test_collection",
			Fields: []*schemapb.FieldSchema{
				{FieldID: 1, Name: "id", DataType: schemapb.DataType_Int64, IsPrimaryKey: true},
				{FieldID: 2, Name: "category", DataType: schemapb.DataType_VarChar},
			},
		}

		funcSchema := &schemapb.FunctionSchema{
			Name:            "category_boost",
			Type:            schemapb.FunctionType_Rerank,
			InputFieldNames: []string{"category"},
			Params: []*commonpb.KeyValuePair{
				{Key: "reranker", Value: "expr"},
				{Key: "expr_code", Value: `let boost = fields["category"] == "featured" ? 2.0 : fields["category"] == "premium" ? 1.5 : 1.0; score * boost`},
			},
		}

		reranker, err := newExprRerank(collSchema, funcSchema)
		require.NoError(t, err)

		searchParams := NewSearchParams(1, 3, 0, -1, -1, 0, false, "", []string{})

		searchResultData := &schemapb.SearchResultData{
			NumQueries: 1,
			TopK:       3,
			Ids:        &schemapb.IDs{IdField: &schemapb.IDs_IntId{IntId: &schemapb.LongArray{Data: []int64{1, 2, 3}}}},
			Scores:     []float32{0.5, 0.5, 0.5},
			Topks:      []int64{3},
			FieldsData: []*schemapb.FieldData{
				{
					FieldId:   2,
					FieldName: "category",
					Type:      schemapb.DataType_VarChar,
					Field: &schemapb.FieldData_Scalars{
						Scalars: &schemapb.ScalarField{
							Data: &schemapb.ScalarField_StringData{
								StringData: &schemapb.StringArray{Data: []string{"featured", "premium", "standard"}},
							},
						},
					},
				},
			},
		}

		inputs, err := newRerankInputs([]*schemapb.SearchResultData{searchResultData}, reranker.GetInputFieldIDs(), false)
		require.NoError(t, err)

		outputs, err := reranker.Process(context.Background(), searchParams, inputs)
		require.NoError(t, err)

		scores := outputs.searchResultData.Scores
		t.Logf("Category boost scores: %v", scores)

		// Featured (2.0x): 0.5 * 2.0 = 1.0
		// Premium (1.5x): 0.5 * 1.5 = 0.75
		// Standard (1.0x): 0.5 * 1.0 = 0.5
		assert.InDelta(t, 1.0, scores[0], 0.01, "Featured should have 2x boost")
		assert.InDelta(t, 0.75, scores[1], 0.01, "Premium should have 1.5x boost")
		assert.InDelta(t, 0.5, scores[2], 0.01, "Standard should have 1x boost")
	})
}

func TestExprRerankPositionBased(t *testing.T) {
	collSchema := &schemapb.CollectionSchema{
		Name: "test_collection",
		Fields: []*schemapb.FieldSchema{
			{FieldID: 1, Name: "id", DataType: schemapb.DataType_Int64, IsPrimaryKey: true},
		},
	}

	funcSchema := &schemapb.FunctionSchema{
		Name:            "position_decay",
		Type:            schemapb.FunctionType_Rerank,
		InputFieldNames: []string{}, // No field data, only rank
		Params: []*commonpb.KeyValuePair{
			{Key: "reranker", Value: "expr"},
			{Key: "expr_code", Value: "score * (1.0 / (1.0 + 0.1 * rank))"},
		},
	}

	reranker, err := newExprRerank(collSchema, funcSchema)
	require.NoError(t, err)

	searchParams := NewSearchParams(1, 5, 0, -1, -1, 0, false, "", []string{})

	// All items have same initial score, but rank should differentiate them
	searchResultData := &schemapb.SearchResultData{
		NumQueries: 1,
		TopK:       5,
		Ids:        &schemapb.IDs{IdField: &schemapb.IDs_IntId{IntId: &schemapb.LongArray{Data: []int64{1, 2, 3, 4, 5}}}},
		Scores:     []float32{1.0, 1.0, 1.0, 1.0, 1.0},
		Topks:      []int64{5},
		FieldsData: []*schemapb.FieldData{},
	}

	inputs, err := newRerankInputs([]*schemapb.SearchResultData{searchResultData}, reranker.GetInputFieldIDs(), false)
	require.NoError(t, err)

	outputs, err := reranker.Process(context.Background(), searchParams, inputs)
	require.NoError(t, err)

	scores := outputs.searchResultData.Scores
	t.Logf("Position decay scores: %v", scores)

	// Rank 0: 1.0 * (1 / (1 + 0)) = 1.0
	// Rank 1: 1.0 * (1 / (1 + 0.1)) ≈ 0.909
	// Rank 2: 1.0 * (1 / (1 + 0.2)) ≈ 0.833
	// Each successive rank should have lower score
	for i := 1; i < len(scores); i++ {
		assert.True(t, scores[i-1] > scores[i], "Score should decrease with rank position")
	}
}

func TestExprRerankMultipleQueries(t *testing.T) {
	collSchema := &schemapb.CollectionSchema{
		Name: "test_collection",
		Fields: []*schemapb.FieldSchema{
			{FieldID: 1, Name: "id", DataType: schemapb.DataType_Int64, IsPrimaryKey: true},
			{FieldID: 2, Name: "boost_factor", DataType: schemapb.DataType_Float},
		},
	}

	funcSchema := &schemapb.FunctionSchema{
		Name:            "simple_boost",
		Type:            schemapb.FunctionType_Rerank,
		InputFieldNames: []string{"boost_factor"},
		Params: []*commonpb.KeyValuePair{
			{Key: "reranker", Value: "expr"},
			{Key: "expr_code", Value: "score * fields[\"boost_factor\"]"},
		},
	}

	reranker, err := newExprRerank(collSchema, funcSchema)
	require.NoError(t, err)

	searchParams := NewSearchParams(2, 3, 0, -1, -1, 0, false, "", []string{})

	searchResultData := &schemapb.SearchResultData{
		NumQueries: 2,
		TopK:       3,
		Ids:        &schemapb.IDs{IdField: &schemapb.IDs_IntId{IntId: &schemapb.LongArray{Data: []int64{1, 2, 3, 4, 5, 6}}}},
		Scores:     []float32{0.9, 0.8, 0.7, 0.6, 0.5, 0.4},
		Topks:      []int64{3, 3},
		FieldsData: []*schemapb.FieldData{
			{
				FieldId:   2,
				FieldName: "boost_factor",
				Type:      schemapb.DataType_Float,
				Field: &schemapb.FieldData_Scalars{
					Scalars: &schemapb.ScalarField{
						Data: &schemapb.ScalarField_FloatData{
							FloatData: &schemapb.FloatArray{Data: []float32{1.0, 2.0, 0.5, 1.5, 1.0, 2.0}},
						},
					},
				},
			},
		},
	}

	inputs, err := newRerankInputs([]*schemapb.SearchResultData{searchResultData}, reranker.GetInputFieldIDs(), false)
	require.NoError(t, err)

	outputs, err := reranker.Process(context.Background(), searchParams, inputs)
	require.NoError(t, err)

	assert.Equal(t, int64(2), outputs.searchResultData.NumQueries)
	assert.Equal(t, 2, len(outputs.searchResultData.Topks))
	t.Logf("Multi-query reranked scores: %v", outputs.searchResultData.Scores)
}

func TestExprRerankMathFunctions(t *testing.T) {
	collSchema := &schemapb.CollectionSchema{
		Name: "test_collection",
		Fields: []*schemapb.FieldSchema{
			{FieldID: 1, Name: "id", DataType: schemapb.DataType_Int64, IsPrimaryKey: true},
			{FieldID: 2, Name: "value", DataType: schemapb.DataType_Float},
		},
	}

	testCases := []struct {
		name     string
		exprCode string
		value    float32
		minScore float32
		maxScore float32
	}{
		{
			name:     "sqrt_function",
			exprCode: "score * sqrt(fields[\"value\"])",
			value:    4.0,
			minScore: 0.9, // 0.5 * sqrt(4) = 1.0
			maxScore: 1.1,
		},
		{
			name:     "log_function",
			exprCode: "score * log(fields[\"value\"])",
			value:    2.718, // e
			minScore: 0.4,
			maxScore: 0.6,
		},
		{
			name:     "exp_function",
			exprCode: "score * exp(fields[\"value\"])",
			value:    0.0, // exp(0) = 1
			minScore: 0.45,
			maxScore: 0.55,
		},
		{
			name:     "pow_function",
			exprCode: "score * pow(fields[\"value\"], 2.0)",
			value:    2.0, // 2^2 = 4
			minScore: 1.9,
			maxScore: 2.1,
		},
		{
			name:     "abs_function",
			exprCode: "score * abs(fields[\"value\"])",
			value:    -2.0, // abs(-2) = 2
			minScore: 0.9,
			maxScore: 1.1,
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			funcSchema := &schemapb.FunctionSchema{
				Name:            tc.name,
				Type:            schemapb.FunctionType_Rerank,
				InputFieldNames: []string{"value"},
				Params: []*commonpb.KeyValuePair{
					{Key: "reranker", Value: "expr"},
					{Key: "expr_code", Value: tc.exprCode},
				},
			}

			reranker, err := newExprRerank(collSchema, funcSchema)
			require.NoError(t, err, "Expression should compile: %s", tc.exprCode)

			searchParams := NewSearchParams(1, 1, 0, -1, -1, 0, false, "", []string{})

			searchResultData := &schemapb.SearchResultData{
				NumQueries: 1,
				TopK:       1,
				Ids:        &schemapb.IDs{IdField: &schemapb.IDs_IntId{IntId: &schemapb.LongArray{Data: []int64{1}}}},
				Scores:     []float32{0.5},
				Topks:      []int64{1},
				FieldsData: []*schemapb.FieldData{
					{
						FieldId:   2,
						FieldName: "value",
						Type:      schemapb.DataType_Float,
						Field: &schemapb.FieldData_Scalars{
							Scalars: &schemapb.ScalarField{
								Data: &schemapb.ScalarField_FloatData{
									FloatData: &schemapb.FloatArray{Data: []float32{tc.value}},
								},
							},
						},
					},
				},
			}

			inputs, err := newRerankInputs([]*schemapb.SearchResultData{searchResultData}, reranker.GetInputFieldIDs(), false)
			require.NoError(t, err)

			outputs, err := reranker.Process(context.Background(), searchParams, inputs)
			require.NoError(t, err)

			score := outputs.searchResultData.Scores[0]
			t.Logf("%s: input=%f, output=%f", tc.name, tc.value, score)
			assert.True(t, score >= tc.minScore && score <= tc.maxScore,
				"Score %f should be between %f and %f", score, tc.minScore, tc.maxScore)
		})
	}
}

// TestExprRerankEdgeCases tests edge cases and error conditions
func TestExprRerankEdgeCases(t *testing.T) {
	collSchema := &schemapb.CollectionSchema{
		Name: "test_collection",
		Fields: []*schemapb.FieldSchema{
			{FieldID: 1, Name: "id", DataType: schemapb.DataType_Int64, IsPrimaryKey: true},
		},
	}

	t.Run("empty_results", func(t *testing.T) {
		funcSchema := &schemapb.FunctionSchema{
			Name:            "test_empty",
			Type:            schemapb.FunctionType_Rerank,
			InputFieldNames: []string{},
			Params: []*commonpb.KeyValuePair{
				{Key: "reranker", Value: "expr"},
				{Key: "expr_code", Value: "score"},
			},
		}

		reranker, err := newExprRerank(collSchema, funcSchema)
		require.NoError(t, err)

		searchParams := NewSearchParams(1, 0, 0, -1, -1, 0, false, "", []string{})

		searchResultData := &schemapb.SearchResultData{
			NumQueries: 1,
			TopK:       0,
			Ids:        &schemapb.IDs{IdField: &schemapb.IDs_IntId{IntId: &schemapb.LongArray{Data: []int64{}}}},
			Scores:     []float32{},
			Topks:      []int64{0},
			FieldsData: []*schemapb.FieldData{},
		}

		inputs, err := newRerankInputs([]*schemapb.SearchResultData{searchResultData}, reranker.GetInputFieldIDs(), false)
		require.NoError(t, err)

		outputs, err := reranker.Process(context.Background(), searchParams, inputs)
		require.NoError(t, err)
		assert.NotNil(t, outputs)
	})

	t.Run("zero_scores", func(t *testing.T) {
		funcSchema := &schemapb.FunctionSchema{
			Name:            "test_zero",
			Type:            schemapb.FunctionType_Rerank,
			InputFieldNames: []string{},
			Params: []*commonpb.KeyValuePair{
				{Key: "reranker", Value: "expr"},
				{Key: "expr_code", Value: "score * 2.0"},
			},
		}

		reranker, err := newExprRerank(collSchema, funcSchema)
		require.NoError(t, err)

		searchParams := NewSearchParams(1, 3, 0, -1, -1, 0, false, "", []string{})

		searchResultData := &schemapb.SearchResultData{
			NumQueries: 1,
			TopK:       3,
			Ids:        &schemapb.IDs{IdField: &schemapb.IDs_IntId{IntId: &schemapb.LongArray{Data: []int64{1, 2, 3}}}},
			Scores:     []float32{0.0, 0.0, 0.0},
			Topks:      []int64{3},
			FieldsData: []*schemapb.FieldData{},
		}

		inputs, err := newRerankInputs([]*schemapb.SearchResultData{searchResultData}, reranker.GetInputFieldIDs(), false)
		require.NoError(t, err)

		outputs, err := reranker.Process(context.Background(), searchParams, inputs)
		require.NoError(t, err)

		// All scores should still be 0
		for _, score := range outputs.searchResultData.Scores {
			assert.Equal(t, float32(0.0), score)
		}
	})
}

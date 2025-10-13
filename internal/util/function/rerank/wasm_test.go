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
	"encoding/base64"
	"os"
	"path/filepath"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/milvus-io/milvus-proto/go-api/v2/commonpb"
	"github.com/milvus-io/milvus-proto/go-api/v2/schemapb"
)

// Helper function to load WASM module for testing
func loadWasmModule(t *testing.T) string {
	// Try multiple paths to find the WASM module
	possiblePaths := []string{
		"tests/rust_reranker/target/wasm32-unknown-unknown/release/rust_reranker.wasm",
		"./tests/rust_reranker/target/wasm32-unknown-unknown/release/rust_reranker.wasm",
		"../../../tests/rust_reranker/target/wasm32-unknown-unknown/release/rust_reranker.wasm",
		"../../../../tests/rust_reranker/target/wasm32-unknown-unknown/release/rust_reranker.wasm",
	}

	var wasmPath string
	for _, path := range possiblePaths {
		if _, err := os.Stat(path); err == nil {
			wasmPath = path
			break
		}
	}

	if wasmPath == "" {
		t.Skip("WASM module not found. Build it with: cd examples/rust_reranker && cargo build --target wasm32-unknown-unknown --release")
		return ""
	}

	wasmBytes, err := os.ReadFile(wasmPath)
	require.NoError(t, err, "Failed to read WASM file")

	return base64.StdEncoding.EncodeToString(wasmBytes)
}

// TestWasmRerankBasic tests basic WASM reranking without field data
func TestWasmRerankBasic(t *testing.T) {
	wasmBase64 := loadWasmModule(t)
	if wasmBase64 == "" {
		return
	}

	collSchema := &schemapb.CollectionSchema{
		Name: "test_collection",
		Fields: []*schemapb.FieldSchema{
			{FieldID: 1, Name: "id", DataType: schemapb.DataType_Int64, IsPrimaryKey: true},
		},
	}

	funcSchema := &schemapb.FunctionSchema{
		Name:            "simple_rerank",
		Type:            schemapb.FunctionType_Rerank,
		InputFieldNames: []string{}, // No field data
		Params: []*commonpb.KeyValuePair{
			{Key: "reranker", Value: "wasm"},
			{Key: "wasm_code", Value: wasmBase64},
			{Key: "entry_point", Value: "rerank"}, // Simple position-based decay
		},
	}

	reranker, err := newWasmFunction(collSchema, funcSchema)
	require.NoError(t, err)
	require.NotNil(t, reranker)

	searchParams := NewSearchParams(1, 5, 0, -1, -1, 0, false, "", []string{})

	// Create test data with uniform scores
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
	require.NotNil(t, outputs)

	scores := outputs.searchResultData.Scores
	t.Logf("WASM reranked scores (position decay): %v", scores)

	// Position decay should reduce scores for lower ranks
	for i := 1; i < len(scores); i++ {
		assert.True(t, scores[i] <= scores[i-1], "Scores should decrease with position")
	}
}

// TestWasmRerankWithPopularity tests WASM reranking with popularity field
func TestWasmRerankWithPopularity(t *testing.T) {
	wasmBase64 := loadWasmModule(t)
	if wasmBase64 == "" {
		return
	}

	collSchema := &schemapb.CollectionSchema{
		Name: "test_collection",
		Fields: []*schemapb.FieldSchema{
			{FieldID: 1, Name: "id", DataType: schemapb.DataType_Int64, IsPrimaryKey: true},
			{FieldID: 2, Name: "popularity", DataType: schemapb.DataType_Float},
		},
	}

	funcSchema := &schemapb.FunctionSchema{
		Name:            "popularity_rerank",
		Type:            schemapb.FunctionType_Rerank,
		InputFieldNames: []string{"popularity"},
		Params: []*commonpb.KeyValuePair{
			{Key: "reranker", Value: "wasm"},
			{Key: "wasm_code", Value: wasmBase64},
			{Key: "entry_point", Value: "rerank_with_popularity"},
		},
	}

	reranker, err := newWasmFunction(collSchema, funcSchema)
	require.NoError(t, err)

	searchParams := NewSearchParams(1, 4, 0, -1, -1, 0, false, "", []string{})

	searchResultData := &schemapb.SearchResultData{
		NumQueries: 1,
		TopK:       4,
		Ids:        &schemapb.IDs{IdField: &schemapb.IDs_IntId{IntId: &schemapb.LongArray{Data: []int64{1, 2, 3, 4}}}},
		Scores:     []float32{0.5, 0.5, 0.5, 0.5}, // Uniform initial scores
		Topks:      []int64{4},
		FieldsData: []*schemapb.FieldData{
			{
				FieldId:   2,
				FieldName: "popularity",
				Type:      schemapb.DataType_Float,
				Field: &schemapb.FieldData_Scalars{
					Scalars: &schemapb.ScalarField{
						Data: &schemapb.ScalarField_FloatData{
							FloatData: &schemapb.FloatArray{Data: []float32{1000.0, 100.0, 10.0, 1.0}},
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
	t.Logf("Popularity-based reranked scores: %v", scores)

	// More popular items should have higher scores (logarithmic boost)
	assert.True(t, scores[0] > scores[1], "Higher popularity should yield higher score")
	assert.True(t, scores[1] > scores[2], "Medium popularity should yield medium score")
	assert.True(t, scores[2] > scores[3], "Lower popularity should yield lower score")
}

// TestWasmRerankWithTimestamp tests WASM reranking with timestamp field
func TestWasmRerankWithTimestamp(t *testing.T) {
	wasmBase64 := loadWasmModule(t)
	if wasmBase64 == "" {
		return
	}

	collSchema := &schemapb.CollectionSchema{
		Name: "test_collection",
		Fields: []*schemapb.FieldSchema{
			{FieldID: 1, Name: "id", DataType: schemapb.DataType_Int64, IsPrimaryKey: true},
			{FieldID: 2, Name: "created_at", DataType: schemapb.DataType_Int64},
		},
	}

	funcSchema := &schemapb.FunctionSchema{
		Name:            "time_decay_rerank",
		Type:            schemapb.FunctionType_Rerank,
		InputFieldNames: []string{"created_at"},
		Params: []*commonpb.KeyValuePair{
			{Key: "reranker", Value: "wasm"},
			{Key: "wasm_code", Value: wasmBase64},
			{Key: "entry_point", Value: "rerank_with_timestamp"},
		},
	}

	reranker, err := newWasmFunction(collSchema, funcSchema)
	require.NoError(t, err)

	searchParams := NewSearchParams(1, 3, 0, -1, -1, 0, false, "", []string{})

	// Timestamps: recent, 30 days ago, 90 days ago (relative to reference time in WASM)
	referenceTime := int64(1704067200000) // Jan 1, 2024 (hardcoded in WASM)
	day := int64(1000 * 60 * 60 * 24)

	searchResultData := &schemapb.SearchResultData{
		NumQueries: 1,
		TopK:       3,
		Ids:        &schemapb.IDs{IdField: &schemapb.IDs_IntId{IntId: &schemapb.LongArray{Data: []int64{1, 2, 3}}}},
		Scores:     []float32{0.7, 0.7, 0.7},
		Topks:      []int64{3},
		FieldsData: []*schemapb.FieldData{
			{
				FieldId:   2,
				FieldName: "created_at",
				Type:      schemapb.DataType_Int64,
				Field: &schemapb.FieldData_Scalars{
					Scalars: &schemapb.ScalarField{
						Data: &schemapb.ScalarField_LongData{
							LongData: &schemapb.LongArray{Data: []int64{
								referenceTime,          // Now (no decay)
								referenceTime - 30*day, // 30 days ago
								referenceTime - 90*day, // 90 days ago
							}},
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
	t.Logf("Time decay reranked scores: %v", scores)

	// Recent items should have higher scores (exponential decay)
	assert.True(t, scores[0] > scores[1], "Recent item should have higher score than 30-day-old")
	assert.True(t, scores[1] > scores[2], "30-day-old should have higher score than 90-day-old")
}

// TestWasmRerankComplexMultiField tests WASM reranking with multiple fields
func TestWasmRerankComplexMultiField(t *testing.T) {
	wasmBase64 := loadWasmModule(t)
	if wasmBase64 == "" {
		return
	}

	collSchema := &schemapb.CollectionSchema{
		Name: "test_collection",
		Fields: []*schemapb.FieldSchema{
			{FieldID: 1, Name: "id", DataType: schemapb.DataType_Int64, IsPrimaryKey: true},
			{FieldID: 2, Name: "popularity", DataType: schemapb.DataType_Float},
			{FieldID: 3, Name: "quality_score", DataType: schemapb.DataType_Float},
		},
	}

	funcSchema := &schemapb.FunctionSchema{
		Name:            "complex_rerank",
		Type:            schemapb.FunctionType_Rerank,
		InputFieldNames: []string{"popularity", "quality_score"},
		Params: []*commonpb.KeyValuePair{
			{Key: "reranker", Value: "wasm"},
			{Key: "wasm_code", Value: wasmBase64},
			{Key: "entry_point", Value: "rerank_complex"},
		},
	}

	reranker, err := newWasmFunction(collSchema, funcSchema)
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
				FieldName: "popularity",
				Type:      schemapb.DataType_Float,
				Field: &schemapb.FieldData_Scalars{
					Scalars: &schemapb.ScalarField{
						Data: &schemapb.ScalarField_FloatData{
							FloatData: &schemapb.FloatArray{Data: []float32{500.0, 100.0, 10.0}},
						},
					},
				},
			},
			{
				FieldId:   3,
				FieldName: "quality_score",
				Type:      schemapb.DataType_Float,
				Field: &schemapb.FieldData_Scalars{
					Scalars: &schemapb.ScalarField{
						Data: &schemapb.ScalarField_FloatData{
							FloatData: &schemapb.FloatArray{Data: []float32{0.9, 0.5, 0.8}},
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
	t.Logf("Complex multi-field reranked scores: %v", scores)

	// Item 1: high popularity (500) + high quality (0.9) = highest score
	// Item 3: low popularity (10) + good quality (0.8) = medium score
	// Item 2: medium popularity (100) + medium quality (0.5) = depends on formula
	assert.True(t, scores[0] > 0.5, "High popularity + quality should boost score")
	assert.NotNil(t, outputs.searchResultData)
}

// TestWasmRerankDifferentEntryPoints tests different WASM entry points
func TestWasmRerankDifferentEntryPoints(t *testing.T) {
	wasmBase64 := loadWasmModule(t)
	if wasmBase64 == "" {
		return
	}

	collSchema := &schemapb.CollectionSchema{
		Name: "test_collection",
		Fields: []*schemapb.FieldSchema{
			{FieldID: 1, Name: "id", DataType: schemapb.DataType_Int64, IsPrimaryKey: true},
		},
	}

	entryPoints := []struct {
		name       string
		entryPoint string
	}{
		{"simple_boost", "simple_boost"},
		{"time_decay", "time_decay_rerank"},
		{"log_boost", "log_boost_rerank"},
		{"nonlinear", "nonlinear_rerank"},
	}

	for _, ep := range entryPoints {
		t.Run(ep.name, func(t *testing.T) {
			funcSchema := &schemapb.FunctionSchema{
				Name:            ep.name,
				Type:            schemapb.FunctionType_Rerank,
				InputFieldNames: []string{},
				Params: []*commonpb.KeyValuePair{
					{Key: "reranker", Value: "wasm"},
					{Key: "wasm_code", Value: wasmBase64},
					{Key: "entry_point", Value: ep.entryPoint},
				},
			}

			reranker, err := newWasmFunction(collSchema, funcSchema)
			require.NoError(t, err, "Failed to create reranker for entry point: %s", ep.entryPoint)

			searchParams := NewSearchParams(1, 3, 0, -1, -1, 0, false, "", []string{})

			searchResultData := &schemapb.SearchResultData{
				NumQueries: 1,
				TopK:       3,
				Ids:        &schemapb.IDs{IdField: &schemapb.IDs_IntId{IntId: &schemapb.LongArray{Data: []int64{1, 2, 3}}}},
				Scores:     []float32{0.5, 0.5, 0.5},
				Topks:      []int64{3},
				FieldsData: []*schemapb.FieldData{},
			}

			inputs, err := newRerankInputs([]*schemapb.SearchResultData{searchResultData}, reranker.GetInputFieldIDs(), false)
			require.NoError(t, err)

			outputs, err := reranker.Process(context.Background(), searchParams, inputs)
			require.NoError(t, err)

			t.Logf("%s scores: %v", ep.name, outputs.searchResultData.Scores)
		})
	}
}

// TestWasmRerankMultipleQueries tests WASM reranking with multiple queries
func TestWasmRerankMultipleQueries(t *testing.T) {
	wasmBase64 := loadWasmModule(t)
	if wasmBase64 == "" {
		return
	}

	collSchema := &schemapb.CollectionSchema{
		Name: "test_collection",
		Fields: []*schemapb.FieldSchema{
			{FieldID: 1, Name: "id", DataType: schemapb.DataType_Int64, IsPrimaryKey: true},
		},
	}

	funcSchema := &schemapb.FunctionSchema{
		Name:            "multi_query",
		Type:            schemapb.FunctionType_Rerank,
		InputFieldNames: []string{},
		Params: []*commonpb.KeyValuePair{
			{Key: "reranker", Value: "wasm"},
			{Key: "wasm_code", Value: wasmBase64},
			{Key: "entry_point", Value: "rerank"},
		},
	}

	reranker, err := newWasmFunction(collSchema, funcSchema)
	require.NoError(t, err)

	searchParams := NewSearchParams(2, 3, 0, -1, -1, 0, false, "", []string{})

	// Two queries with 3 results each
	searchResultData := &schemapb.SearchResultData{
		NumQueries: 2,
		TopK:       3,
		Ids:        &schemapb.IDs{IdField: &schemapb.IDs_IntId{IntId: &schemapb.LongArray{Data: []int64{1, 2, 3, 4, 5, 6}}}},
		Scores:     []float32{0.9, 0.8, 0.7, 0.6, 0.5, 0.4},
		Topks:      []int64{3, 3},
		FieldsData: []*schemapb.FieldData{},
	}

	inputs, err := newRerankInputs([]*schemapb.SearchResultData{searchResultData}, reranker.GetInputFieldIDs(), false)
	require.NoError(t, err)

	outputs, err := reranker.Process(context.Background(), searchParams, inputs)
	require.NoError(t, err)

	assert.Equal(t, int64(2), outputs.searchResultData.NumQueries)
	assert.Equal(t, 2, len(outputs.searchResultData.Topks))
	t.Logf("Multi-query WASM reranked scores: %v", outputs.searchResultData.Scores)
}

// TestWasmRerankEdgeCases tests edge cases
func TestWasmRerankEdgeCases(t *testing.T) {
	wasmBase64 := loadWasmModule(t)
	if wasmBase64 == "" {
		return
	}

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
				{Key: "reranker", Value: "wasm"},
				{Key: "wasm_code", Value: wasmBase64},
				{Key: "entry_point", Value: "rerank"},
			},
		}

		reranker, err := newWasmFunction(collSchema, funcSchema)
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

	t.Run("invalid_entry_point", func(t *testing.T) {
		funcSchema := &schemapb.FunctionSchema{
			Name:            "test_invalid",
			Type:            schemapb.FunctionType_Rerank,
			InputFieldNames: []string{},
			Params: []*commonpb.KeyValuePair{
				{Key: "reranker", Value: "wasm"},
				{Key: "wasm_code", Value: wasmBase64},
				{Key: "entry_point", Value: "nonexistent_function"},
			},
		}

		_, err := newWasmFunction(collSchema, funcSchema)
		assert.Error(t, err, "Should fail with invalid entry point")
		assert.Contains(t, err.Error(), "not found")
	})
}

// Benchmark WASM reranking performance
func BenchmarkWasmRerank(b *testing.B) {
	wasmPath := filepath.Join("examples", "rust_reranker", "target", "wasm32-unknown-unknown", "release", "rust_reranker.wasm")

	if _, err := os.Stat(wasmPath); os.IsNotExist(err) {
		b.Skip("WASM module not found")
		return
	}

	wasmBytes, err := os.ReadFile(wasmPath)
	require.NoError(b, err)
	wasmBase64 := base64.StdEncoding.EncodeToString(wasmBytes)

	collSchema := &schemapb.CollectionSchema{
		Name: "bench_collection",
		Fields: []*schemapb.FieldSchema{
			{FieldID: 1, Name: "id", DataType: schemapb.DataType_Int64, IsPrimaryKey: true},
		},
	}

	funcSchema := &schemapb.FunctionSchema{
		Name:            "bench_rerank",
		Type:            schemapb.FunctionType_Rerank,
		InputFieldNames: []string{},
		Params: []*commonpb.KeyValuePair{
			{Key: "reranker", Value: "wasm"},
			{Key: "wasm_code", Value: wasmBase64},
			{Key: "entry_point", Value: "rerank"},
		},
	}

	reranker, err := newWasmFunction(collSchema, funcSchema)
	require.NoError(b, err)

	// Create benchmark data
	numResults := 100
	ids := make([]int64, numResults)
	scores := make([]float32, numResults)
	for i := 0; i < numResults; i++ {
		ids[i] = int64(i + 1)
		scores[i] = float32(i) / float32(numResults)
	}

	searchParams := NewSearchParams(1, int64(numResults), 0, -1, -1, 0, false, "", []string{})
	searchResultData := &schemapb.SearchResultData{
		NumQueries: 1,
		TopK:       int64(numResults),
		Ids:        &schemapb.IDs{IdField: &schemapb.IDs_IntId{IntId: &schemapb.LongArray{Data: ids}}},
		Scores:     scores,
		Topks:      []int64{int64(numResults)},
		FieldsData: []*schemapb.FieldData{},
	}

	inputs, err := newRerankInputs([]*schemapb.SearchResultData{searchResultData}, reranker.GetInputFieldIDs(), false)
	require.NoError(b, err)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_, err := reranker.Process(context.Background(), searchParams, inputs)
		if err != nil {
			b.Fatal(err)
		}
	}
}

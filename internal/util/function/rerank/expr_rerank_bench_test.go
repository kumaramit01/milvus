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
	"fmt"
	"math/rand"
	"testing"

	"github.com/expr-lang/expr"
	"github.com/expr-lang/expr/vm"

	"github.com/milvus-io/milvus-proto/go-api/v2/commonpb"
	"github.com/milvus-io/milvus-proto/go-api/v2/schemapb"
)

// Isolated benchmark functions that don't require full rerank infrastructure

// BenchmarkToFloat64Conversion tests the performance of type conversion
func BenchmarkToFloat64Conversion(b *testing.B) {
	testCases := []struct {
		name  string
		value interface{}
	}{
		{"float64", float64(3.14159)},
		{"float32", float32(2.71828)},
		{"int", int(42)},
		{"int32", int32(100)},
		{"int64", int64(1000)},
		{"string", "not a number"}, // fallback case
	}

	for _, tc := range testCases {
		b.Run(tc.name, func(b *testing.B) {
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				_ = toFloat64(tc.value)
			}
		})
	}
}

// BenchmarkMathFunctionsIsolated tests individual math functions
func BenchmarkMathFunctionsIsolated(b *testing.B) {
	testCases := []struct {
		name string
		fn   func()
	}{
		{"abs", func() { _ = mathFunctions["abs"].(func(interface{}) float64)(-42.5) }},
		{"sqrt", func() { _ = mathFunctions["sqrt"].(func(interface{}) float64)(42.5) }},
		{"log", func() { _ = mathFunctions["log"].(func(interface{}) float64)(42.5) }},
		{"exp", func() { _ = mathFunctions["exp"].(func(interface{}) float64)(2.5) }},
		{"pow", func() { _ = mathFunctions["pow"].(func(interface{}, interface{}) float64)(2.5, 3.0) }},
		{"min", func() { _ = mathFunctions["min"].(func(interface{}, interface{}) float64)(10.0, 20.0) }},
		{"max", func() { _ = mathFunctions["max"].(func(interface{}, interface{}) float64)(10.0, 20.0) }},
		{"clamp", func() {
			_ = mathFunctions["clamp"].(func(interface{}, interface{}, interface{}) float64)(50.0, 0.0, 100.0)
		}},
	}

	for _, tc := range testCases {
		b.Run(tc.name, func(b *testing.B) {
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				tc.fn()
			}
		})
	}
}

// BenchmarkExpressionCompilation tests compilation performance
func BenchmarkExpressionCompilation(b *testing.B) {
	expressions := map[string]string{
		"Simple":      "score",
		"Arithmetic":  "score * 1.5 + 0.1",
		"FieldAccess": "score * fields[\"quality_score\"]",
		"MathFunc":    "score * sqrt(fields[\"quality_score\"])",
		"Conditional": "fields[\"quality_score\"] > 50 ? score * 2.0 : score",
		"Complex": `let age_days = (1704067200000 - fields["created_at"]) / 86400000;
			let recency = exp(-0.005 * age_days);
			let quality = fields["quality_score"] / 100.0;
			score * recency * quality`,
	}

	for name, exprCode := range expressions {
		b.Run(name, func(b *testing.B) {
			env := map[string]interface{}{
				"score":  float32(0),
				"rank":   int(0),
				"fields": map[string]interface{}{},
			}

			// Add math functions
			for fname, fn := range mathFunctions {
				env[fname] = fn
			}

			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				_, err := expr.Compile(exprCode, expr.Env(env))
				if err != nil {
					b.Fatal(err)
				}
			}
		})
	}
}

// BenchmarkExpressionExecution tests execution performance
func BenchmarkExpressionExecution(b *testing.B) {
	expressions := map[string]string{
		"Simple":      "score",
		"Arithmetic":  "score * 1.5 + 0.1",
		"FieldAccess": "score * fields[\"quality_score\"]",
		"MathFunc":    "score * sqrt(fields[\"quality_score\"])",
		"Conditional": "fields[\"quality_score\"] > 50 ? score * 2.0 : score",
		"Complex": `let age_days = (1704067200000 - fields["created_at"]) / 86400000;
			let recency = exp(-0.005 * age_days);
			let quality = fields["quality_score"] / 100.0;
			score * recency * quality`,
	}

	for name, exprCode := range expressions {
		b.Run(name, func(b *testing.B) {
			// Pre-compile the expression
			env := map[string]interface{}{
				"score":  float32(0),
				"rank":   int(0),
				"fields": map[string]interface{}{},
			}

			// Add math functions
			for fname, fn := range mathFunctions {
				env[fname] = fn
			}

			program, err := expr.Compile(exprCode, expr.Env(env))
			if err != nil {
				b.Fatal(err)
			}

			// Prepare runtime environment
			runtimeEnv := map[string]interface{}{
				"score": float32(0.75),
				"rank":  int(5),
				"fields": map[string]interface{}{
					"quality_score": float32(85.5),
					"popularity":    int64(1500),
					"created_at":    int64(1704067200000 - 86400000*30), // 30 days ago
				},
			}

			// Add math functions to runtime env
			for fname, fn := range mathFunctions {
				runtimeEnv[fname] = fn
			}

			b.ResetTimer()
			b.ReportAllocs()
			for i := 0; i < b.N; i++ {
				_, err := expr.Run(program, runtimeEnv)
				if err != nil {
					b.Fatal(err)
				}
			}
		})
	}
}

// BenchmarkEnvironmentCreation tests the cost of creating execution environments
func BenchmarkEnvironmentCreation(b *testing.B) {
	fieldNames := []string{"quality_score", "popularity", "created_at", "category", "rating"}

	b.Run("SmallEnv", func(b *testing.B) {
		b.ResetTimer()
		b.ReportAllocs()
		for i := 0; i < b.N; i++ {
			env := make(map[string]interface{}, 3+len(mathFunctions))
			env["score"] = float32(0.75)
			env["rank"] = int(5)

			// Add math functions (just copying references)
			for name, fn := range mathFunctions {
				env[name] = fn
			}

			// Add fields
			fields := make(map[string]interface{}, len(fieldNames))
			for _, fieldName := range fieldNames {
				fields[fieldName] = float32(50.0)
			}
			env["fields"] = fields
		}
	})

	b.Run("LargeEnv", func(b *testing.B) {
		// Simulate larger field set
		largeFieldNames := make([]string, 50)
		for i := 0; i < 50; i++ {
			largeFieldNames[i] = fmt.Sprintf("field_%d", i)
		}

		b.ResetTimer()
		b.ReportAllocs()
		for i := 0; i < b.N; i++ {
			env := make(map[string]interface{}, 3+len(mathFunctions))
			env["score"] = float32(0.75)
			env["rank"] = int(5)

			// Add math functions
			for name, fn := range mathFunctions {
				env[name] = fn
			}

			// Add many fields
			fields := make(map[string]interface{}, len(largeFieldNames))
			for _, fieldName := range largeFieldNames {
				fields[fieldName] = float32(50.0)
			}
			env["fields"] = fields
		}
	})
}

// BenchmarkDataProcessingScenarios tests realistic data processing scenarios
func BenchmarkDataProcessingScenarios(b *testing.B) {
	// Pre-compile expressions
	expressions := map[string]*vm.Program{}
	exprCodes := map[string]string{
		"QualityBoost":   "score * (fields[\"quality_score\"] / 100.0)",
		"PopularityLog":  "score * (1.0 + log(fields[\"popularity\"] + 1) / 10.0)",
		"RecencyDecay":   "score * exp(-0.01 * ((1704067200000 - fields[\"created_at\"]) / 86400000))",
		"MultiFactorial": "score * (fields[\"quality_score\"] / 100.0) * (1.0 + log(fields[\"popularity\"] + 1) / 20.0) * exp(-0.005 * ((1704067200000 - fields[\"created_at\"]) / 86400000))",
	}

	env := map[string]interface{}{
		"score":  float32(0),
		"rank":   int(0),
		"fields": map[string]interface{}{},
	}
	for fname, fn := range mathFunctions {
		env[fname] = fn
	}

	for name, exprCode := range exprCodes {
		program, err := expr.Compile(exprCode, expr.Env(env))
		if err != nil {
			b.Fatal(err)
		}
		expressions[name] = program
	}

	// Generate test data
	rand.Seed(42)
	testData := make([]map[string]interface{}, 1000)
	for i := 0; i < 1000; i++ {
		testData[i] = map[string]interface{}{
			"score": float32(rand.Float32()),
			"rank":  int(i),
			"fields": map[string]interface{}{
				"quality_score": float32(rand.Float32() * 100),
				"popularity":    int64(rand.Int63n(10000)),
				"created_at":    int64(1704067200000 - rand.Int63n(86400000*365)),
			},
		}
		// Add math functions to each test data env
		for fname, fn := range mathFunctions {
			testData[i][fname] = fn
		}
	}

	for name, program := range expressions {
		b.Run(name, func(b *testing.B) {
			b.ResetTimer()
			b.ReportAllocs()
			for i := 0; i < b.N; i++ {
				dataIdx := i % len(testData)
				_, err := expr.Run(program, testData[dataIdx])
				if err != nil {
					b.Fatal(err)
				}
			}
		})
	}
}

// Benchmark helper functions

func createBenchmarkCollectionSchema() *schemapb.CollectionSchema {
	return &schemapb.CollectionSchema{
		Name: "benchmark_collection",
		Fields: []*schemapb.FieldSchema{
			{FieldID: 1, Name: "id", DataType: schemapb.DataType_Int64, IsPrimaryKey: true},
			{FieldID: 2, Name: "quality_score", DataType: schemapb.DataType_Float},
			{FieldID: 3, Name: "popularity", DataType: schemapb.DataType_Int64},
			{FieldID: 4, Name: "created_at", DataType: schemapb.DataType_Int64},
			{FieldID: 5, Name: "category", DataType: schemapb.DataType_VarChar},
			{FieldID: 6, Name: "rating", DataType: schemapb.DataType_Float},
		},
	}
}

func createBenchmarkFunctionSchema(exprCode string) *schemapb.FunctionSchema {
	return &schemapb.FunctionSchema{
		Name:            "benchmark_expr_rerank",
		Type:            schemapb.FunctionType_Rerank,
		InputFieldNames: []string{"quality_score", "popularity", "created_at", "category", "rating"},
		Params: []*commonpb.KeyValuePair{
			{Key: "reranker", Value: "expr"},
			{Key: "expr_code", Value: exprCode},
		},
	}
}

func generateBenchmarkData(size int) []*columns {
	rand.Seed(42) // Fixed seed for reproducible benchmarks

	ids := make([]int64, size)
	scores := make([]float32, size)
	qualityScores := make([]float32, size)
	popularity := make([]int64, size)
	createdAt := make([]int64, size)
	categories := make([]string, size)
	ratings := make([]float32, size)

	categories_list := []string{"featured", "premium", "standard", "basic"}

	for i := 0; i < size; i++ {
		ids[i] = int64(i)
		scores[i] = rand.Float32() * 100
		qualityScores[i] = rand.Float32() * 100
		popularity[i] = rand.Int63n(10000)
		createdAt[i] = 1704067200000 - rand.Int63n(86400000*365) // Random time within a year
		categories[i] = categories_list[rand.Intn(len(categories_list))]
		ratings[i] = 1.0 + rand.Float32()*4.0 // Rating between 1-5
	}

	return []*columns{{
		data:   []any{qualityScores, popularity, createdAt, categories, ratings},
		size:   int64(size),
		ids:    ids,
		scores: scores,
	}}
}

func createBenchmarkSearchParams(limit int64) *SearchParams {
	return NewSearchParams(1, limit, 0, -1, 0, 0, false, "", []string{})
}

func createBenchmarkRerankInputs(cols []*columns) *rerankInputs {
	return &rerankInputs{
		data:          [][]*columns{cols},
		idGroupValue:  map[any]any{},
		nq:            1,
		fieldData:     []*schemapb.SearchResultData{},
		inputFieldIds: []int64{2, 3, 4, 5, 6}, // quality_score, popularity, created_at, category, rating
	}
}

// Expression compilation benchmarks

func BenchmarkExprRerank_Compilation_Simple(b *testing.B) {
	collSchema := createBenchmarkCollectionSchema()
	funcSchema := createBenchmarkFunctionSchema("score")

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		reranker, err := newExprRerank(collSchema, funcSchema)
		if err != nil {
			b.Fatal(err)
		}
		_ = reranker
	}
}

func BenchmarkExprRerank_Compilation_Complex(b *testing.B) {
	collSchema := createBenchmarkCollectionSchema()
	complexExpr := `let age_days = (1704067200000 - fields["created_at"]) / 86400000;
	let recency = exp(-0.005 * age_days);
	let quality = fields["quality_score"] / 100.0;
	let pop_boost = 1.0 + log(fields["popularity"] + 1) / 20.0;
	let category_boost = fields["category"] == "featured" ? 2.0 : fields["category"] == "premium" ? 1.5 : 1.0;
	score * recency * quality * pop_boost * category_boost`

	funcSchema := createBenchmarkFunctionSchema(complexExpr)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		reranker, err := newExprRerank(collSchema, funcSchema)
		if err != nil {
			b.Fatal(err)
		}
		_ = reranker
	}
}

func BenchmarkExprRerank_Compilation_MathFunctions(b *testing.B) {
	collSchema := createBenchmarkCollectionSchema()
	mathExpr := `score * sqrt(fields["quality_score"]) * pow(fields["rating"], 2.0) * 
	exp(-0.01 * abs(fields["popularity"] - 5000)) * 
	min(max(fields["quality_score"] / 100.0, 0.1), 1.0)`

	funcSchema := createBenchmarkFunctionSchema(mathExpr)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		reranker, err := newExprRerank(collSchema, funcSchema)
		if err != nil {
			b.Fatal(err)
		}
		_ = reranker
	}
}

// Expression execution benchmarks with different data sizes

func benchmarkExprRerankExecution(b *testing.B, exprCode string, dataSize int) {
	collSchema := createBenchmarkCollectionSchema()
	funcSchema := createBenchmarkFunctionSchema(exprCode)

	reranker, err := newExprRerank(collSchema, funcSchema)
	if err != nil {
		b.Fatal(err)
	}

	cols := generateBenchmarkData(dataSize)
	searchParams := createBenchmarkSearchParams(int64(dataSize))
	inputs := createBenchmarkRerankInputs(cols)
	ctx := context.Background()

	b.ResetTimer()
	b.ReportAllocs()

	for i := 0; i < b.N; i++ {
		_, err := reranker.Process(ctx, searchParams, inputs)
		if err != nil {
			b.Fatal(err)
		}
	}
}

// Simple expression benchmarks
func BenchmarkExprRerank_Simple_100(b *testing.B) {
	benchmarkExprRerankExecution(b, "score", 100)
}

func BenchmarkExprRerank_Simple_1000(b *testing.B) {
	benchmarkExprRerankExecution(b, "score", 1000)
}

func BenchmarkExprRerank_Simple_10000(b *testing.B) {
	benchmarkExprRerankExecution(b, "score", 10000)
}

// Quality boost expression benchmarks
func BenchmarkExprRerank_QualityBoost_100(b *testing.B) {
	benchmarkExprRerankExecution(b, `score * (fields["quality_score"] / 100.0)`, 100)
}

func BenchmarkExprRerank_QualityBoost_1000(b *testing.B) {
	benchmarkExprRerankExecution(b, `score * (fields["quality_score"] / 100.0)`, 1000)
}

func BenchmarkExprRerank_QualityBoost_10000(b *testing.B) {
	benchmarkExprRerankExecution(b, `score * (fields["quality_score"] / 100.0)`, 10000)
}

// Mathematical functions benchmarks
func BenchmarkExprRerank_MathFunctions_100(b *testing.B) {
	expr := `score * sqrt(fields["quality_score"]) * log(fields["popularity"] + 1)`
	benchmarkExprRerankExecution(b, expr, 100)
}

func BenchmarkExprRerank_MathFunctions_1000(b *testing.B) {
	expr := `score * sqrt(fields["quality_score"]) * log(fields["popularity"] + 1)`
	benchmarkExprRerankExecution(b, expr, 1000)
}

func BenchmarkExprRerank_MathFunctions_10000(b *testing.B) {
	expr := `score * sqrt(fields["quality_score"]) * log(fields["popularity"] + 1)`
	benchmarkExprRerankExecution(b, expr, 10000)
}

// Complex expression benchmarks
func BenchmarkExprRerank_Complex_100(b *testing.B) {
	complexExpr := `let age_days = (1704067200000 - fields["created_at"]) / 86400000;
	let recency = exp(-0.005 * age_days);
	let quality = fields["quality_score"] / 100.0;
	let pop_boost = 1.0 + log(fields["popularity"] + 1) / 20.0;
	score * recency * quality * pop_boost`
	benchmarkExprRerankExecution(b, complexExpr, 100)
}

func BenchmarkExprRerank_Complex_1000(b *testing.B) {
	complexExpr := `let age_days = (1704067200000 - fields["created_at"]) / 86400000;
	let recency = exp(-0.005 * age_days);
	let quality = fields["quality_score"] / 100.0;
	let pop_boost = 1.0 + log(fields["popularity"] + 1) / 20.0;
	score * recency * quality * pop_boost`
	benchmarkExprRerankExecution(b, complexExpr, 1000)
}

func BenchmarkExprRerank_Complex_10000(b *testing.B) {
	complexExpr := `let age_days = (1704067200000 - fields["created_at"]) / 86400000;
	let recency = exp(-0.005 * age_days);
	let quality = fields["quality_score"] / 100.0;
	let pop_boost = 1.0 + log(fields["popularity"] + 1) / 20.0;
	score * recency * quality * pop_boost`
	benchmarkExprRerankExecution(b, complexExpr, 10000)
}

// Conditional logic benchmarks
func BenchmarkExprRerank_Conditional_100(b *testing.B) {
	conditionalExpr := `let category_boost = fields["category"] == "featured" ? 2.0 : 
	fields["category"] == "premium" ? 1.5 : 
	fields["category"] == "standard" ? 1.2 : 1.0;
	let quality_filter = fields["quality_score"] > 80.0 ? 1.0 : 0.5;
	score * category_boost * quality_filter`
	benchmarkExprRerankExecution(b, conditionalExpr, 100)
}

func BenchmarkExprRerank_Conditional_1000(b *testing.B) {
	conditionalExpr := `let category_boost = fields["category"] == "featured" ? 2.0 : 
	fields["category"] == "premium" ? 1.5 : 
	fields["category"] == "standard" ? 1.2 : 1.0;
	let quality_filter = fields["quality_score"] > 80.0 ? 1.0 : 0.5;
	score * category_boost * quality_filter`
	benchmarkExprRerankExecution(b, conditionalExpr, 1000)
}

func BenchmarkExprRerank_Conditional_10000(b *testing.B) {
	conditionalExpr := `let category_boost = fields["category"] == "featured" ? 2.0 : 
	fields["category"] == "premium" ? 1.5 : 
	fields["category"] == "standard" ? 1.2 : 1.0;
	let quality_filter = fields["quality_score"] > 80.0 ? 1.0 : 0.5;
	score * category_boost * quality_filter`
	benchmarkExprRerankExecution(b, conditionalExpr, 10000)
}

// Comprehensive real-world scenario benchmark
func BenchmarkExprRerank_RealWorld_100(b *testing.B) {
	realWorldExpr := `let age_days = (1704067200000 - fields["created_at"]) / 86400000;
	let is_recent = age_days < 7;
	let is_high_quality = fields["quality_score"] > 85;
	let is_popular = fields["popularity"] > 500;
	let is_highly_rated = fields["rating"] > 4.0;
	let category_boost = fields["category"] == "featured" ? 2.0 : 
	fields["category"] == "premium" ? 1.5 : 1.0;
	let recency_boost = is_recent ? 1.5 : exp(-0.01 * age_days);
	let quality_boost = is_high_quality ? 1.3 : sqrt(fields["quality_score"] / 100.0);
	let popularity_boost = is_popular ? 1.2 : 1.0 + log(fields["popularity"] + 1) / 50.0;
	let rating_boost = is_highly_rated ? 1.1 : fields["rating"] / 5.0;
	score * category_boost * recency_boost * quality_boost * popularity_boost * rating_boost`
	benchmarkExprRerankExecution(b, realWorldExpr, 100)
}

func BenchmarkExprRerank_RealWorld_1000(b *testing.B) {
	realWorldExpr := `let age_days = (1704067200000 - fields["created_at"]) / 86400000;
	let is_recent = age_days < 7;
	let is_high_quality = fields["quality_score"] > 85;
	let is_popular = fields["popularity"] > 500;
	let is_highly_rated = fields["rating"] > 4.0;
	let category_boost = fields["category"] == "featured" ? 2.0 : 
	fields["category"] == "premium" ? 1.5 : 1.0;
	let recency_boost = is_recent ? 1.5 : exp(-0.01 * age_days);
	let quality_boost = is_high_quality ? 1.3 : sqrt(fields["quality_score"] / 100.0);
	let popularity_boost = is_popular ? 1.2 : 1.0 + log(fields["popularity"] + 1) / 50.0;
	let rating_boost = is_highly_rated ? 1.1 : fields["rating"] / 5.0;
	score * category_boost * recency_boost * quality_boost * popularity_boost * rating_boost`
	benchmarkExprRerankExecution(b, realWorldExpr, 1000)
}

func BenchmarkExprRerank_RealWorld_10000(b *testing.B) {
	realWorldExpr := `let age_days = (1704067200000 - fields["created_at"]) / 86400000;
	let is_recent = age_days < 7;
	let is_high_quality = fields["quality_score"] > 85;
	let is_popular = fields["popularity"] > 500;
	let is_highly_rated = fields["rating"] > 4.0;
	let category_boost = fields["category"] == "featured" ? 2.0 : 
	fields["category"] == "premium" ? 1.5 : 1.0;
	let recency_boost = is_recent ? 1.5 : exp(-0.01 * age_days);
	let quality_boost = is_high_quality ? 1.3 : sqrt(fields["quality_score"] / 100.0);
	let popularity_boost = is_popular ? 1.2 : 1.0 + log(fields["popularity"] + 1) / 50.0;
	let rating_boost = is_highly_rated ? 1.1 : fields["rating"] / 5.0;
	score * category_boost * recency_boost * quality_boost * popularity_boost * rating_boost`
	benchmarkExprRerankExecution(b, realWorldExpr, 10000)
}

// Type conversion benchmarks
func BenchmarkToFloat64_Float64(b *testing.B) {
	val := float64(3.14159)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = toFloat64(val)
	}
}

func BenchmarkToFloat64_Float32(b *testing.B) {
	val := float32(3.14159)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = toFloat64(val)
	}
}

func BenchmarkToFloat64_Int64(b *testing.B) {
	val := int64(42)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = toFloat64(val)
	}
}

func BenchmarkToFloat64_Int(b *testing.B) {
	val := int(42)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = toFloat64(val)
	}
}

func BenchmarkToFloat64_String(b *testing.B) {
	val := "not a number"
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = toFloat64(val)
	}
}

// Mathematical functions benchmarks
func BenchmarkMathFunctions_Abs(b *testing.B) {
	if mathFunc, ok := mathFunctions["abs"]; ok {
		fn := mathFunc.(func(interface{}) float64)
		val := -42.5
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			_ = fn(val)
		}
	}
}

func BenchmarkMathFunctions_Sqrt(b *testing.B) {
	if mathFunc, ok := mathFunctions["sqrt"]; ok {
		fn := mathFunc.(func(interface{}) float64)
		val := 42.5
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			_ = fn(val)
		}
	}
}

func BenchmarkMathFunctions_Log(b *testing.B) {
	if mathFunc, ok := mathFunctions["log"]; ok {
		fn := mathFunc.(func(interface{}) float64)
		val := 42.5
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			_ = fn(val)
		}
	}
}

func BenchmarkMathFunctions_Exp(b *testing.B) {
	if mathFunc, ok := mathFunctions["exp"]; ok {
		fn := mathFunc.(func(interface{}) float64)
		val := 2.5
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			_ = fn(val)
		}
	}
}

func BenchmarkMathFunctions_Pow(b *testing.B) {
	if mathFunc, ok := mathFunctions["pow"]; ok {
		fn := mathFunc.(func(interface{}, interface{}) float64)
		base := 2.5
		exp := 3.0
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			_ = fn(base, exp)
		}
	}
}

func BenchmarkMathFunctions_Clamp(b *testing.B) {
	if mathFunc, ok := mathFunctions["clamp"]; ok {
		fn := mathFunc.(func(interface{}, interface{}, interface{}) float64)
		val := 50.0
		min := 0.0
		max := 100.0
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			_ = fn(val, min, max)
		}
	}
}

// Memory allocation benchmarks for environment creation
func BenchmarkExprRerank_EnvironmentCreation_100(b *testing.B) {
	collSchema := createBenchmarkCollectionSchema()
	funcSchema := createBenchmarkFunctionSchema("score * fields[\"quality_score\"]")

	reranker, err := newExprRerank(collSchema, funcSchema)
	if err != nil {
		b.Fatal(err)
	}

	exprRerank := reranker.(*ExprRerank[int64])

	b.ResetTimer()
	b.ReportAllocs()

	for i := 0; i < b.N; i++ {
		// Simulate environment creation as done in processOneSearchData
		env := make(map[string]interface{}, 3+len(mathFunctions))
		env["score"] = float32(0.5)
		env["rank"] = int(0)

		// Add mathematical functions
		for name, fn := range mathFunctions {
			env[name] = fn
		}

		// Add field values
		fields := make(map[string]interface{}, len(exprRerank.inputFieldNames))
		for _, fieldName := range exprRerank.inputFieldNames {
			fields[fieldName] = float32(50.0) // Sample value
		}
		env["fields"] = fields
	}
}

// Comparative benchmarks for different expression types
func BenchmarkExprRerank_Comparison(b *testing.B) {
	expressions := map[string]string{
		"Passthrough":  "score",
		"SimpleBoost":  "score * 1.5",
		"FieldAccess":  "score * fields[\"quality_score\"]",
		"MathFunction": "score * sqrt(fields[\"quality_score\"])",
		"Conditional":  "fields[\"quality_score\"] > 50 ? score * 2.0 : score",
		"MultiMath":    "score * sqrt(fields[\"quality_score\"]) * log(fields[\"popularity\"] + 1)",
		"ComplexLogic": "let q = fields[\"quality_score\"]; let p = fields[\"popularity\"]; score * (q > 80 ? 2.0 : 1.0) * (1.0 + log(p + 1) / 10.0)",
	}

	for name, expr := range expressions {
		b.Run(name, func(b *testing.B) {
			benchmarkExprRerankExecution(b, expr, 1000)
		})
	}
}

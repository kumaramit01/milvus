package rerank

import (
	"context"
	"encoding/base64"
	"math"
	"math/rand"
	"os"
	"testing"

	"github.com/milvus-io/milvus-proto/go-api/v2/commonpb"
	"github.com/milvus-io/milvus-proto/go-api/v2/schemapb"
)

// Native Go reranking functions for comparison

func nativeRerank(score float32, rank int32) float32 {
	decayFactor := 1.0 / (1.0 + 0.05*float32(rank))
	return score * decayFactor
}

func nativeRerankWithPopularity(score float32, rank int32, popularity float32) float32 {
	popularityBoost := 1.0 + float32(math.Log1p(float64(math.Max(0.0, float64(popularity)))))/10.0
	positionDecay := 1.0 / (1.0 + 0.05*float32(rank))
	return score * popularityBoost * positionDecay
}

func nativeRerankComplex(score float32, rank int32, popularity, quality float32) float32 {
	popularityFactor := 1.0 + float32(math.Log1p(float64(math.Max(0.0, float64(popularity)))))/20.0
	qualityFactor := 0.7 + 0.6*float32(math.Max(0.0, math.Min(1.0, float64(quality))))
	combinedBoost := float32(math.Sqrt(float64(popularityFactor * qualityFactor)))
	positionFactor := 1.0 / (1.0 + 0.04*float32(rank))
	return score * combinedBoost * positionFactor
}

// Helper to load Rust WASM module
func loadRustWasmModule(t testing.TB) []byte {
	// Try to load the pre-built Rust WASM module
	wasmPath := "../tests/reranker_wasm/rust_reranker/target/wasm32-unknown-unknown/release/rust_reranker.wasm"
	wasmBytes, err := os.ReadFile(wasmPath)
	if err != nil {
		t.Skipf("Skipping WASM benchmark: Rust WASM module not found at %s. Build it with: cd ../tests/reranker_wasm/rust_reranker && cargo build --target wasm32-unknown-unknown --release", wasmPath)
		return nil
	}
	return wasmBytes
}

// Helper function to generate test data for WASM benchmarks
func generateWasmBenchmarkData(size int, includePopularity, includeQuality bool) []*columns {
	rand.Seed(42) // Fixed seed for reproducible benchmarks

	ids := make([]int64, size)
	scores := make([]float32, size)
	var data []any

	for i := 0; i < size; i++ {
		ids[i] = int64(i)
		scores[i] = rand.Float32() * 100
	}

	if includePopularity {
		popularity := make([]float32, size)
		for i := 0; i < size; i++ {
			popularity[i] = rand.Float32() * 200
		}
		data = append(data, popularity)
	}

	if includeQuality {
		quality := make([]float32, size)
		for i := 0; i < size; i++ {
			quality[i] = rand.Float32()
		}
		data = append(data, quality)
	}

	return []*columns{{
		data:   data,
		size:   int64(size),
		ids:    ids,
		scores: scores,
	}}
}

// Benchmark: Native Go simple rerank
func BenchmarkNativeGoSimpleRerank(b *testing.B) {
	score := float32(0.8)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = nativeRerank(score, int32(i%100))
	}

	opsPerSec := float64(b.N) / b.Elapsed().Seconds()
	nsPerOp := b.Elapsed().Nanoseconds() / int64(b.N)
	b.ReportMetric(opsPerSec, "ops/sec")
	b.ReportMetric(float64(nsPerOp), "ns/op")
}

// Benchmark: Native Go with popularity
func BenchmarkNativeGoWithPopularity(b *testing.B) {
	score := float32(0.8)
	popularity := float32(100.0)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = nativeRerankWithPopularity(score, int32(i%100), popularity)
	}

	opsPerSec := float64(b.N) / b.Elapsed().Seconds()
	nsPerOp := b.Elapsed().Nanoseconds() / int64(b.N)
	b.ReportMetric(opsPerSec, "ops/sec")
	b.ReportMetric(float64(nsPerOp), "ns/op")
}

// Benchmark: Native Go complex
func BenchmarkNativeGoComplex(b *testing.B) {
	score := float32(0.8)
	popularity := float32(100.0)
	quality := float32(0.8)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = nativeRerankComplex(score, int32(i%100), popularity, quality)
	}

	opsPerSec := float64(b.N) / b.Elapsed().Seconds()
	nsPerOp := b.Elapsed().Nanoseconds() / int64(b.N)
	b.ReportMetric(opsPerSec, "ops/sec")
	b.ReportMetric(float64(nsPerOp), "ns/op")
}

// Benchmark: WASM simple rerank using Process method
func BenchmarkWasmSimpleRerank_100(b *testing.B) {
	wasmBytes := loadRustWasmModule(b)
	if wasmBytes == nil {
		return
	}

	wasmBase64 := base64.StdEncoding.EncodeToString(wasmBytes)

	collSchema := &schemapb.CollectionSchema{
		Name: "test_collection",
		Fields: []*schemapb.FieldSchema{
			{FieldID: 100, Name: "id", DataType: schemapb.DataType_Int64, IsPrimaryKey: true},
			{FieldID: 101, Name: "vector", DataType: schemapb.DataType_FloatVector},
		},
	}

	funcSchema := &schemapb.FunctionSchema{
		Name: "test_wasm",
		Type: schemapb.FunctionType_Rerank,
		Params: []*commonpb.KeyValuePair{
			{Key: "reranker", Value: "wasm"},
			{Key: "wasm_code", Value: wasmBase64},
			{Key: "entry_point", Value: "rerank"},
		},
	}

	wasmFunc, err := newWasmFunction(collSchema, funcSchema)
	if err != nil {
		b.Fatalf("Failed to create WASM function: %v", err)
	}

	// Generate test data
	dataSize := 100
	cols := generateWasmBenchmarkData(dataSize, false, false)
	searchParams := createBenchmarkSearchParams(int64(dataSize))
	inputs := createBenchmarkRerankInputs(cols)
	ctx := context.Background()

	b.ResetTimer()
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		_, err := wasmFunc.Process(ctx, searchParams, inputs)
		if err != nil {
			b.Fatalf("WASM Process failed: %v", err)
		}
	}
}

// Benchmark: WASM with popularity using Process method
func BenchmarkWasmWithPopularity_100(b *testing.B) {
	wasmBytes := loadRustWasmModule(b)
	if wasmBytes == nil {
		return
	}

	wasmBase64 := base64.StdEncoding.EncodeToString(wasmBytes)

	collSchema := &schemapb.CollectionSchema{
		Name: "test_collection",
		Fields: []*schemapb.FieldSchema{
			{FieldID: 100, Name: "id", DataType: schemapb.DataType_Int64, IsPrimaryKey: true},
			{FieldID: 101, Name: "vector", DataType: schemapb.DataType_FloatVector},
			{FieldID: 102, Name: "popularity", DataType: schemapb.DataType_Float},
		},
	}

	funcSchema := &schemapb.FunctionSchema{
		Name:            "test_wasm",
		Type:            schemapb.FunctionType_Rerank,
		InputFieldNames: []string{"popularity"},
		Params: []*commonpb.KeyValuePair{
			{Key: "reranker", Value: "wasm"},
			{Key: "wasm_code", Value: wasmBase64},
			{Key: "entry_point", Value: "rerank_with_popularity"},
		},
	}

	wasmFunc, err := newWasmFunction(collSchema, funcSchema)
	if err != nil {
		b.Fatalf("Failed to create WASM function: %v", err)
	}

	// Generate test data with popularity
	dataSize := 100
	cols := generateWasmBenchmarkData(dataSize, true, false)
	searchParams := createBenchmarkSearchParams(int64(dataSize))
	inputs := &rerankInputs{
		data:          [][]*columns{cols},
		idGroupValue:  map[any]any{},
		nq:            1,
		fieldData:     []*schemapb.SearchResultData{},
		inputFieldIds: []int64{102}, // popularity field
	}
	ctx := context.Background()

	b.ResetTimer()
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		_, err := wasmFunc.Process(ctx, searchParams, inputs)
		if err != nil {
			b.Fatalf("WASM Process failed: %v", err)
		}
	}
}

// Benchmark: WASM complex using Process method
func BenchmarkWasmComplex_100(b *testing.B) {
	wasmBytes := loadRustWasmModule(b)
	if wasmBytes == nil {
		return
	}

	wasmBase64 := base64.StdEncoding.EncodeToString(wasmBytes)

	collSchema := &schemapb.CollectionSchema{
		Name: "test_collection",
		Fields: []*schemapb.FieldSchema{
			{FieldID: 100, Name: "id", DataType: schemapb.DataType_Int64, IsPrimaryKey: true},
			{FieldID: 101, Name: "vector", DataType: schemapb.DataType_FloatVector},
			{FieldID: 102, Name: "popularity", DataType: schemapb.DataType_Float},
			{FieldID: 103, Name: "quality", DataType: schemapb.DataType_Float},
		},
	}

	funcSchema := &schemapb.FunctionSchema{
		Name:            "test_wasm",
		Type:            schemapb.FunctionType_Rerank,
		InputFieldNames: []string{"popularity", "quality"},
		Params: []*commonpb.KeyValuePair{
			{Key: "reranker", Value: "wasm"},
			{Key: "wasm_code", Value: wasmBase64},
			{Key: "entry_point", Value: "rerank_complex"},
		},
	}

	wasmFunc, err := newWasmFunction(collSchema, funcSchema)
	if err != nil {
		b.Fatalf("Failed to create WASM function: %v", err)
	}

	// Generate test data with popularity and quality
	dataSize := 100
	cols := generateWasmBenchmarkData(dataSize, true, true)
	searchParams := createBenchmarkSearchParams(int64(dataSize))
	inputs := &rerankInputs{
		data:          [][]*columns{cols},
		idGroupValue:  map[any]any{},
		nq:            1,
		fieldData:     []*schemapb.SearchResultData{},
		inputFieldIds: []int64{102, 103}, // popularity and quality fields
	}
	ctx := context.Background()

	b.ResetTimer()
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		_, err := wasmFunc.Process(ctx, searchParams, inputs)
		if err != nil {
			b.Fatalf("WASM Process failed: %v", err)
		}
	}
}

// Benchmark: Full reranking pipeline with native Go
func BenchmarkFullPipelineNativeGo(b *testing.B) {
	// Simulate 100 search results
	numResults := 100
	scores := make([]float32, numResults)
	popularities := make([]float32, numResults)
	qualities := make([]float32, numResults)

	for i := 0; i < numResults; i++ {
		scores[i] = 0.9 - float32(i)*0.005
		popularities[i] = 100.0 - float32(i)
		qualities[i] = 0.8
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		rerankedScores := make([]float32, numResults)
		for j := 0; j < numResults; j++ {
			rerankedScores[j] = nativeRerankComplex(scores[j], int32(j), popularities[j], qualities[j])
		}
		_ = rerankedScores
	}

	totalOps := int64(b.N) * int64(numResults)
	opsPerSec := float64(totalOps) / b.Elapsed().Seconds()
	nsPerOp := b.Elapsed().Nanoseconds() / totalOps
	b.ReportMetric(opsPerSec, "ops/sec")
	b.ReportMetric(float64(nsPerOp), "ns/op")
	b.ReportMetric(float64(b.N)/b.Elapsed().Seconds(), "queries/sec")
}

// Benchmark: Full reranking pipeline with WASM
func BenchmarkFullPipelineWasm(b *testing.B) {
	wasmBytes := loadRustWasmModule(b)
	if wasmBytes == nil {
		return
	}

	wasmBase64 := base64.StdEncoding.EncodeToString(wasmBytes)

	collSchema := &schemapb.CollectionSchema{
		Name: "test_collection",
		Fields: []*schemapb.FieldSchema{
			{FieldID: 100, Name: "id", DataType: schemapb.DataType_Int64, IsPrimaryKey: true},
			{FieldID: 101, Name: "vector", DataType: schemapb.DataType_FloatVector},
			{FieldID: 102, Name: "popularity", DataType: schemapb.DataType_Float},
			{FieldID: 103, Name: "quality", DataType: schemapb.DataType_Float},
		},
	}

	funcSchema := &schemapb.FunctionSchema{
		Name:            "test_wasm",
		Type:            schemapb.FunctionType_Rerank,
		InputFieldNames: []string{"popularity", "quality"},
		Params: []*commonpb.KeyValuePair{
			{Key: "reranker", Value: "wasm"},
			{Key: "wasm_code", Value: wasmBase64},
			{Key: "entry_point", Value: "rerank_complex"},
		},
	}

	wasmFunc, err := newWasmFunction(collSchema, funcSchema)
	if err != nil {
		b.Fatalf("Failed to create WASM function: %v", err)
	}

	// Generate test data with popularity and quality
	dataSize := 100
	cols := generateWasmBenchmarkData(dataSize, true, true)
	searchParams := createBenchmarkSearchParams(int64(dataSize))
	inputs := &rerankInputs{
		data:          [][]*columns{cols},
		idGroupValue:  map[any]any{},
		nq:            1,
		fieldData:     []*schemapb.SearchResultData{},
		inputFieldIds: []int64{102, 103}, // popularity and quality fields
	}
	ctx := context.Background()

	b.ResetTimer()
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		_, err := wasmFunc.Process(ctx, searchParams, inputs)
		if err != nil {
			b.Fatalf("WASM Process failed: %v", err)
		}
	}

	totalOps := int64(b.N) * int64(dataSize)
	opsPerSec := float64(totalOps) / b.Elapsed().Seconds()
	nsPerOp := b.Elapsed().Nanoseconds() / totalOps
	b.ReportMetric(opsPerSec, "ops/sec")
	b.ReportMetric(float64(nsPerOp), "ns/op")
	b.ReportMetric(float64(b.N)/b.Elapsed().Seconds(), "queries/sec")
}

// Benchmark: WASM module compilation overhead
func BenchmarkWasmModuleCompilation(b *testing.B) {
	wasmBytes := loadRustWasmModule(b)
	if wasmBytes == nil {
		return
	}

	wasmBase64 := base64.StdEncoding.EncodeToString(wasmBytes)

	collSchema := &schemapb.CollectionSchema{
		Name: "test_collection",
		Fields: []*schemapb.FieldSchema{
			{FieldID: 100, Name: "id", DataType: schemapb.DataType_Int64, IsPrimaryKey: true},
			{FieldID: 101, Name: "vector", DataType: schemapb.DataType_FloatVector},
		},
	}

	funcSchema := &schemapb.FunctionSchema{
		Name: "test_wasm",
		Type: schemapb.FunctionType_Rerank,
		Params: []*commonpb.KeyValuePair{
			{Key: "reranker", Value: "wasm"},
			{Key: "wasm_code", Value: wasmBase64},
			{Key: "entry_point", Value: "rerank"},
		},
	}

	b.ResetTimer()
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		_, err := newWasmFunction(collSchema, funcSchema)
		if err != nil {
			b.Fatalf("Failed to create WASM function: %v", err)
		}
	}

	msPerOp := float64(b.Elapsed().Milliseconds()) / float64(b.N)
	b.ReportMetric(msPerOp, "ms/compilation")
}

// Benchmark: WASM with different data sizes
func BenchmarkWasmRerank_1000(b *testing.B) {
	wasmBytes := loadRustWasmModule(b)
	if wasmBytes == nil {
		return
	}

	wasmBase64 := base64.StdEncoding.EncodeToString(wasmBytes)

	collSchema := &schemapb.CollectionSchema{
		Name: "test_collection",
		Fields: []*schemapb.FieldSchema{
			{FieldID: 100, Name: "id", DataType: schemapb.DataType_Int64, IsPrimaryKey: true},
			{FieldID: 101, Name: "vector", DataType: schemapb.DataType_FloatVector},
		},
	}

	funcSchema := &schemapb.FunctionSchema{
		Name: "test_wasm",
		Type: schemapb.FunctionType_Rerank,
		Params: []*commonpb.KeyValuePair{
			{Key: "reranker", Value: "wasm"},
			{Key: "wasm_code", Value: wasmBase64},
			{Key: "entry_point", Value: "rerank"},
		},
	}

	wasmFunc, err := newWasmFunction(collSchema, funcSchema)
	if err != nil {
		b.Fatalf("Failed to create WASM function: %v", err)
	}

	dataSize := 1000
	cols := generateWasmBenchmarkData(dataSize, false, false)
	searchParams := createBenchmarkSearchParams(int64(dataSize))
	inputs := createBenchmarkRerankInputs(cols)
	ctx := context.Background()

	b.ResetTimer()
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		_, err := wasmFunc.Process(ctx, searchParams, inputs)
		if err != nil {
			b.Fatalf("WASM Process failed: %v", err)
		}
	}
}

func BenchmarkWasmRerank_10000(b *testing.B) {
	wasmBytes := loadRustWasmModule(b)
	if wasmBytes == nil {
		return
	}

	wasmBase64 := base64.StdEncoding.EncodeToString(wasmBytes)

	collSchema := &schemapb.CollectionSchema{
		Name: "test_collection",
		Fields: []*schemapb.FieldSchema{
			{FieldID: 100, Name: "id", DataType: schemapb.DataType_Int64, IsPrimaryKey: true},
			{FieldID: 101, Name: "vector", DataType: schemapb.DataType_FloatVector},
		},
	}

	funcSchema := &schemapb.FunctionSchema{
		Name: "test_wasm",
		Type: schemapb.FunctionType_Rerank,
		Params: []*commonpb.KeyValuePair{
			{Key: "reranker", Value: "wasm"},
			{Key: "wasm_code", Value: wasmBase64},
			{Key: "entry_point", Value: "rerank"},
		},
	}

	wasmFunc, err := newWasmFunction(collSchema, funcSchema)
	if err != nil {
		b.Fatalf("Failed to create WASM function: %v", err)
	}

	dataSize := 10000
	cols := generateWasmBenchmarkData(dataSize, false, false)
	searchParams := createBenchmarkSearchParams(int64(dataSize))
	inputs := createBenchmarkRerankInputs(cols)
	ctx := context.Background()

	b.ResetTimer()
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		_, err := wasmFunc.Process(ctx, searchParams, inputs)
		if err != nil {
			b.Fatalf("WASM Process failed: %v", err)
		}
	}
}

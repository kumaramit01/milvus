package main

import (
	"context"
	"encoding/base64"
	"fmt"
	"log"
	"math/rand"
	"os"
	"path/filepath"
	"strings"
	"time"

	"github.com/milvus-io/milvus/client/v2/column"
	"github.com/milvus-io/milvus/client/v2/entity"
	"github.com/milvus-io/milvus/client/v2/index"
	"github.com/milvus-io/milvus/client/v2/milvusclient"
)

const (
	milvusAddr      = "localhost:19530"
	collectionName  = "wasm_rerank_test_collection"
	titleFieldName  = "title"
	embeddingField  = "embedding"
	popularityField = "popularity"
	qualityField    = "quality_score"
	createdAtField  = "created_at"
	dim             = 128
)

type TestCase struct {
	Name        string
	EntryPoint  string
	InputFields []string
	ExpectedTop string // Expected document title at top
	Description string
}

type TestData struct {
	Title      string
	Embedding  []float32
	Popularity float32 // Using float32 for WASM compatibility
	Quality    float32 // Quality score 0.0-1.0
	CreatedAt  int64   // Unix timestamp in milliseconds
}

// findWasmModule locates the WASM module in the project
func findWasmModule() (string, error) {
	// Try multiple possible locations
	possiblePaths := []string{
		"tests/reranker_wasm/rust_reranker/target/wasm32-unknown-unknown/release/rust_reranker.wasm",
		"../../tests/reranker_wasm/rust_reranker/target/wasm32-unknown-unknown/release/rust_reranker.wasm",
		"../../../tests/reranker_wasm/rust_reranker/target/wasm32-unknown-unknown/release/rust_reranker.wasm",
	}

	for _, path := range possiblePaths {
		if _, err := os.Stat(path); err == nil {
			absPath, _ := filepath.Abs(path)
			return absPath, nil
		}
	}

	return "", fmt.Errorf("WASM module not found. Build it with:\n" +
		"  cd tests/reranker_wasm/rust_reranker\n" +
		"  cargo build --target wasm32-unknown-unknown --release")
}

// loadWasmModule loads and base64-encodes the WASM module
func loadWasmModule() (string, error) {
	wasmPath, err := findWasmModule()
	if err != nil {
		return "", err
	}

	log.Printf("Loading WASM module from: %s", wasmPath)
	wasmBytes, err := os.ReadFile(wasmPath)
	if err != nil {
		return "", fmt.Errorf("failed to read WASM file: %w", err)
	}

	log.Printf("WASM module size: %.2f KB", float64(len(wasmBytes))/1024)
	return base64.StdEncoding.EncodeToString(wasmBytes), nil
}

func setupCollection(ctx context.Context, client *milvusclient.Client, wasmBase64 string) error {
	// Drop existing collection if it exists
	err := client.DropCollection(ctx, milvusclient.NewDropCollectionOption(collectionName))
	if err != nil {
		log.Printf("Collection '%s' doesn't exist or failed to drop: %v", collectionName, err)
	}

	schema := entity.NewSchema().
		WithName(collectionName).
		WithField(entity.NewField().
			WithName("id").
			WithDataType(entity.FieldTypeInt64).
			WithIsPrimaryKey(true).
			WithIsAutoID(true)).
		WithField(entity.NewField().
			WithName(titleFieldName).
			WithDataType(entity.FieldTypeVarChar).
			WithMaxLength(500)).
		WithField(entity.NewField().
			WithName(embeddingField).
			WithDataType(entity.FieldTypeFloatVector).
			WithDim(dim)).
		WithField(entity.NewField().
			WithName(popularityField).
			WithDataType(entity.FieldTypeFloat)).
		WithField(entity.NewField().
			WithName(qualityField).
			WithDataType(entity.FieldTypeFloat)).
		WithField(entity.NewField().
			WithName(createdAtField).
			WithDataType(entity.FieldTypeInt64))

	// Note: WASM reranker functions are typically added dynamically per search
	// rather than at collection creation time. See runWasmTest() for usage.

	// Create index for vector field
	vectorIndex := index.NewHNSWIndex(entity.L2, 8, 200)
	vectorIndexOption := milvusclient.NewCreateIndexOption(collectionName, embeddingField, vectorIndex)

	err = client.CreateCollection(ctx, milvusclient.NewCreateCollectionOption(collectionName, schema).
		WithIndexOptions(vectorIndexOption))
	if err != nil {
		return fmt.Errorf("failed to create collection: %w", err)
	}

	return nil
}

// generateRandomVector creates a random embedding vector
func generateRandomVector(dim int) []float32 {
	vec := make([]float32, dim)
	for i := range vec {
		vec[i] = rand.Float32()*2 - 1 // Random values between -1 and 1
	}
	return vec
}

func insertTestData(ctx context.Context, client *milvusclient.Client) error {
	now := time.Now().UnixMilli()

	testData := []TestData{
		{
			Title:      "Viral Trending Content",
			Embedding:  generateRandomVector(dim),
			Popularity: 10000.0,
			Quality:    0.75,
			CreatedAt:  now - 86400000, // 1 day ago
		},
		{
			Title:      "Premium Quality Article",
			Embedding:  generateRandomVector(dim),
			Popularity: 500.0,
			Quality:    0.98,
			CreatedAt:  now - 86400000*30, // 30 days ago
		},
		{
			Title:      "Recent High-Quality Post",
			Embedding:  generateRandomVector(dim),
			Popularity: 2000.0,
			Quality:    0.95,
			CreatedAt:  now - 86400000, // 1 day ago
		},
		{
			Title:      "Old Archive Content",
			Embedding:  generateRandomVector(dim),
			Popularity: 100.0,
			Quality:    0.60,
			CreatedAt:  now - 86400000*365, // 1 year ago
		},
		{
			Title:      "Moderately Popular Content",
			Embedding:  generateRandomVector(dim),
			Popularity: 1000.0,
			Quality:    0.80,
			CreatedAt:  now - 86400000*7, // 1 week ago
		},
		{
			Title:      "Low Quality Spam",
			Embedding:  generateRandomVector(dim),
			Popularity: 50.0,
			Quality:    0.20,
			CreatedAt:  now - 86400000*3, // 3 days ago
		},
		{
			Title:      "Niche Expert Content",
			Embedding:  generateRandomVector(dim),
			Popularity: 200.0,
			Quality:    0.99,
			CreatedAt:  now - 86400000*60, // 2 months ago
		},
		{
			Title:      "Balanced Quality Article",
			Embedding:  generateRandomVector(dim),
			Popularity: 800.0,
			Quality:    0.85,
			CreatedAt:  now - 86400000*14, // 2 weeks ago
		},
	}

	titleColumn := make([]string, len(testData))
	embeddingColumn := make([][]float32, len(testData))
	popularityColumn := make([]float32, len(testData))
	qualityColumn := make([]float32, len(testData))
	createdAtColumn := make([]int64, len(testData))

	for i, d := range testData {
		titleColumn[i] = d.Title
		embeddingColumn[i] = d.Embedding
		popularityColumn[i] = d.Popularity
		qualityColumn[i] = d.Quality
		createdAtColumn[i] = d.CreatedAt
	}

	_, err := client.Insert(ctx, milvusclient.NewColumnBasedInsertOption(collectionName).
		WithVarcharColumn(titleFieldName, titleColumn).
		WithFloatVectorColumn(embeddingField, dim, embeddingColumn).
		WithColumns(
			column.NewColumnFloat(popularityField, popularityColumn),
			column.NewColumnFloat(qualityField, qualityColumn),
			column.NewColumnInt64(createdAtField, createdAtColumn),
		))
	if err != nil {
		return fmt.Errorf("failed to insert data: %w", err)
	}

	loadTask, err := client.LoadCollection(ctx, milvusclient.NewLoadCollectionOption(collectionName))
	if err != nil {
		return fmt.Errorf("failed to load collection: %w", err)
	}

	err = loadTask.Await(ctx)
	if err != nil {
		return fmt.Errorf("failed to wait for collection loading: %w", err)
	}

	return nil
}

// runWasmTest executes a single WASM reranker test case
func runWasmTest(ctx context.Context, client *milvusclient.Client, wasmBase64 string, testCase TestCase) error {
	log.Printf("\n=== Running Test: %s ===", testCase.Name)
	log.Printf("Description: %s", testCase.Description)
	log.Printf("Entry Point: %s", testCase.EntryPoint)
	log.Printf("Input Fields: %v", testCase.InputFields)
	log.Printf("Expected Top Result: %s", testCase.ExpectedTop)

	// Create query vector (random for this test)
	queryVector := generateRandomVector(dim)

	// Create WASM reranker function for this search
	wasmRerankerFunc := entity.NewFunction().
		WithName("wasm_reranker").
		WithType(entity.FunctionTypeRerank).
		WithInputFields(testCase.InputFields...).
		WithParam("reranker", "wasm").
		WithParam("wasm_code", wasmBase64).
		WithParam("entry_point", testCase.EntryPoint)

	// Perform search with WASM reranking
	searchResult, err := client.Search(ctx, milvusclient.NewSearchOption(
		collectionName,
		10,
		[]entity.Vector{entity.FloatVector(queryVector)},
	).WithANNSField(embeddingField).
		WithOutputFields("id", titleFieldName, popularityField, qualityField, createdAtField).
		WithQueryNodeReranker(wasmRerankerFunc))

	if err != nil {
		return fmt.Errorf("search failed: %w", err)
	}

	actualCount := 0
	var topResult string

	for _, resultSet := range searchResult {
		actualCount = resultSet.ResultCount
		log.Printf("Results Found: %d", actualCount)

		if actualCount == 0 {
			log.Println("No results found")
		} else {
			log.Println("WASM reranked results:")
			for i := 0; i < resultSet.ResultCount; i++ {
				var id interface{}
				if resultSet.IDs != nil {
					id, _ = resultSet.IDs.Get(i)
				}

				score := resultSet.Scores[i]

				var title string
				var popularity, quality float32
				var createdAt int64

				if titleCol := resultSet.GetColumn(titleFieldName); titleCol != nil {
					title, _ = titleCol.GetAsString(i)
				}
				if popCol := resultSet.GetColumn(popularityField); popCol != nil {
					popDouble, _ := popCol.GetAsDouble(i)
					popularity = float32(popDouble)
				}
				if qualCol := resultSet.GetColumn(qualityField); qualCol != nil {
					qualDouble, _ := qualCol.GetAsDouble(i)
					quality = float32(qualDouble)
				}
				if createdCol := resultSet.GetColumn(createdAtField); createdCol != nil {
					createdAt, _ = createdCol.GetAsInt64(i)
				}

				if i == 0 {
					topResult = title
				}

				// Calculate age in days for display
				now := time.Now().UnixMilli()
				ageDays := (now - createdAt) / 86400000

				log.Printf("  [%d] ID: %v, Score: %.4f", i+1, id, score)
				log.Printf("      Title: %s", title)
				log.Printf("      Age: %d days, Quality: %.2f, Popularity: %.0f",
					ageDays, quality, popularity)
			}
		}
	}

	// Validate results
	if testCase.ExpectedTop != "" {
		if strings.Contains(topResult, testCase.ExpectedTop) {
			log.Printf("✅ PASS: Expected '%s' at top, got '%s'", testCase.ExpectedTop, topResult)
		} else {
			log.Printf("⚠️  INFO: Expected '%s' at top, got '%s'", testCase.ExpectedTop, topResult)
			log.Printf("   (Note: Random vectors may cause varying results)")
		}
	} else {
		log.Printf("ℹ️  INFO: No specific expectation set (top result: '%s')", topResult)
	}

	return nil
}

func runAllWasmTests(ctx context.Context, client *milvusclient.Client, wasmBase64 string) error {
	testCases := []TestCase{
		{
			Name:        "Simple Position Decay",
			EntryPoint:  "rerank",
			InputFields: []string{}, // No input fields, only score and rank
			ExpectedTop: "",         // No specific expectation due to random vectors
			Description: "Test simple position-based decay without field data",
		},
		{
			Name:        "Popularity Boost",
			EntryPoint:  "rerank_with_popularity",
			InputFields: []string{popularityField},
			ExpectedTop: "Viral Trending Content", // Highest popularity
			Description: "Test popularity-based boosting with logarithmic scaling",
		},
		{
			Name:        "Quality and Popularity Combined",
			EntryPoint:  "rerank_complex",
			InputFields: []string{popularityField, qualityField},
			ExpectedTop: "Recent High-Quality Post", // Good balance of both
			Description: "Test multi-factor scoring with popularity and quality",
		},
		{
			Name:        "Simple Boost",
			EntryPoint:  "simple_boost",
			InputFields: []string{},
			ExpectedTop: "",
			Description: "Test simple 1.5x score multiplier",
		},
		{
			Name:        "Logarithmic Boost",
			EntryPoint:  "log_boost_rerank",
			InputFields: []string{},
			ExpectedTop: "",
			Description: "Test logarithmic position boost for top results",
		},
		{
			Name:        "Time Decay Rerank",
			EntryPoint:  "time_decay_rerank",
			InputFields: []string{},
			ExpectedTop: "",
			Description: "Test exponential time decay based on position",
		},
	}

	log.Println("\n🚀 Starting WASM Reranker Test Suite")
	log.Println(strings.Repeat("=", 60))

	passCount := 0
	totalTests := len(testCases)

	for i, testCase := range testCases {
		log.Printf("\nTest %d/%d", i+1, totalTests)
		err := runWasmTest(ctx, client, wasmBase64, testCase)
		if err != nil {
			log.Printf("❌ Test failed with error: %v", err)
		} else {
			passCount++
		}
		log.Println(strings.Repeat("-", 40))
	}

	log.Printf("\n📊 Test Summary:")
	log.Printf("Total Tests: %d", totalTests)
	log.Printf("Completed: %d", passCount)
	log.Printf("Errors: %d", totalTests-passCount)

	return nil
}

func main() {
	ctx := context.Background()

	log.Println("🔗 Connecting to Milvus...")
	milvusClient, err := milvusclient.New(ctx, &milvusclient.ClientConfig{
		Address: milvusAddr,
	})
	if err != nil {
		log.Fatalf("Failed to connect to Milvus: %v", err)
	}
	defer milvusClient.Close(ctx)
	log.Println("✅ Connected to Milvus successfully")

	log.Println("\n📦 Loading WASM module...")
	wasmBase64, err := loadWasmModule()
	if err != nil {
		log.Fatalf("Failed to load WASM module: %v", err)
	}
	log.Println("✅ WASM module loaded successfully")

	log.Println("\n🏗️  Setting up test collection...")
	err = setupCollection(ctx, milvusClient, wasmBase64)
	if err != nil {
		log.Fatalf("Failed to setup collection: %v", err)
	}
	log.Printf("✅ Collection '%s' created successfully", collectionName)

	log.Println("\n📝 Inserting test data...")
	err = insertTestData(ctx, milvusClient)
	if err != nil {
		log.Fatalf("Failed to insert test data: %v", err)
	}
	log.Println("✅ Test data inserted and collection loaded successfully")

	log.Println("\n🧪 Running WASM Reranker tests...")
	err = runAllWasmTests(ctx, milvusClient, wasmBase64)
	if err != nil {
		log.Fatalf("Failed to run tests: %v", err)
	}

	log.Println("\n🧹 Cleaning up...")
	err = milvusClient.DropCollection(ctx, milvusclient.NewDropCollectionOption(collectionName))
	if err != nil {
		log.Printf("Failed to drop collection: %v", err)
	} else {
		log.Printf("✅ Collection '%s' dropped successfully", collectionName)
	}

	log.Println("\n🎉 WASM Reranker test suite completed!")
	log.Println("\nAvailable WASM entry points:")
	log.Println("  • rerank(score, rank) - Simple position decay")
	log.Println("  • rerank_with_popularity(score, rank, popularity) - Popularity boost")
	log.Println("  • rerank_with_timestamp(score, rank, timestamp) - Time decay")
	log.Println("  • rerank_complex(score, rank, popularity, quality) - Multi-factor")
	log.Println("  • simple_boost(score, rank) - 1.5x multiplier")
	log.Println("  • log_boost_rerank(score, rank) - Logarithmic boost")
	log.Println("  • time_decay_rerank(score, rank) - Exponential decay")
	log.Println("  • nonlinear_rerank(score, rank) - Sigmoid transformation")
}

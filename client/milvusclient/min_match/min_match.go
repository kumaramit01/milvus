package main

import (
	"context"
	"fmt"
	"log"
	"strings"

	"github.com/milvus-io/milvus/client/v2/entity"
	"github.com/milvus-io/milvus/client/v2/index"
	"github.com/milvus-io/milvus/client/v2/milvusclient"
)

const (
	milvusAddr           = "localhost:19530"
	collectionName       = "text_search_collection"
	titleFieldName       = "title"
	textFieldName        = "document_text"
	titleSparseFieldName = "title_sparse_vector"
	textSparseFieldName  = "text_sparse_vector"
)

// TestCase represents a single test case for minShouldMatch functionality
type TestCase struct {
	Name           string
	Query          string
	MinShouldMatch int
	ExpectedCount  int // Expected number of results
	Description    string
}

// TestData represents the test documents
type TestData struct {
	Title string
	Text  string
}

// setupCollection creates and configures the collection for testing
func setupCollection(ctx context.Context, client *milvusclient.Client) error {
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
			WithMaxLength(500).
			WithEnableAnalyzer(true).
			WithEnableMatch(true)).
		WithField(entity.NewField().
			WithName(textFieldName).
			WithDataType(entity.FieldTypeVarChar).
			WithMaxLength(2000).
			WithEnableAnalyzer(true).
			WithEnableMatch(true)).
		WithField(entity.NewField().
			WithName(titleSparseFieldName).
			WithDataType(entity.FieldTypeSparseVector)).
		WithField(entity.NewField().
			WithName(textSparseFieldName).
			WithDataType(entity.FieldTypeSparseVector))

	// Create separate BM25 functions for title and text fields
	titleBM25Function := entity.NewFunction().
		WithName("title_bm25_func").
		WithType(entity.FunctionTypeBM25).
		WithInputFields(titleFieldName).
		WithOutputFields(titleSparseFieldName)

	textBM25Function := entity.NewFunction().
		WithName("text_bm25_func").
		WithType(entity.FunctionTypeBM25).
		WithInputFields(textFieldName).
		WithOutputFields(textSparseFieldName)

	schema.WithFunction(titleBM25Function).WithFunction(textBM25Function)

	titleIndex := index.NewInvertedIndex()
	titleIndexOption := milvusclient.NewCreateIndexOption(collectionName, titleFieldName, titleIndex)

	textIndex := index.NewInvertedIndex()
	textIndexOption := milvusclient.NewCreateIndexOption(collectionName, textFieldName, textIndex)

	titleSparseIndex := index.NewSparseInvertedIndex(entity.BM25, 0.2)
	titleSparseIndexOption := milvusclient.NewCreateIndexOption(collectionName, titleSparseFieldName, titleSparseIndex)

	textSparseIndex := index.NewSparseInvertedIndex(entity.BM25, 0.2)
	textSparseIndexOption := milvusclient.NewCreateIndexOption(collectionName, textSparseFieldName, textSparseIndex)

	err = client.CreateCollection(ctx, milvusclient.NewCreateCollectionOption(collectionName, schema).
		WithIndexOptions(titleIndexOption, textIndexOption, titleSparseIndexOption, textSparseIndexOption))
	if err != nil {
		return fmt.Errorf("failed to create collection: %v", err)
	}

	return nil
}

// insertTestData inserts comprehensive test data into the collection
func insertTestData(ctx context.Context, client *milvusclient.Client) error {
	testData := []TestData{
		{"History of AI", "Artificial intelligence was founded in 1956 by computer scientists."},
		{"Alan Turing Biography", "Alan Turing was the first person to propose AI and machine learning concepts."},
		{"Turing's Background", "Born in Maida Vale, London, Turing was a brilliant mathematician and computer scientist."},
		{"Machine Learning Overview", "Machine learning is a subset of artificial intelligence that focuses on algorithms."},
		{"Deep Learning Applications", "Deep learning neural networks are used in modern AI applications."},
		{"AI Technology Domains", "Computer vision and natural language processing are key AI domains."},
		{"Turing Test Explained", "The Turing test evaluates machine intelligence and human-like responses."},
		{"London Tech Hub", "London is the capital city of England and a major technology hub."},
		{"Mathematical Foundations", "Mathematics and statistics form the foundation of machine learning algorithms."},
		{"Neural Network Structure", "Artificial neural networks mimic the human brain structure for learning."},
	}

	titleColumn := make([]string, len(testData))
	textColumn := make([]string, len(testData))
	for i, d := range testData {
		titleColumn[i] = d.Title
		textColumn[i] = d.Text
	}

	_, err := client.Insert(ctx, milvusclient.NewColumnBasedInsertOption(collectionName).
		WithVarcharColumn(titleFieldName, titleColumn).
		WithVarcharColumn(textFieldName, textColumn))
	if err != nil {
		return fmt.Errorf("failed to insert data: %v", err)
	}

	loadTask, err := client.LoadCollection(ctx, milvusclient.NewLoadCollectionOption(collectionName))
	if err != nil {
		return fmt.Errorf("failed to load collection: %v", err)
	}

	err = loadTask.Await(ctx)
	if err != nil {
		return fmt.Errorf("failed to wait for collection loading: %v", err)
	}

	return nil
}

// runSearchTest executes a single search test case
func runSearchTest(ctx context.Context, client *milvusclient.Client, testCase TestCase) error {
	log.Printf("\n=== Running Test: %s ===", testCase.Name)
	log.Printf("Description: %s", testCase.Description)
	log.Printf("Query: '%s'", testCase.Query)
	log.Printf("MinShouldMatch: %d", testCase.MinShouldMatch)
	log.Printf("Expected Results: %d", testCase.ExpectedCount)

	textVector := entity.Text(testCase.Query)
	// Create a text match filter for both title and text fields
	textMatchFilter := fmt.Sprintf(`text_match(%s, "%s") OR text_match(%s, "%s")`,
		titleFieldName, testCase.Query, textFieldName, testCase.Query)

	// Create a rerank function that boosts title matches
	titleBoostFunction := entity.NewFunction().
		WithName("title_boost").
		WithType(entity.FunctionTypeRerank).
		WithParam("reranker", "boost").
		WithParam("filter", fmt.Sprintf(`text_match(%s, "%s")`, titleFieldName, testCase.Query)).
		WithParam("weight", "2.0")

	searchResult, err := client.Search(ctx, milvusclient.NewSearchOption(
		collectionName,
		10,
		[]entity.Vector{textVector},
	).WithANNSField(textSparseFieldName).
		WithFilter(textMatchFilter).
		WithMinShouldMatch(testCase.MinShouldMatch).
		WithOutputFields("id", titleFieldName, textFieldName).
		WithFunctionReranker(titleBoostFunction))

	if err != nil {
		return fmt.Errorf("search failed: %v", err)
	}

	actualCount := 0
	for _, resultSet := range searchResult {
		actualCount = resultSet.ResultCount
		log.Printf("Actual Results Found: %d", actualCount)

		if actualCount == 0 {
			log.Println("No matching documents found")
		} else {
			log.Println("Matching documents:")
			for i := 0; i < resultSet.ResultCount; i++ {
				var id interface{}
				if resultSet.IDs != nil {
					id, _ = resultSet.IDs.Get(i)
				}

				score := resultSet.Scores[i]

				var title, text string
				titleColumn := resultSet.GetColumn(titleFieldName)
				if titleColumn != nil {
					title, _ = titleColumn.GetAsString(i)
				}

				textColumn := resultSet.GetColumn(textFieldName)
				if textColumn != nil {
					text, _ = textColumn.GetAsString(i)
				}

				log.Printf("  [%d] ID: %v, Score: %.4f", i+1, id, score)
				log.Printf("      Title: %s", title)
				log.Printf("      Text: %s", text)

				// Count matching terms for validation (check both title and text)
				queryTerms := strings.Fields(strings.ToLower(testCase.Query))
				titleTerms := strings.Fields(strings.ToLower(title))
				docTerms := strings.Fields(strings.ToLower(text))

				titleMatchCount := 0
				textMatchCount := 0

				for _, qTerm := range queryTerms {
					// Check title matches
					for _, tTerm := range titleTerms {
						if strings.Contains(tTerm, qTerm) || strings.Contains(qTerm, tTerm) {
							titleMatchCount++
							break
						}
					}
					// Check text matches
					for _, dTerm := range docTerms {
						if strings.Contains(dTerm, qTerm) || strings.Contains(qTerm, dTerm) {
							textMatchCount++
							break
						}
					}
				}
				totalMatches := titleMatchCount + textMatchCount
				log.Printf("      Title matches: %d, Text matches: %d, Total: %d", titleMatchCount, textMatchCount, totalMatches)
			}
		}
	}

	// Validate results
	if testCase.ExpectedCount >= 0 {
		if actualCount == testCase.ExpectedCount {
			log.Printf("✅ PASS: Expected %d results, got %d", testCase.ExpectedCount, actualCount)
		} else {
			log.Printf("❌ FAIL: Expected %d results, got %d", testCase.ExpectedCount, actualCount)
		}
	} else {
		log.Printf("ℹ️  INFO: No specific expectation set (got %d results)", actualCount)
	}

	return nil
}

// runAllTests executes all minShouldMatch test cases
func runAllTests(ctx context.Context, client *milvusclient.Client) error {
	testCases := []TestCase{
		{
			Name:           "Single Term Match",
			Query:          "artificial",
			MinShouldMatch: 1,
			ExpectedCount:  3, // Should find 3 documents containing "artificial"
			Description:    "Test with single term requiring 1 match - title boost should prioritize 'History of AI'",
		},
		{
			Name:           "Two Terms - Require Both",
			Query:          "artificial intelligence",
			MinShouldMatch: 2,
			ExpectedCount:  2, // Should find 2 documents with both terms
			Description:    "Test with two terms requiring both to match",
		},
		{
			Name:           "Two Terms - Require One",
			Query:          "artificial intelligence",
			MinShouldMatch: 1,
			ExpectedCount:  4, // Should find 4 documents with at least one term
			Description:    "Test with two terms requiring only one to match",
		},
		{
			Name:           "Multiple Terms - High Requirement",
			Query:          "artificial intelligence machine learning",
			MinShouldMatch: 3,
			ExpectedCount:  1, // Should find 1 document with 3+ matching terms
			Description:    "Test with four terms requiring three matches",
		},
		{
			Name:           "Multiple Terms - Low Requirement",
			Query:          "artificial intelligence machine learning",
			MinShouldMatch: 1,
			ExpectedCount:  7, // Should find 7 documents with at least one term
			Description:    "Test with four terms requiring only one match",
		},
		{
			Name:           "Title Boost Test - Biography",
			Query:          "Alan Turing",
			MinShouldMatch: 2,
			ExpectedCount:  1, // Should find 1 document with both terms (Alan Turing Biography)
			Description:    "Test title boost - 'Alan Turing Biography' should rank higher due to title match",
		},
		{
			Name:           "Title Boost Test - Learning",
			Query:          "learning",
			MinShouldMatch: 1,
			ExpectedCount:  5, // Should find 5 documents containing "learning"
			Description:    "Test title boost - documents with 'learning' in title should rank higher",
		},
		{
			Name:           "Technical Terms",
			Query:          "neural networks deep learning algorithms",
			MinShouldMatch: 2,
			ExpectedCount:  4, // Should find 4 documents with at least 2 technical terms
			Description:    "Test with technical terms requiring 2 matches",
		},
		{
			Name:           "Mixed Common and Rare Terms",
			Query:          "computer vision processing applications",
			MinShouldMatch: 2,
			ExpectedCount:  1, // Should find 1 document with computer vision and processing
			Description:    "Test mixing common and specific terms",
		},
		{
			Name:           "Impossible Match",
			Query:          "nonexistent impossible terms here",
			MinShouldMatch: 1,
			ExpectedCount:  0, // Should find 0 documents (no matching terms)
			Description:    "Test with terms that don't exist in any document",
		},
		{
			Name:           "Very High Requirement",
			Query:          "artificial intelligence machine learning deep neural networks",
			MinShouldMatch: 6,
			ExpectedCount:  0, // Should find 0 documents (impossible requirement)
			Description:    "Test requiring more matches than query terms (should return 0)",
		},
	}

	log.Println("\n🚀 Starting MinShouldMatch Test Suite")
	log.Println(strings.Repeat("=", 60))

	passCount := 0
	totalTests := len(testCases)

	for i, testCase := range testCases {
		log.Printf("\nTest %d/%d", i+1, totalTests)
		err := runSearchTest(ctx, client, testCase)
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
		log.Fatalf("failed to connect to Milvus: %v", err)
	}
	defer milvusClient.Close(ctx)
	log.Println("✅ Connected to Milvus successfully.")

	log.Println("\n🏗️  Setting up test collection...")
	err = setupCollection(ctx, milvusClient)
	if err != nil {
		log.Fatalf("failed to setup collection: %v", err)
	}
	log.Printf("✅ Collection '%s' created successfully.", collectionName)

	log.Println("\n📝 Inserting comprehensive test data...")
	err = insertTestData(ctx, milvusClient)
	if err != nil {
		log.Fatalf("failed to insert test data: %v", err)
	}
	log.Println("✅ Test data inserted and collection loaded successfully.")

	log.Println("\n🧪 Running MinShouldMatch tests...")
	err = runAllTests(ctx, milvusClient)
	if err != nil {
		log.Fatalf("failed to run tests: %v", err)
	}

	log.Println("\n🧹 Cleaning up...")
	err = milvusClient.DropCollection(ctx, milvusclient.NewDropCollectionOption(collectionName))
	if err != nil {
		log.Printf("failed to drop collection: %v", err)
	} else {
		log.Printf("✅ Collection '%s' dropped successfully.", collectionName)
	}

	log.Println("\n🎉 MinShouldMatch test suite completed!")
}

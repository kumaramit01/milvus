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

// nolint
package milvusclient_test

import (
	"context"
	"encoding/base64"
	"fmt"
	"log"
	"os"

	"github.com/milvus-io/milvus/client/v2/column"
	"github.com/milvus-io/milvus/client/v2/entity"
	"github.com/milvus-io/milvus/client/v2/index"
	"github.com/milvus-io/milvus/client/v2/milvusclient"
)

func ExampleClient_Search_basic() {
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	milvusAddr := "127.0.0.1:19530"
	token := "root:Milvus"

	cli, err := milvusclient.New(ctx, &milvusclient.ClientConfig{
		Address: milvusAddr,
		APIKey:  token,
	})
	if err != nil {
		log.Fatal("failed to connect to milvus server: ", err.Error())
	}

	defer cli.Close(ctx)

	queryVector := []float32{0.3580376395471989, -0.6023495712049978, 0.18414012509913835, -0.26286205330961354, 0.9029438446296592}

	resultSets, err := cli.Search(ctx, milvusclient.NewSearchOption(
		"quick_setup", // collectionName
		3,             // limit
		[]entity.Vector{entity.FloatVector(queryVector)},
	))
	if err != nil {
		log.Fatal("failed to perform basic ANN search collection: ", err.Error())
	}

	for _, resultSet := range resultSets {
		log.Println("IDs: ", resultSet.IDs)
		log.Println("Scores: ", resultSet.Scores)
	}
}

func ExampleClient_Search_multivectors() {
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	milvusAddr := "127.0.0.1:19530"
	token := "root:Milvus"

	cli, err := milvusclient.New(ctx, &milvusclient.ClientConfig{
		Address: milvusAddr,
		APIKey:  token,
	})
	if err != nil {
		log.Fatal("failed to connect to milvus server: ", err.Error())
	}

	defer cli.Close(ctx)

	queryVectors := []entity.Vector{
		entity.FloatVector([]float32{0.3580376395471989, -0.6023495712049978, 0.18414012509913835, -0.26286205330961354, 0.9029438446296592}),
		entity.FloatVector([]float32{0.19886812562848388, 0.06023560599112088, 0.6976963061752597, 0.2614474506242501, 0.838729485096104}),
	}

	resultSets, err := cli.Search(ctx, milvusclient.NewSearchOption(
		"quick_setup", // collectionName
		3,             // limit
		queryVectors,
	))
	if err != nil {
		log.Fatal("failed to perform basic ANN search collection: ", err.Error())
	}

	for _, resultSet := range resultSets {
		log.Println("IDs: ", resultSet.IDs)
		log.Println("Scores: ", resultSet.Scores)
	}
}

func ExampleClient_Search_partition() {
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	milvusAddr := "127.0.0.1:19530"
	token := "root:Milvus"

	cli, err := milvusclient.New(ctx, &milvusclient.ClientConfig{
		Address: milvusAddr,
		APIKey:  token,
	})
	if err != nil {
		log.Fatal("failed to connect to milvus server: ", err.Error())
	}

	defer cli.Close(ctx)

	queryVector := []float32{0.3580376395471989, -0.6023495712049978, 0.18414012509913835, -0.26286205330961354, 0.9029438446296592}

	resultSets, err := cli.Search(ctx, milvusclient.NewSearchOption(
		"quick_setup", // collectionName
		3,             // limit
		[]entity.Vector{entity.FloatVector(queryVector)},
	).WithPartitions("partitionA"))
	if err != nil {
		log.Fatal("failed to perform basic ANN search collection: ", err.Error())
	}

	for _, resultSet := range resultSets {
		log.Println("IDs: ", resultSet.IDs)
		log.Println("Scores: ", resultSet.Scores)
	}
}

func ExampleClient_Search_outputFields() {
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	milvusAddr := "127.0.0.1:19530"
	token := "root:Milvus"

	cli, err := milvusclient.New(ctx, &milvusclient.ClientConfig{
		Address: milvusAddr,
		APIKey:  token,
	})
	if err != nil {
		log.Fatal("failed to connect to milvus server: ", err.Error())
	}

	defer cli.Close(ctx)

	queryVector := []float32{0.3580376395471989, -0.6023495712049978, 0.18414012509913835, -0.26286205330961354, 0.9029438446296592}

	resultSets, err := cli.Search(ctx, milvusclient.NewSearchOption(
		"quick_setup", // collectionName
		3,             // limit
		[]entity.Vector{entity.FloatVector(queryVector)},
	).WithOutputFields("color"))
	if err != nil {
		log.Fatal("failed to perform basic ANN search collection: ", err.Error())
	}

	for _, resultSet := range resultSets {
		log.Println("IDs: ", resultSet.IDs)
		log.Println("Scores: ", resultSet.Scores)
		log.Println("Colors: ", resultSet.GetColumn("color"))
	}
}

func ExampleClient_Search_offsetLimit() {
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	milvusAddr := "127.0.0.1:19530"
	token := "root:Milvus"

	cli, err := milvusclient.New(ctx, &milvusclient.ClientConfig{
		Address: milvusAddr,
		APIKey:  token,
	})
	if err != nil {
		log.Fatal("failed to connect to milvus server: ", err.Error())
	}

	defer cli.Close(ctx)

	queryVector := []float32{0.3580376395471989, -0.6023495712049978, 0.18414012509913835, -0.26286205330961354, 0.9029438446296592}

	resultSets, err := cli.Search(ctx, milvusclient.NewSearchOption(
		"quick_setup", // collectionName
		3,             // limit
		[]entity.Vector{entity.FloatVector(queryVector)},
	).WithOffset(10))
	if err != nil {
		log.Fatal("failed to perform basic ANN search collection: ", err.Error())
	}

	for _, resultSet := range resultSets {
		log.Println("IDs: ", resultSet.IDs)
		log.Println("Scores: ", resultSet.Scores)
	}
}

func ExampleClient_Search_jsonExpr() {
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	cli, err := milvusclient.New(ctx, &milvusclient.ClientConfig{
		Address: milvusAddr,
	})
	if err != nil {
		log.Fatal("failed to connect to milvus server: ", err.Error())
	}

	defer cli.Close(ctx)

	queryVector := []float32{0.3, -0.6, -0.1}

	annParam := index.NewCustomAnnParam()
	annParam.WithExtraParam("nprobe", 10)
	resultSets, err := cli.Search(ctx, milvusclient.NewSearchOption(
		"my_json_collection", // collectionName
		5,                    // limit
		[]entity.Vector{entity.FloatVector(queryVector)},
	).WithOutputFields("metadata").WithAnnParam(annParam))
	if err != nil {
		log.Fatal("failed to perform basic ANN search collection: ", err.Error())
	}

	for _, resultSet := range resultSets {
		log.Println("IDs: ", resultSet.IDs)
		log.Println("Scores: ", resultSet.Scores)
	}
}

func ExampleClient_Search_binaryVector() {
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	milvusAddr := "127.0.0.1:19530"
	token := "root:Milvus"

	cli, err := milvusclient.New(ctx, &milvusclient.ClientConfig{
		Address: milvusAddr,
		APIKey:  token,
	})
	if err != nil {
		log.Fatal("failed to connect to milvus server: ", err.Error())
	}

	defer cli.Close(ctx)

	queryVector := []byte{0b10011011, 0b01010100, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}

	annSearchParams := index.NewCustomAnnParam()
	annSearchParams.WithExtraParam("nprobe", 10)
	resultSets, err := cli.Search(ctx, milvusclient.NewSearchOption(
		"my_binary_collection", // collectionName
		5,                      // limit
		[]entity.Vector{entity.BinaryVector(queryVector)},
	).WithOutputFields("pk").WithAnnParam(annSearchParams))
	if err != nil {
		log.Fatal("failed to perform basic ANN search collection: ", err.Error())
	}

	for _, resultSet := range resultSets {
		log.Println("IDs: ", resultSet.IDs)
		log.Println("Scores: ", resultSet.Scores)
		log.Println("Pks: ", resultSet.GetColumn("pk"))
	}
}

func ExampleClient_Get() {
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	milvusAddr := "127.0.0.1:19530"

	cli, err := milvusclient.New(ctx, &milvusclient.ClientConfig{
		Address: milvusAddr,
	})
	if err != nil {
		log.Fatal("failed to connect to milvus server: ", err.Error())
	}

	defer cli.Close(ctx)

	rs, err := cli.Get(ctx, milvusclient.NewQueryOption("quick_setup").
		WithIDs(column.NewColumnInt64("id", []int64{1, 2, 3})))
	if err != nil {
		// handle error
	}

	fmt.Println(rs.GetColumn("id"))
}

func ExampleClient_Query() {
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	milvusAddr := "127.0.0.1:19530"

	cli, err := milvusclient.New(ctx, &milvusclient.ClientConfig{
		Address: milvusAddr,
	})
	if err != nil {
		log.Fatal("failed to connect to milvus server: ", err.Error())
	}

	defer cli.Close(ctx)

	rs, err := cli.Query(ctx, milvusclient.NewQueryOption("quick_setup").
		WithFilter("emb_type == 3").
		WithOutputFields("id", "emb_type"))
	if err != nil {
		// handle error
	}

	fmt.Println(rs.GetColumn("id"))
}

func ExampleClient_Query_jsonExpr_notnull() {
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	cli, err := milvusclient.New(ctx, &milvusclient.ClientConfig{
		Address: milvusAddr,
	})
	if err != nil {
		log.Fatal("failed to connect to milvus server: ", err.Error())
	}

	defer cli.Close(ctx)

	rs, err := cli.Query(ctx, milvusclient.NewQueryOption("my_json_collection").
		WithFilter("metadata is not null").
		WithOutputFields("metadata", "pk"))
	if err != nil {
		// handle error
	}

	fmt.Println(rs.GetColumn("pk"))
	fmt.Println(rs.GetColumn("metadata"))
}

func ExampleClient_Query_jsonExpr_leafChild() {
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	cli, err := milvusclient.New(ctx, &milvusclient.ClientConfig{
		Address: milvusAddr,
	})
	if err != nil {
		log.Fatal("failed to connect to milvus server: ", err.Error())
	}

	defer cli.Close(ctx)

	rs, err := cli.Query(ctx, milvusclient.NewQueryOption("my_json_collection").
		WithFilter(`metadata["product_info"]["category"] == "electronics"`).
		WithOutputFields("metadata", "pk"))
	if err != nil {
		// handle error
	}

	fmt.Println(rs.GetColumn("pk"))
	fmt.Println(rs.GetColumn("metadata"))
}

func ExampleClient_HybridSearch() {
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	milvusAddr := "127.0.0.1:19530"
	token := "root:Milvus"

	cli, err := milvusclient.New(ctx, &milvusclient.ClientConfig{
		Address: milvusAddr,
		APIKey:  token,
	})
	if err != nil {
		log.Fatal("failed to connect to milvus server: ", err.Error())
	}

	defer cli.Close(ctx)

	queryVector := []float32{0.3580376395471989, -0.6023495712049978, 0.18414012509913835, -0.26286205330961354, 0.9029438446296592}
	sparseVector, _ := entity.NewSliceSparseEmbedding([]uint32{1, 21, 100}, []float32{0.1, 0.2, 0.3})

	resultSets, err := cli.HybridSearch(ctx, milvusclient.NewHybridSearchOption(
		"quick_setup",
		3,
		milvusclient.NewAnnRequest("dense_vector", 10, entity.FloatVector(queryVector)),
		milvusclient.NewAnnRequest("sparse_vector", 10, sparseVector),
	).WithReranker(milvusclient.NewRRFReranker()))
	if err != nil {
		log.Fatal("failed to perform basic ANN search collection: ", err.Error())
	}

	for _, resultSet := range resultSets {
		log.Println("IDs: ", resultSet.IDs)
		log.Println("Scores: ", resultSet.Scores)
	}
}

// ExampleClient_Search_wasmRerank demonstrates how to use WASM-based reranking
func ExampleClient_Search_wasmRerank() {
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	milvusAddr := "127.0.0.1:19530"
	token := "root:Milvus"

	cli, err := milvusclient.New(ctx, &milvusclient.ClientConfig{
		Address: milvusAddr,
		APIKey:  token,
	})
	if err != nil {
		log.Fatal("failed to connect to milvus server: ", err.Error())
	}

	defer cli.Close(ctx)

	// WASM-based reranking with Rust module
	//
	// Prerequisites - Build the Rust WASM module:
	//   cd tests/reranker_wasm/rust_reranker
	//   rustup target add wasm32-unknown-unknown
	//   cargo build --target wasm32-unknown-unknown --release
	//
	// Available entry points in rust_reranker.wasm:
	//   - rerank(score, rank) - Position-based decay
	//   - rerank_with_popularity(score, rank, popularity) - Popularity boost
	//   - rerank_with_timestamp(score, rank, timestamp) - Time decay
	//   - rerank_complex(score, rank, popularity, quality) - Multi-factor
	//   - time_decay_rerank(score, rank) - Exponential time decay
	//   - log_boost_rerank(score, rank) - Logarithmic position boost
	//   - simple_boost(score, rank) - Simple 1.5x multiplier
	//   - nonlinear_rerank(score, rank) - Sigmoid-like transformation

	wasmPath := "../../tests/reranker_wasm/rust_reranker/target/wasm32-unknown-unknown/release/rust_reranker.wasm"
	wasmBytes, err := os.ReadFile(wasmPath)
	if err != nil {
		log.Fatal("Failed to read WASM module:", err)
	}
	wasmBase64 := base64.StdEncoding.EncodeToString(wasmBytes)

	// Example 1: Simple position-based reranking (no fields required)
	schema1 := entity.NewSchema().
		WithName("wasm_simple_rerank").
		WithField(entity.NewField().WithName("id").WithDataType(entity.FieldTypeInt64).WithIsPrimaryKey(true).WithIsAutoID(true)).
		WithField(entity.NewField().WithName("embedding").WithDataType(entity.FieldTypeFloatVector).WithDim(128)).
		WithField(entity.NewField().WithName("text").WithDataType(entity.FieldTypeVarChar).WithMaxLength(1000))

	// Simple position decay: score * (1.0 / (1.0 + 0.05 * rank))
	rerankFunc1 := entity.NewFunction().
		WithName("simple_rerank").
		WithType(entity.FunctionTypeRerank).
		WithInputFields(). // No input fields for simple rerank
		WithParam("reranker", "wasm").
		WithParam("wasm_code", wasmBase64).
		WithParam("entry_point", "rerank")

	schema1.WithFunction(rerankFunc1)

	// Example 2: Multi-factor reranking with popularity and quality
	schema2 := entity.NewSchema().
		WithName("wasm_complex_rerank").
		WithField(entity.NewField().WithName("id").WithDataType(entity.FieldTypeInt64).WithIsPrimaryKey(true).WithIsAutoID(true)).
		WithField(entity.NewField().WithName("embedding").WithDataType(entity.FieldTypeFloatVector).WithDim(128)).
		WithField(entity.NewField().WithName("text").WithDataType(entity.FieldTypeVarChar).WithMaxLength(1000)).
		WithField(entity.NewField().WithName("popularity").WithDataType(entity.FieldTypeFloat)).
		WithField(entity.NewField().WithName("quality_score").WithDataType(entity.FieldTypeFloat))

	// Complex reranking formula:
	//   popularity_factor = 1.0 + ln(popularity + 1) / 20.0
	//   quality_factor = 0.7 + 0.6 * quality_score
	//   position_factor = 1.0 / (1.0 + 0.04 * rank)
	//   final_score = score * sqrt(popularity_factor * quality_factor) * position_factor
	rerankFunc2 := entity.NewFunction().
		WithName("complex_rerank").
		WithType(entity.FunctionTypeRerank).
		WithInputFields("popularity", "quality_score").
		WithParam("reranker", "wasm").
		WithParam("wasm_code", wasmBase64).
		WithParam("entry_point", "rerank_complex")

	schema2.WithFunction(rerankFunc2)

	// Create collections (commented to avoid actual creation in examples)
	// err = cli.CreateCollection(ctx, milvusclient.NewCreateCollectionOption("wasm_simple_rerank", schema1))
	// err = cli.CreateCollection(ctx, milvusclient.NewCreateCollectionOption("wasm_complex_rerank", schema2))

	// Search with automatic WASM reranking
	// results, _ := cli.Search(ctx, milvusclient.NewSearchOption(
	// 	"wasm_complex_rerank",
	// 	10,
	// 	[]entity.Vector{entity.FloatVector(queryVector)},
	// ).WithOutputFields("text", "popularity", "quality_score"))

	fmt.Println("WASM Reranker Examples")
	fmt.Println("For complete working examples, see tests/reranker_wasm/python_wasm_example.py")
}

// ExampleClient_Search_exprRerank demonstrates how to use expression-based reranking
func ExampleClient_Search_exprRerank() {
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	milvusAddr := "127.0.0.1:19530"
	token := "root:Milvus"

	cli, err := milvusclient.New(ctx, &milvusclient.ClientConfig{
		Address: milvusAddr,
		APIKey:  token,
	})
	if err != nil {
		log.Fatal("failed to connect to milvus server: ", err.Error())
	}

	defer cli.Close(ctx)

	// Expression-based reranking allows custom scoring with simple expressions
	//
	// Available variables:
	//   - score: original search score
	//   - rank: position in result list (0-indexed)
	//   - fields["field_name"]: access to field values
	//
	// Supported functions:
	//   - Math: exp, log, sqrt, pow, abs, min, max
	//   - Conditionals: ? : (ternary operator)
	//   - Operators: +, -, *, /, ==, !=, <, >, <=, >=, &&, ||

	// Example 1: Quality-based boosting
	schema1 := entity.NewSchema().
		WithName("expr_quality_boost").
		WithField(entity.NewField().WithName("id").WithDataType(entity.FieldTypeInt64).WithIsPrimaryKey(true).WithIsAutoID(true)).
		WithField(entity.NewField().WithName("embedding").WithDataType(entity.FieldTypeFloatVector).WithDim(128)).
		WithField(entity.NewField().WithName("text").WithDataType(entity.FieldTypeVarChar).WithMaxLength(1000)).
		WithField(entity.NewField().WithName("quality_score").WithDataType(entity.FieldTypeFloat))

	rerankFunc1 := entity.NewFunction().
		WithName("quality_rerank").
		WithType(entity.FunctionTypeRerank).
		WithInputFields("quality_score").
		WithParam("reranker", "expr").
		WithParam("expr_code", `score * (fields["quality_score"] / 100.0)`)

	schema1.WithFunction(rerankFunc1)

	// Example 2: Popularity boost with logarithmic scaling
	schema2 := entity.NewSchema().
		WithName("expr_popularity_boost").
		WithField(entity.NewField().WithName("id").WithDataType(entity.FieldTypeInt64).WithIsPrimaryKey(true).WithIsAutoID(true)).
		WithField(entity.NewField().WithName("embedding").WithDataType(entity.FieldTypeFloatVector).WithDim(128)).
		WithField(entity.NewField().WithName("text").WithDataType(entity.FieldTypeVarChar).WithMaxLength(1000)).
		WithField(entity.NewField().WithName("popularity").WithDataType(entity.FieldTypeInt64))

	rerankFunc2 := entity.NewFunction().
		WithName("popularity_rerank").
		WithType(entity.FunctionTypeRerank).
		WithInputFields("popularity").
		WithParam("reranker", "expr").
		WithParam("expr_code", `score * (1.0 + log(fields["popularity"] + 1) / 10.0)`)

	schema2.WithFunction(rerankFunc2)

	// Example 3: Multi-factor reranking with quality, popularity, and recency
	schema3 := entity.NewSchema().
		WithName("expr_multi_factor").
		WithField(entity.NewField().WithName("id").WithDataType(entity.FieldTypeInt64).WithIsPrimaryKey(true).WithIsAutoID(true)).
		WithField(entity.NewField().WithName("embedding").WithDataType(entity.FieldTypeFloatVector).WithDim(128)).
		WithField(entity.NewField().WithName("text").WithDataType(entity.FieldTypeVarChar).WithMaxLength(1000)).
		WithField(entity.NewField().WithName("quality_score").WithDataType(entity.FieldTypeFloat)).
		WithField(entity.NewField().WithName("popularity").WithDataType(entity.FieldTypeInt64)).
		WithField(entity.NewField().WithName("created_at").WithDataType(entity.FieldTypeInt64))

	// Complex multi-line expression with variables
	exprCode3 := `
		let quality = fields["quality_score"] / 100.0;
		let pop_boost = 1.0 + log(fields["popularity"] + 1) / 20.0;
		let age_days = (1704067200000 - fields["created_at"]) / 86400000;
		let recency = exp(-0.005 * age_days);
		score * quality * pop_boost * recency
	`

	rerankFunc3 := entity.NewFunction().
		WithName("multi_factor_rerank").
		WithType(entity.FunctionTypeRerank).
		WithInputFields("quality_score", "popularity", "created_at").
		WithParam("reranker", "expr").
		WithParam("expr_code", exprCode3)

	schema3.WithFunction(rerankFunc3)

	// Example 4: Conditional boosting based on category
	schema4 := entity.NewSchema().
		WithName("expr_conditional_boost").
		WithField(entity.NewField().WithName("id").WithDataType(entity.FieldTypeInt64).WithIsPrimaryKey(true).WithIsAutoID(true)).
		WithField(entity.NewField().WithName("embedding").WithDataType(entity.FieldTypeFloatVector).WithDim(128)).
		WithField(entity.NewField().WithName("text").WithDataType(entity.FieldTypeVarChar).WithMaxLength(1000)).
		WithField(entity.NewField().WithName("category").WithDataType(entity.FieldTypeVarChar).WithMaxLength(50))

	rerankFunc4 := entity.NewFunction().
		WithName("conditional_rerank").
		WithType(entity.FunctionTypeRerank).
		WithInputFields("category").
		WithParam("reranker", "expr").
		WithParam("expr_code", `let boost = fields["category"] == "featured" ? 2.0 : fields["category"] == "premium" ? 1.5 : 1.0; score * boost`)

	schema4.WithFunction(rerankFunc4)

	// Create collections (commented to avoid actual creation in examples)
	// err = cli.CreateCollection(ctx, milvusclient.NewCreateCollectionOption("expr_quality_boost", schema1))
	// err = cli.CreateCollection(ctx, milvusclient.NewCreateCollectionOption("expr_popularity_boost", schema2))
	// err = cli.CreateCollection(ctx, milvusclient.NewCreateCollectionOption("expr_multi_factor", schema3))
	// err = cli.CreateCollection(ctx, milvusclient.NewCreateCollectionOption("expr_conditional_boost", schema4))

	// Search with automatic expression reranking
	// results, _ := cli.Search(ctx, milvusclient.NewSearchOption(
	// 	"expr_multi_factor",
	// 	10,
	// 	[]entity.Vector{entity.FloatVector(queryVector)},
	// ).WithOutputFields("text", "quality_score", "popularity", "created_at"))

	fmt.Println("Expression Reranker Examples")
	fmt.Println("For complete working examples, see tests/reranker_wasm/python_expr_example.py")
	// Output: Expression Reranker Examples
	// For complete working examples, see tests/reranker_wasm/python_expr_example.py

}

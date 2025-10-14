package rerank

import (
	"context"
	"fmt"

	"google.golang.org/protobuf/proto"

	"github.com/milvus-io/milvus-proto/go-api/v2/schemapb"
	"github.com/milvus-io/milvus/pkg/v2/proto/internalpb"
)

// ApplySegmentLevelRerank applies reranking at the QueryNode level
// This is called after segment results are reduced but before sending to proxy
func ApplySegmentLevelRerank(
	ctx context.Context,
	searchResults *internalpb.SearchResults,
	collSchema *schemapb.CollectionSchema,
	funcSchema *schemapb.FunctionSchema,
) (*internalpb.SearchResults, error) {
	if funcSchema == nil {
		return searchResults, nil
	}

	// Only process expression-based rerankers
	if !IsQueryNodeRanker(funcSchema) {
		return searchResults, nil
	}

	reranker, err := createFunction(collSchema, funcSchema)
	if err != nil {
		return nil, fmt.Errorf("failed to create segment-level reranker: %w", err)
	}

	// Decode search result data
	if searchResults.SlicedBlob == nil {
		return searchResults, nil
	}

	var searchResultData schemapb.SearchResultData
	err = proto.Unmarshal(searchResults.SlicedBlob, &searchResultData)
	if err != nil {
		return nil, fmt.Errorf("failed to decode search results for reranking: %w", err)
	}

	if searchResultData.Ids == nil {
		return searchResults, nil
	}

	// Create search params from the search results metadata
	searchParams := &SearchParams{
		nq:              searchResults.NumQueries,
		limit:           searchResults.TopK,
		offset:          0,
		roundDecimal:    -1, // Default: no rounding
		groupByFieldId:  -1, // No grouping at segment level
		groupSize:       0,
		strictGroupSize: false,
		groupScore:      maxScorer,
		searchMetrics:   []string{searchResults.MetricType},
	}

	// Convert to rerank inputs format
	inputs, err := newRerankInputs([]*schemapb.SearchResultData{&searchResultData}, reranker.GetInputFieldIDs(), false)
	if err != nil {
		return nil, fmt.Errorf("failed to create rerank inputs: %w", err)
	}

	// Apply reranking
	outputs, err := reranker.Process(ctx, searchParams, inputs)
	if err != nil {
		return nil, fmt.Errorf("failed to apply segment-level reranking: %w", err)
	}

	// Encode the reranked results back
	slicedBlob, err := proto.Marshal(outputs.searchResultData)
	if err != nil {
		return nil, fmt.Errorf("failed to encode reranked results: %w", err)
	}

	rerankedResults := &internalpb.SearchResults{
		Status:         searchResults.Status,
		NumQueries:     searchResults.NumQueries,
		TopK:           searchResults.TopK,
		MetricType:     searchResults.MetricType,
		SlicedBlob:     slicedBlob,
		SlicedNumCount: 1,
		SlicedOffset:   1,
	}

	// Preserve metadata from original search results
	rerankedResults.ChannelsMvcc = searchResults.ChannelsMvcc
	rerankedResults.CostAggregation = searchResults.CostAggregation
	rerankedResults.IsTopkReduce = searchResults.IsTopkReduce
	rerankedResults.IsRecallEvaluation = searchResults.IsRecallEvaluation
	rerankedResults.ScannedRemoteBytes = searchResults.ScannedRemoteBytes
	rerankedResults.ScannedTotalBytes = searchResults.ScannedTotalBytes

	return rerankedResults, nil
}

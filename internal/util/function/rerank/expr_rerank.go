/*
 * # Licensed to the LF AI & Data foundation under one
 * # or more contributor license agreements. See the NOTICE file
 * # distributed with this work for additional information
 * # regarding copyright ownership. The ASF licenses this file
 * # to you under the Apache License, Version 2.0 (the
 * # "License"); you may not use this file except in compliance
 * # with the License. You may obtain a copy of the License at
 * #
 * #     http://www.apache.org/licenses/LICENSE-2.0
 * #
 * # Unless required by applicable law or agreed to in writing, software
 * # distributed under the License is distributed on an "AS IS" BASIS,
 * # WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * # See the License for the specific language governing permissions and
 * # limitations under the License.
 */

package rerank

import (
	"context"
	"fmt"
	"strings"

	"github.com/expr-lang/expr"
	"github.com/expr-lang/expr/vm"

	"github.com/milvus-io/milvus-proto/go-api/v2/schemapb"
)

// ExprRerank allows users to define reranking logic using expressions
type ExprRerank[T PKType] struct {
	RerankBase

	program       *vm.Program
	exprString    string
	needNormalize bool
}

const (
	ExprRerankName   = "expr"
	ExprCodeKey      = "expr_code"
	ExprNormalizeKey = "normalize"
)

func newExprRerank(collSchema *schemapb.CollectionSchema, funcSchema *schemapb.FunctionSchema) (Reranker, error) {
	base, err := newRerankBase(collSchema, funcSchema, ExprRerankName, false)
	if err != nil {
		return nil, err
	}

	var exprCode string
	needNormalize := false

	for _, param := range funcSchema.Params {
		switch strings.ToLower(param.Key) {
		case ExprCodeKey:
			exprCode = param.Value
		case ExprNormalizeKey:
			if norm, err := parseBool(param.Value); err != nil {
				return nil, fmt.Errorf("invalid normalize value: %w", err)
			} else {
				needNormalize = norm
			}
		}
	}

	if exprCode == "" {
		return nil, fmt.Errorf("expr rerank requires %s parameter", ExprCodeKey)
	}

	// Compile the expression
	program, err := expr.Compile(exprCode, expr.Env(map[string]interface{}{
		"score":  float32(0),
		"rank":   int(0),
		"fields": map[string]interface{}{},
	}))
	if err != nil {
		return nil, fmt.Errorf("failed to compile expression: %w", err)
	}

	if base.pkType == schemapb.DataType_Int64 {
		return &ExprRerank[int64]{
			RerankBase:    *base,
			program:       program,
			exprString:    exprCode,
			needNormalize: needNormalize,
		}, nil
	}
	return &ExprRerank[string]{
		RerankBase:    *base,
		program:       program,
		exprString:    exprCode,
		needNormalize: needNormalize,
	}, nil
}

func (e *ExprRerank[T]) processOneSearchData(ctx context.Context, searchParams *SearchParams, cols []*columns, idGroup map[any]any) (*IDScores[T], error) {
	newScores := map[T]float32{}
	idLocations := make(map[T]IDLoc)

	for colIdx, col := range cols {
		if col.size == 0 {
			continue
		}

		ids := col.ids.([]T)
		scores := col.scores

		for idx, id := range ids {
			if _, exists := newScores[id]; exists {
				continue // Already processed (use first occurrence)
			}

			// Prepare environment for expression evaluation
			env := map[string]interface{}{
				"score": scores[idx],
				"rank":  idx,
			}

			// Add field values to environment
			fields := make(map[string]interface{})
			for fieldIdx, fieldName := range e.inputFieldNames {
				if fieldIdx < len(col.data) {
					switch data := col.data[fieldIdx].(type) {
					case []int32:
						if idx < len(data) {
							fields[fieldName] = data[idx]
						}
					case []int64:
						if idx < len(data) {
							fields[fieldName] = data[idx]
						}
					case []float32:
						if idx < len(data) {
							fields[fieldName] = data[idx]
						}
					case []float64:
						if idx < len(data) {
							fields[fieldName] = data[idx]
						}
					case []string:
						if idx < len(data) {
							fields[fieldName] = data[idx]
						}
					case []bool:
						if idx < len(data) {
							fields[fieldName] = data[idx]
						}
					}
				}
			}
			env["fields"] = fields

			// Execute the expression
			output, err := expr.Run(e.program, env)
			if err != nil {
				return nil, fmt.Errorf("failed to execute expression for id %v: %w", id, err)
			}

			// Convert output to float32
			var newScore float32
			switch v := output.(type) {
			case float32:
				newScore = v
			case float64:
				newScore = float32(v)
			case int:
				newScore = float32(v)
			case int64:
				newScore = float32(v)
			default:
				return nil, fmt.Errorf("expression returned unsupported type %T for id %v", output, id)
			}

			newScores[id] = newScore
			idLocations[id] = IDLoc{batchIdx: colIdx, offset: idx}
		}
	}

	if searchParams.isGrouping() {
		return newGroupingIDScores(newScores, idLocations, searchParams, idGroup)
	}
	return newIDScores(newScores, idLocations, searchParams, true), nil
}

func (e *ExprRerank[T]) Process(ctx context.Context, searchParams *SearchParams, inputs *rerankInputs) (*rerankOutputs, error) {
	outputs := newRerankOutputs(inputs, searchParams)

	for _, cols := range inputs.data {
		idScore, err := e.processOneSearchData(ctx, searchParams, cols, inputs.idGroupValue)
		if err != nil {
			return nil, err
		}
		appendResult(inputs, outputs, idScore)
	}

	return outputs, nil
}

func parseBool(s string) (bool, error) {
	switch strings.ToLower(s) {
	case "true", "1", "yes":
		return true, nil
	case "false", "0", "no":
		return false, nil
	default:
		return false, fmt.Errorf("invalid boolean value: %s", s)
	}
}

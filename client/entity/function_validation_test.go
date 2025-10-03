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

package entity

import (
	"testing"

	"github.com/stretchr/testify/assert"
)

func TestFunctionValidateParams(t *testing.T) {
	t.Run("valid custom boost mode with expression", func(t *testing.T) {
		function := NewFunction().
			WithName("test_boost").
			WithType(FunctionTypeRerank).
			WithParam("reranker", "boost").
			WithParam("boost_mode", "custom").
			WithParam("boost_expr", "original_score * 0.7 + boost_score * 0.3")

		err := function.ValidateParams()
		assert.NoError(t, err)
	})

	t.Run("valid custom boost mode with case insensitive", func(t *testing.T) {
		function := NewFunction().
			WithName("test_boost").
			WithType(FunctionTypeRerank).
			WithParam("reranker", "boost").
			WithParam("boost_mode", "CUSTOM").
			WithParam("boost_expr", "min(original_score + boost_score, 1.0)")

		err := function.ValidateParams()
		assert.NoError(t, err)
	})

	t.Run("valid multiply boost mode without expression", func(t *testing.T) {
		function := NewFunction().
			WithName("test_boost").
			WithType(FunctionTypeRerank).
			WithParam("reranker", "boost").
			WithParam("boost_mode", "multiply")

		err := function.ValidateParams()
		assert.NoError(t, err)
	})

	t.Run("valid sum boost mode without expression", func(t *testing.T) {
		function := NewFunction().
			WithName("test_boost").
			WithType(FunctionTypeRerank).
			WithParam("reranker", "boost").
			WithParam("boost_mode", "sum")

		err := function.ValidateParams()
		assert.NoError(t, err)
	})

	t.Run("invalid custom boost mode missing expression", func(t *testing.T) {
		function := NewFunction().
			WithName("test_boost").
			WithType(FunctionTypeRerank).
			WithParam("reranker", "boost").
			WithParam("boost_mode", "custom")
			// Missing boost_expr parameter

		err := function.ValidateParams()
		assert.Error(t, err)
		assert.Contains(t, err.Error(), "custom boost mode requires 'boost_expr' parameter")
	})

	t.Run("invalid custom boost mode empty expression", func(t *testing.T) {
		function := NewFunction().
			WithName("test_boost").
			WithType(FunctionTypeRerank).
			WithParam("reranker", "boost").
			WithParam("boost_mode", "custom").
			WithParam("boost_expr", "")

		err := function.ValidateParams()
		assert.Error(t, err)
		assert.Contains(t, err.Error(), "custom boost mode requires 'boost_expr' parameter")
	})

	t.Run("invalid custom boost mode whitespace only expression", func(t *testing.T) {
		function := NewFunction().
			WithName("test_boost").
			WithType(FunctionTypeRerank).
			WithParam("reranker", "boost").
			WithParam("boost_mode", "custom").
			WithParam("boost_expr", "   ")

		err := function.ValidateParams()
		assert.Error(t, err)
		assert.Contains(t, err.Error(), "custom boost mode requires 'boost_expr' parameter")
	})

	t.Run("valid function without boost mode", func(t *testing.T) {
		function := NewFunction().
			WithName("test_function").
			WithType(FunctionTypeBM25).
			WithParam("some_param", "value")

		err := function.ValidateParams()
		assert.NoError(t, err)
	})
}

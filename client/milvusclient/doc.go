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

// Package milvusclient implements the official Go Milvus client for v2.
//
// # Examples
//
// See read_example_test.go for usage examples including:
//   - Basic search and query operations
//   - Hybrid search with multiple vectors
//   - Expression-based reranking (ExampleClient_Search_exprRerank)
//   - WASM-based reranking (ExampleClient_Search_wasmRerank)
//
// Expression-based reranking:
//   - client/milvusclient/expr_impl/expr_match.go - Full standalone example with test suite
//
// WASM-based reranking:
//   - client/milvusclient/wasm_impl/wasm_rerank.go - Full standalone example with test suite
//   - tests/reranker_wasm/ - Rust/Python WASM implementations
package milvusclient

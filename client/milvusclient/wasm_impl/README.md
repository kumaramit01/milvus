# WASM Reranking Implementation Example

This directory contains a complete, runnable example of WASM-based reranking with the Milvus Go client.

## Overview

`wasm_rerank.go` is a standalone program that demonstrates how to:
- Load a compiled WASM module
- Create a collection with fields needed for WASM reranking
- Insert test data with multiple attributes (popularity, quality, timestamps)
- Perform searches with different WASM entry points
- Test multiple reranking strategies

## Prerequisites

### 1. Build the Rust WASM Module

```bash
cd tests/reranker_wasm/rust_reranker
rustup target add wasm32-unknown-unknown
cargo build --target wasm32-unknown-unknown --release
```

The WASM module will be created at:
```
tests/reranker_wasm/rust_reranker/target/wasm32-unknown-unknown/release/rust_reranker.wasm
```

### 2. Start Milvus

Make sure you have a Milvus server running at `localhost:19530`.

## Running the Example

```bash
cd client/milvusclient/wasm_impl
go run wasm_rerank.go
```

## What It Does

The program will:

1. **Connect to Milvus** at localhost:19530
2. **Load WASM Module** from tests/reranker_wasm/
3. **Create Collection** with:
   - `id` (Int64, primary key)
   - `title` (VarChar)
   - `embedding` (FloatVector, dim=128)
   - `popularity` (Float)
   - `quality_score` (Float)
   - `created_at` (Int64, timestamp)
4. **Insert Test Data** with 8 documents having different characteristics
5. **Run Test Suite** with multiple WASM entry points:
   - Simple position decay
   - Popularity-based boosting
   - Quality + popularity combined
   - Various mathematical transformations
6. **Display Results** showing how different WASM functions affect ranking
7. **Clean Up** by dropping the test collection

## Available WASM Entry Points

The example tests these Rust WASM functions:

| Entry Point | Parameters | Description |
|------------|------------|-------------|
| `rerank` | `score, rank` | Simple position decay: `score * (1.0 / (1.0 + 0.05 * rank))` |
| `rerank_with_popularity` | `score, rank, popularity` | Logarithmic popularity boost |
| `rerank_complex` | `score, rank, popularity, quality` | Multi-factor scoring with sqrt combination |
| `simple_boost` | `score, rank` | Simple 1.5x score multiplier |
| `log_boost_rerank` | `score, rank` | Logarithmic position boost |
| `time_decay_rerank` | `score, rank` | Exponential time decay |
| `nonlinear_rerank` | `score, rank` | Sigmoid transformation with clamping |

## Example Output

```
🔗 Connecting to Milvus...
✅ Connected to Milvus successfully

📦 Loading WASM module...
Loading WASM module from: /path/to/rust_reranker.wasm
WASM module size: 1.23 KB
✅ WASM module loaded successfully

🏗️  Setting up test collection...
✅ Collection 'wasm_rerank_test_collection' created successfully

📝 Inserting test data...
✅ Test data inserted and collection loaded successfully

🧪 Running WASM Reranker tests...

=== Running Test: Popularity Boost ===
Description: Test popularity-based boosting with logarithmic scaling
Entry Point: rerank_with_popularity
Input Fields: [popularity]
Expected Top Result: Viral Trending Content
Results Found: 8
WASM reranked results:
  [1] ID: 123, Score: 0.8543
      Title: Viral Trending Content
      Age: 1 days, Quality: 0.75, Popularity: 10000
  [2] ID: 124, Score: 0.7821
      Title: Recent High-Quality Post
      ...
✅ PASS: Expected 'Viral Trending Content' at top

## Using WASM Reranking in Your Application

### Basic Usage

```go
// 1. Load and encode your WASM module
wasmBytes, _ := os.ReadFile("path/to/reranker.wasm")
wasmBase64 := base64.StdEncoding.EncodeToString(wasmBytes)

// 2. Create WASM reranker function for search
wasmReranker := entity.NewFunction().
    WithName("my_reranker").
    WithType(entity.FunctionTypeRerank).
    WithInputFields("popularity", "quality_score").
    WithParam("reranker", "wasm").
    WithParam("wasm_code", wasmBase64).
    WithParam("entry_point", "rerank_complex")

// 3. Use in search
results, _ := client.Search(ctx, milvusclient.NewSearchOption(
    collectionName,
    10,
    []entity.Vector{entity.FloatVector(queryVector)},
).WithOutputFields("popularity", "quality_score").
  WithQueryNodeReranker(wasmReranker))
```

### Advanced: Multiple Rerankers

You can combine WASM reranking at QueryNode level with decay reranking at Proxy level:

```go
// QueryNode WASM reranker
wasmReranker := entity.NewFunction().
    WithName("wasm_rerank").
    WithType(entity.FunctionTypeRerank).
    WithInputFields("popularity", "quality").
    WithParam("reranker", "wasm").
    WithParam("wasm_code", wasmBase64).
    WithParam("entry_point", "rerank_complex")

// Proxy decay reranker
decayReranker := entity.NewFunction().
    WithName("time_decay").
    WithType(entity.FunctionTypeRerank).
    WithInputFields("created_at").
    WithParam("reranker", "decay").
    WithParam("function", "exp").
    WithParam("origin", time.Now().UnixMilli()).
    WithParam("scale", 7*24*60*60*1000) // 1 week

// Use both in search
results, _ := client.Search(ctx, option.
    WithQueryNodeReranker(wasmReranker).
    WithFunctionReranker(decayReranker))
```

## Writing Custom WASM Rerankers

See the Rust implementations in `tests/reranker_wasm/rust_reranker/src/lib.rs` for examples.

Basic template:

```rust
#[no_mangle]
pub extern "C" fn my_rerank(score: f32, rank: i32, field1: f32, field2: f32) -> f32 {
    // Your custom scoring logic
    let boost = 1.0 + field1 / 100.0;
    let decay = 1.0 / (1.0 + 0.05 * rank as f32);
    score * boost * decay
}
```

Key points:
- Use `#[no_mangle]` and `pub extern "C"`
- First two parameters are always `score: f32, rank: i32`
- Additional parameters match your `InputFields` in order
- Return type must be `f32`
- Keep it simple - avoid allocations, complex dependencies

## Comparison with Expression Reranking

| Feature | WASM | Expression |
|---------|------|------------|
| Performance | ~2M ops/sec (no fields) | ~500k ops/sec |
| Flexibility | Full programming language | Expression syntax |
| Compilation | Required (Rust → WASM) | None (interpreted) |
| Best For | High-performance, complex logic | Rapid iteration, simple logic |

## See Also

- **Expression Reranking**: `client/milvusclient/expr_impl/expr_match.go`
- **WASM Examples**: `tests/reranker_wasm/`
- **Python Examples**: `tests/reranker_wasm/python_wasm_example.py`
- **Rust Source**: `tests/reranker_wasm/rust_reranker/src/lib.rs`

## Troubleshooting

**"WASM module not found"**
- Build the Rust module first (see Prerequisites)
- Check the path in `findWasmModule()`

**"Entry point not found"**
- Verify the entry point name matches the Rust function
- Use `wasm-objdump -x module.wasm | grep export` to list exports

**Poor Performance**
- Minimize field access in WASM functions
- Use f32 instead of f64
- Avoid allocations and complex operations

## License

Apache License 2.0 - See LICENSE file for details.


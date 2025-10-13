#!/usr/bin/env python3
"""
Examples of WASM-based reranking with Milvus.
"""

import base64
import os
import subprocess
from pathlib import Path
from pymilvus import (
    connections, Collection, CollectionSchema, FieldSchema,
    DataType, Function
)
import numpy as np
from datetime import datetime


def build_rust_wasm_module():
    """Build the Rust WASM module if not already built"""
    wasm_dir = Path(__file__).parent / "rust_reranker"
    wasm_output = wasm_dir / "target" / "wasm32-unknown-unknown" / "release" / "rust_reranker.wasm"
    
    if wasm_output.exists():
        print(f"✅ WASM module already exists at {wasm_output}")
        return wasm_output
    
    print("Building Rust WASM module...")
    print(f"Working directory: {wasm_dir}")
    
    if not wasm_dir.exists():
        print(f"❌ Rust reranker directory not found: {wasm_dir}")
        return None
    
    try:
        # Check if Rust is installed
        result = subprocess.run(["rustc", "--version"], capture_output=True, text=True)
        print(f"Rust version: {result.stdout.strip()}")
        
        # Check if wasm32 target is installed
        subprocess.run(
            ["rustup", "target", "add", "wasm32-unknown-unknown"],
            cwd=wasm_dir,
            check=True
        )
        
        # Build the WASM module
        print("Compiling Rust to WASM...")
        subprocess.run(
            ["cargo", "build", "--target", "wasm32-unknown-unknown", "--release"],
            cwd=wasm_dir,
            check=True
        )
        
        if wasm_output.exists():
            print(f"✅ WASM module built successfully: {wasm_output}")
            return wasm_output
        else:
            print(f"❌ Build succeeded but output not found: {wasm_output}")
            return None
            
    except FileNotFoundError:
        print("❌ Rust toolchain not found. Install it from https://rustup.rs/")
        return None
    except subprocess.CalledProcessError as e:
        print(f"❌ Build failed: {e}")
        return None


def load_wasm_module(wasm_path):
    """Load and encode WASM module to base64"""
    if not os.path.exists(wasm_path):
        print(f"❌ WASM file not found: {wasm_path}")
        return None
    
    with open(wasm_path, 'rb') as f:
        wasm_bytes = f.read()
        wasm_base64 = base64.b64encode(wasm_bytes).decode('utf-8')
        print(f"✅ WASM module loaded: {len(wasm_bytes)} bytes → {len(wasm_base64)} bytes (base64)")
        return wasm_base64


def create_collection_with_wasm_reranker(collection_name, wasm_base64, entry_point, input_fields):
    """Create a Milvus collection with WASM-based reranker"""
    
    # Define reranker function
    rerank_func = Function(
        name="wasm_reranker",
        function_type="Rerank",
        input_field_names=input_fields,
        output_field_names=[],
        params={
            "reranker": "wasm",
            "wasm_code": wasm_base64,
            "entry_point": entry_point,
        }
    )
    
    # Define collection schema
    fields = [
        FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
        FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=128),
        FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=1000),
        FieldSchema(name="popularity", dtype=DataType.FLOAT),
        FieldSchema(name="quality_score", dtype=DataType.FLOAT),
        FieldSchema(name="created_at", dtype=DataType.INT64),
    ]
    
    schema = CollectionSchema(
        fields=fields,
        description=f"Collection with WASM reranker: {collection_name}",
        functions=[rerank_func]
    )
    
    # Drop existing collection
    if Collection.exists(collection_name):
        Collection(collection_name).drop()
    
    # Create collection
    return Collection(name=collection_name, schema=schema)


def example_simple_position_decay(wasm_base64):
    """Example 1: Simple position-based decay without field data"""
    print("\n" + "="*70)
    print("Example 1: Simple Position-Based Decay (No Field Data)")
    print("="*70)
    
    print("Entry point: rerank")
    print("Description: Apply position-based decay using rank only")
    print("Formula: score * (1.0 / (1.0 + 0.05 * rank))")
    
    collection = create_collection_with_wasm_reranker(
        "wasm_position_decay",
        wasm_base64,
        "rerank",
        []  # No input fields
    )
    
    # Insert data
    embeddings = np.random.rand(50, 128).astype(np.float32).tolist()
    texts = [f"Document {i}" for i in range(50)]
    popularities = [100.0] * 50
    quality_scores = [75.0] * 50
    created_ats = [int(datetime.now().timestamp() * 1000)] * 50
    
    collection.insert([embeddings, texts, popularities, quality_scores, created_ats])
    
    # Create index
    index_params = {
        "index_type": "IVF_FLAT",
        "metric_type": "L2",
        "params": {"nlist": 128}
    }
    collection.create_index(field_name="embedding", index_params=index_params)
    collection.load()
    
    # Search
    query_vector = np.random.rand(1, 128).astype(np.float32).tolist()
    results = collection.search(
        data=query_vector,
        anns_field="embedding",
        param={"metric_type": "L2", "params": {"nprobe": 10}},
        limit=10,
        output_fields=["text"]
    )
    
    print("\nTop 10 results (position-decayed):")
    for hits in results:
        for rank, hit in enumerate(hits):
            expected_decay = 1.0 / (1.0 + 0.05 * rank)
            print(f"  {rank+1}. ID={hit.id}, Score={hit.score:.4f}, Expected decay={expected_decay:.4f}")
    
    collection.drop()
    print("\n✅ Example completed successfully!")


def example_popularity_boost(wasm_base64):
    """Example 2: Boost scores based on popularity field"""
    print("\n" + "="*70)
    print("Example 2: Popularity-Based Boosting (With Field Data)")
    print("="*70)
    
    print("Entry point: rerank_with_popularity")
    print("Description: Boost scores using popularity field with logarithmic scaling")
    print("Formula: score * (1.0 + ln(popularity + 1) / 10.0) * position_decay")
    
    collection = create_collection_with_wasm_reranker(
        "wasm_popularity_boost",
        wasm_base64,
        "rerank_with_popularity",
        ["popularity"]
    )
    
    # Insert data with varying popularity
    embeddings = np.random.rand(50, 128).astype(np.float32).tolist()
    texts = [f"Product {i}" for i in range(50)]
    
    # Create items with different popularity levels
    popularities = [1.0, 10.0, 100.0, 1000.0, 10000.0] * 10
    quality_scores = [75.0] * 50
    created_ats = [int(datetime.now().timestamp() * 1000)] * 50
    
    collection.insert([embeddings, texts, popularities, quality_scores, created_ats])
    
    index_params = {
        "index_type": "IVF_FLAT",
        "metric_type": "L2",
        "params": {"nlist": 128}
    }
    collection.create_index(field_name="embedding", index_params=index_params)
    collection.load()
    
    query_vector = np.random.rand(1, 128).astype(np.float32).tolist()
    results = collection.search(
        data=query_vector,
        anns_field="embedding",
        param={"metric_type": "L2", "params": {"nprobe": 10}},
        limit=10,
        output_fields=["text", "popularity"]
    )
    
    print("\nTop 10 results (popularity-boosted):")
    for hits in results:
        for rank, hit in enumerate(hits):
            popularity = hit.entity.get('popularity', 0)
            boost_factor = 1.0 + np.log(popularity + 1) / 10.0
            print(f"  {rank+1}. ID={hit.id}, Score={hit.score:.4f}, Popularity={popularity:.0f}, Boost={boost_factor:.3f}x")
    
    collection.drop()
    print("\n✅ Example completed successfully!")


def example_time_decay(wasm_base64):
    """Example 3: Time-based decay for recency boosting"""
    print("\n" + "="*70)
    print("Example 3: Time-Based Decay (Recency Boosting)")
    print("="*70)
    
    print("Entry point: rerank_with_timestamp")
    print("Description: Boost recent items with exponential time decay")
    
    collection = create_collection_with_wasm_reranker(
        "wasm_time_decay",
        wasm_base64,
        "rerank_with_timestamp",
        ["created_at"]
    )
    
    # Insert data with different timestamps
    embeddings = np.random.rand(30, 128).astype(np.float32).tolist()
    texts = [f"Article {i}" for i in range(30)]
    popularities = [100.0] * 30
    quality_scores = [75.0] * 30
    
    # Reference time in WASM module (Jan 1, 2024)
    reference_time = 1704067200000
    day_ms = 24 * 3600 * 1000
    
    # Create items with specific ages
    created_ats = [
        reference_time,              # Today
        reference_time - 7 * day_ms,  # 1 week ago
        reference_time - 30 * day_ms, # 1 month ago
        reference_time - 60 * day_ms, # 2 months ago
        reference_time - 90 * day_ms, # 3 months ago
    ] * 6
    
    collection.insert([embeddings, texts, popularities, quality_scores, created_ats])
    
    index_params = {
        "index_type": "IVF_FLAT",
        "metric_type": "L2",
        "params": {"nlist": 128}
    }
    collection.create_index(field_name="embedding", index_params=index_params)
    collection.load()
    
    query_vector = np.random.rand(1, 128).astype(np.float32).tolist()
    results = collection.search(
        data=query_vector,
        anns_field="embedding",
        param={"metric_type": "L2", "params": {"nprobe": 10}},
        limit=10,
        output_fields=["text", "created_at"]
    )
    
    print("\nTop 10 results (time-decayed):")
    for hits in results:
        for rank, hit in enumerate(hits):
            created_at = hit.entity.get('created_at', 0)
            age_days = (reference_time - created_at) / (1000 * 60 * 60 * 24)
            print(f"  {rank+1}. ID={hit.id}, Score={hit.score:.4f}, Age={age_days:.0f} days")
    
    collection.drop()
    print("\n✅ Example completed successfully!")


def example_complex_multi_field(wasm_base64):
    """Example 4: Complex reranking with multiple fields"""
    print("\n" + "="*70)
    print("Example 4: Complex Multi-Field Reranking")
    print("="*70)
    
    print("Entry point: rerank_complex")
    print("Description: Combine popularity and quality with non-linear scaling")
    
    collection = create_collection_with_wasm_reranker(
        "wasm_complex_rerank",
        wasm_base64,
        "rerank_complex",
        ["popularity", "quality_score"]
    )
    
    # Insert diverse data
    embeddings = np.random.rand(100, 128).astype(np.float32).tolist()
    texts = [f"Item {i}" for i in range(100)]
    
    # Varied popularity and quality
    popularities = (np.random.exponential(200, 100) + 1).tolist()
    quality_scores = (np.random.rand(100) * 0.5 + 0.5).tolist()  # 0.5-1.0 range
    
    created_ats = [int(datetime.now().timestamp() * 1000)] * 100
    
    collection.insert([embeddings, texts, popularities, quality_scores, created_ats])
    
    index_params = {
        "index_type": "IVF_FLAT",
        "metric_type": "L2",
        "params": {"nlist": 128}
    }
    collection.create_index(field_name="embedding", index_params=index_params)
    collection.load()
    
    query_vector = np.random.rand(1, 128).astype(np.float32).tolist()
    results = collection.search(
        data=query_vector,
        anns_field="embedding",
        param={"metric_type": "L2", "params": {"nprobe": 10}},
        limit=10,
        output_fields=["text", "popularity", "quality_score"]
    )
    
    print("\nTop 10 results (complex multi-field reranking):")
    for hits in results:
        for rank, hit in enumerate(hits):
            popularity = hit.entity.get('popularity', 0)
            quality = hit.entity.get('quality_score', 0)
            print(f"  {rank+1}. Score={hit.score:.4f}, Pop={popularity:.0f}, Quality={quality:.3f}")
    
    collection.drop()
    print("\n✅ Example completed successfully!")


def main():
    """Run all WASM reranking examples"""
    print("\n" + "╔" + "="*68 + "╗")
    print("║" + " "*12 + "WASM-Based Reranker Examples" + " "*28 + "║")
    print("╚" + "="*68 + "╝")
    
    # Build or load WASM module
    wasm_path = build_rust_wasm_module()
    if not wasm_path:
        print("\n❌ Could not build or find WASM module")
        print("Please build it manually:")
        print("  cd examples/rust_reranker")
        print("  cargo build --target wasm32-unknown-unknown --release")
        return
    
    wasm_base64 = load_wasm_module(wasm_path)
    if not wasm_base64:
        return
    
    # Connect to Milvus
    try:
        connections.connect(host="localhost", port="19530")
        print("\n✅ Connected to Milvus")
    except Exception as e:
        print(f"\n❌ Error connecting to Milvus: {e}")
        print("Make sure Milvus is running on localhost:19530")
        return
    
    try:
        example_simple_position_decay(wasm_base64)
        example_popularity_boost(wasm_base64)
        example_time_decay(wasm_base64)
        example_complex_multi_field(wasm_base64)
        
        print("\n" + "╔" + "="*68 + "╗")
        print("║" + " "*15 + "All examples completed successfully!" + " "*17 + "║")
        print("╚" + "="*68 + "╝\n")
        
    except Exception as e:
        print(f"\n❌ Error running examples: {e}")
        import traceback
        traceback.print_exc()
    finally:
        connections.disconnect("default")


if __name__ == "__main__":
    main()


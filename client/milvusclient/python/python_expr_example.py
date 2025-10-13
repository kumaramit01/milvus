#!/usr/bin/env python3
"""
Examples of expression-based reranking with Milvus.
"""

from pymilvus import (
    connections, Collection, CollectionSchema, FieldSchema,
    DataType, Function
)
import numpy as np
from datetime import datetime, timedelta


def create_collection_with_expr_reranker(collection_name, expr_code, input_fields):
    """
    Create a Milvus collection with expression-based reranker.
    
    Args:
        collection_name: Name of the collection
        expr_code: Expression code for reranking
        input_fields: List of field names used in the expression
    """
    # Define reranker function
    rerank_func = Function(
        name="expr_reranker",
        function_type="Rerank",
        input_field_names=input_fields,
        output_field_names=[],
        params={
            "reranker": "expr",
            "expr_code": expr_code,
        }
    )

    # Define collection schema
    fields = [
        FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
        FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=128),
        FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=1000),
        FieldSchema(name="quality_score", dtype=DataType.FLOAT),
        FieldSchema(name="popularity", dtype=DataType.INT64),
        FieldSchema(name="created_at", dtype=DataType.INT64),
        FieldSchema(name="category", dtype=DataType.VARCHAR, max_length=50),
        FieldSchema(name="price", dtype=DataType.FLOAT),
    ]

    schema = CollectionSchema(
        fields=fields,
        description=f"Collection with expression reranker: {collection_name}",
        functions=[rerank_func]
    )

    # Drop existing collection
    if Collection.exists(collection_name):
        Collection(collection_name).drop()

    # Create collection
    return Collection(name=collection_name, schema=schema)


def example_quality_boost():
    """Example 1: Boost results based on quality score (0-100 scale)"""
    print("\n" + "="*70)
    print("Example 1: Quality-Based Boosting")
    print("="*70)
    
    expr_code = 'score * (fields["quality_score"] / 100.0)'
    print(f"Expression: {expr_code}")
    print("Description: Multiply score by normalized quality (0-1 range)")
    
    collection = create_collection_with_expr_reranker(
        "quality_boost_example",
        expr_code,
        ["quality_score"]
    )
    
    # Insert sample data
    embeddings = np.random.rand(100, 128).astype(np.float32).tolist()
    texts = [f"Document {i}" for i in range(100)]
    quality_scores = (np.random.rand(100) * 100).tolist()
    popularities = (np.random.randint(1, 1000, 100)).tolist()
    
    # Current timestamp minus random days
    now_ms = int(datetime.now().timestamp() * 1000)
    created_ats = [now_ms - np.random.randint(0, 90) * 24 * 3600 * 1000 for _ in range(100)]
    
    categories = np.random.choice(["standard", "premium", "featured"], 100).tolist()
    prices = (np.random.rand(100) * 100).tolist()
    
    collection.insert([
        embeddings, texts, quality_scores, popularities, 
        created_ats, categories, prices
    ])
    
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
        limit=5,
        output_fields=["text", "quality_score"]
    )
    
    print("\nTop 5 results (reranked by quality):")
    for hits in results:
        for rank, hit in enumerate(hits):
            quality = hit.entity.get('quality_score', 0)
            print(f"  {rank+1}. ID={hit.id}, Score={hit.score:.4f}, Quality={quality:.2f}, Text={hit.entity.get('text')}")
    
    collection.drop()
    print("\n✅ Example completed successfully!")


def example_recency_boost():
    """Example 2: Boost recent content with exponential time decay"""
    print("\n" + "="*70)
    print("Example 2: Recency-Based Boosting with Time Decay")
    print("="*70)
    
    now_ms = int(datetime.now().timestamp() * 1000)
    expr_code = f'let age_days = ({now_ms} - fields["created_at"]) / 86400000; score * exp(-0.01 * age_days)'
    
    print(f"Expression: {expr_code}")
    print("Description: Recent items get exponentially higher scores")
    print("  - Items from today: ~1.0x boost")
    print("  - Items from 30 days ago: ~0.74x boost")
    print("  - Items from 90 days ago: ~0.41x boost")
    
    collection = create_collection_with_expr_reranker(
        "recency_boost_example",
        expr_code,
        ["created_at"]
    )
    
    # Insert data with specific timestamps
    embeddings = np.random.rand(20, 128).astype(np.float32).tolist()
    texts = [f"Article {i}" for i in range(20)]
    quality_scores = [80.0] * 20
    popularities = [500] * 20
    
    # Create items with different ages
    day_ms = 24 * 3600 * 1000
    created_ats = [
        now_ms,                # Today
        now_ms - 1 * day_ms,   # 1 day ago
        now_ms - 7 * day_ms,   # 1 week ago
        now_ms - 14 * day_ms,  # 2 weeks ago
        now_ms - 30 * day_ms,  # 1 month ago
        now_ms - 60 * day_ms,  # 2 months ago
        now_ms - 90 * day_ms,  # 3 months ago
    ] * 3  # Repeat to have 21 items
    created_ats = created_ats[:20]
    
    categories = ["standard"] * 20
    prices = [50.0] * 20
    
    collection.insert([
        embeddings, texts, quality_scores, popularities,
        created_ats, categories, prices
    ])
    
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
        limit=7,
        output_fields=["text", "created_at"]
    )
    
    print("\nTop 7 results (reranked by recency):")
    for hits in results:
        for rank, hit in enumerate(hits):
            created_at = hit.entity.get('created_at', 0)
            age_days = (now_ms - created_at) / (1000 * 60 * 60 * 24)
            print(f"  {rank+1}. ID={hit.id}, Score={hit.score:.4f}, Age={age_days:.1f} days, Text={hit.entity.get('text')}")
    
    collection.drop()
    print("\n✅ Example completed successfully!")


def example_popularity_log_boost():
    """Example 3: Logarithmic popularity boost to prevent extreme values"""
    print("\n" + "="*70)
    print("Example 3: Popularity Boost with Logarithmic Scaling")
    print("="*70)
    
    expr_code = 'score * (1.0 + log(fields["popularity"] + 1) / 10.0)'
    print(f"Expression: {expr_code}")
    print("Description: Use logarithmic scaling to boost popular items without extreme scores")
    
    collection = create_collection_with_expr_reranker(
        "popularity_boost_example",
        expr_code,
        ["popularity"]
    )
    
    embeddings = np.random.rand(50, 128).astype(np.float32).tolist()
    texts = [f"Product {i}" for i in range(50)]
    quality_scores = [75.0] * 50
    
    # Create items with different popularity levels
    popularities = [1, 10, 100, 1000, 10000] * 10
    
    now_ms = int(datetime.now().timestamp() * 1000)
    created_ats = [now_ms] * 50
    categories = ["standard"] * 50
    prices = [50.0] * 50
    
    collection.insert([
        embeddings, texts, quality_scores, popularities,
        created_ats, categories, prices
    ])
    
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
    
    print("\nTop 10 results (reranked by popularity with log scaling):")
    for hits in results:
        for rank, hit in enumerate(hits):
            popularity = hit.entity.get('popularity', 0)
            boost_factor = 1.0 + np.log(popularity + 1) / 10.0
            print(f"  {rank+1}. ID={hit.id}, Score={hit.score:.4f}, Popularity={popularity}, Boost={boost_factor:.3f}x")
    
    collection.drop()
    print("\n✅ Example completed successfully!")


def example_category_conditional():
    """Example 4: Conditional boosting based on category"""
    print("\n" + "="*70)
    print("Example 4: Conditional Category-Based Boosting")
    print("="*70)
    
    expr_code = 'let boost = fields["category"] == "featured" ? 2.0 : fields["category"] == "premium" ? 1.5 : 1.0; score * boost'
    print(f"Expression: {expr_code}")
    print("Description: Apply different boosts based on category")
    print("  - featured: 2.0x boost")
    print("  - premium: 1.5x boost")
    print("  - standard: 1.0x (no boost)")
    
    collection = create_collection_with_expr_reranker(
        "category_boost_example",
        expr_code,
        ["category"]
    )
    
    embeddings = np.random.rand(30, 128).astype(np.float32).tolist()
    texts = [f"Item {i}" for i in range(30)]
    quality_scores = [70.0] * 30
    popularities = [100] * 30
    
    now_ms = int(datetime.now().timestamp() * 1000)
    created_ats = [now_ms] * 30
    
    # Mix of categories
    categories = ["featured"] * 10 + ["premium"] * 10 + ["standard"] * 10
    prices = [50.0] * 30
    
    collection.insert([
        embeddings, texts, quality_scores, popularities,
        created_ats, categories, prices
    ])
    
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
        output_fields=["text", "category"]
    )
    
    print("\nTop 10 results (reranked by category):")
    for hits in results:
        for rank, hit in enumerate(hits):
            category = hit.entity.get('category', 'unknown')
            print(f"  {rank+1}. ID={hit.id}, Score={hit.score:.4f}, Category={category}, Text={hit.entity.get('text')}")
    
    collection.drop()
    print("\n✅ Example completed successfully!")


def example_multi_factor():
    """Example 5: Complex multi-factor scoring combining quality, popularity, and recency"""
    print("\n" + "="*70)
    print("Example 5: Multi-Factor Scoring")
    print("="*70)
    
    now_ms = int(datetime.now().timestamp() * 1000)
    expr_code = f'''
    let quality = fields["quality_score"] / 100.0;
    let pop_boost = 1.0 + log(fields["popularity"] + 1) / 20.0;
    let age_days = ({now_ms} - fields["created_at"]) / 86400000;
    let recency = exp(-0.005 * age_days);
    score * quality * pop_boost * recency
    '''
    
    print(f"Expression: {expr_code.strip()}")
    print("Description: Combine quality, popularity (log), and recency (exponential decay)")
    
    collection = create_collection_with_expr_reranker(
        "multi_factor_example",
        expr_code,
        ["quality_score", "popularity", "created_at"]
    )
    
    # Create diverse dataset
    embeddings = np.random.rand(100, 128).astype(np.float32).tolist()
    texts = [f"Content {i}" for i in range(100)]
    
    # Varied quality scores
    quality_scores = (np.random.rand(100) * 50 + 50).tolist()  # 50-100 range
    
    # Varied popularity (exponential distribution)
    popularities = (np.random.exponential(200, 100) + 1).astype(int).tolist()
    
    # Varied ages (0-90 days)
    day_ms = 24 * 3600 * 1000
    created_ats = [now_ms - np.random.randint(0, 90) * day_ms for _ in range(100)]
    
    categories = ["standard"] * 100
    prices = [50.0] * 100
    
    collection.insert([
        embeddings, texts, quality_scores, popularities,
        created_ats, categories, prices
    ])
    
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
        output_fields=["text", "quality_score", "popularity", "created_at"]
    )
    
    print("\nTop 10 results (multi-factor reranking):")
    for hits in results:
        for rank, hit in enumerate(hits):
            quality = hit.entity.get('quality_score', 0)
            popularity = hit.entity.get('popularity', 0)
            created_at = hit.entity.get('created_at', 0)
            age_days = (now_ms - created_at) / (1000 * 60 * 60 * 24)
            print(f"  {rank+1}. Score={hit.score:.4f}, Quality={quality:.1f}, Pop={popularity}, Age={age_days:.0f}d")
    
    collection.drop()
    print("\n✅ Example completed successfully!")


def main():
    """Run all examples"""
    print("\n" + "╔" + "="*68 + "╗")
    print("║" + " "*10 + "Expression-Based Reranker Examples" + " "*24 + "║")
    print("╚" + "="*68 + "╝")
    
    # Connect to Milvus
    try:
        connections.connect(host="localhost", port="19530")
        print("\n✅ Connected to Milvus")
    except Exception as e:
        print(f"\n❌ Error connecting to Milvus: {e}")
        print("Make sure Milvus is running on localhost:19530")
        return
    
    try:
        example_quality_boost()
        example_recency_boost()
        example_popularity_log_boost()
        example_category_conditional()
        example_multi_factor()
        
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


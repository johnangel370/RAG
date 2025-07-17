import json
import time
from collections import Counter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document


if __name__ == '__main__':
    # Load patent data
    with open('.\\data\\patent_samples.json', 'r', encoding='utf-8') as f:
        patent_samples = json.load(f)

    with open('patent_triz_results.json', 'r', encoding='utf-8') as f:
        patent_triz_results = json.load(f)

    # Create Document objects for patents with TRIZ parameters
    documents = []
    for patent, patent_triz in zip(patent_samples, patent_triz_results):
        # Create content that includes title, abstract, and TRIZ parameter names
        triz_params = ", ".join([param['name'] for param in patent_triz['top_parameters']])
        content = f"Title: {patent['title']}\nAbstract: {patent['abstract']}\nTRIZ Parameters: {triz_params}"
        
        # Create Document with metadata
        doc = Document(
            page_content=content,
            metadata={
                'id': patent['id'],
                'doc_num': patent['doc_num'],
                'title': patent['title'],
                'abstract': patent['abstract'],
                'triz_parameters': json.dumps(patent_triz['top_parameters'])
            }
        )
        documents.append(doc)

    print(f"Loaded {len(documents)} patent documents")

    # Use embedding model 
    # # "thenlper/gte-small"
    embedding_model = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        multi_process=True,
        model_kwargs={"device": "cuda"},
        encode_kwargs={"normalize_embeddings": True},
    )

    print("Setting retrieval model...")
    start_retrieval = time.time()
    # Create embeddings for patents
    vectorstore = Chroma.from_documents(documents=documents, embedding=embedding_model)

    # Create a retriever
    retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 10})
    end_retrieval = time.time()
    print(f"End of retrieval. Retrieval model setting time: {end_retrieval - start_retrieval:.2f} seconds\n")

    # Query top 10 patents based on TRIZ parameters
    query = "reduce size of the body of a car"
    print("Start querying...")
    start = time.time()
    relevant_patents = retriever.invoke(query)
    end = time.time()
    print(f"End of query. Query time: {end - start:.2f} seconds\n")

    print("Question:", query)
    print(f"Top 10 most relevant patents:")
    print("=" * 50)
    
    # Display results
    for i, patent in enumerate(relevant_patents, 1):
        print(f"{i}. ID: {patent.metadata['id']}")
        print(f"   Title: {patent.metadata['title']}")
        triz_params = json.loads(patent.metadata['triz_parameters'])
        print(f"   TRIZ Parameters: {', '.join([param['name'] for param in triz_params])}")
        print(f"   Abstract: {patent.metadata['abstract'][:200]}...")
        print()
    
    # =================== VERIFICATION METHODS ===================
    print("\n" + "=" * 60)
    print("VERIFICATION RESULTS")
    print("=" * 60)
    
    # 1. Get results with similarity scores
    print("\n1. RELEVANCE SCORE VERIFICATION:")
    print("-" * 40)
    relevant_patents_with_scores = vectorstore.similarity_search_with_score(query, k=10)
    for i, (patent, score) in enumerate(relevant_patents_with_scores, 1):
        print(f"{i}. ID: {patent.metadata['id']} (Similarity Score: {score:.4f})")
    
    # 2. TRIZ Parameter Matching Verification
    print("\n2. TRIZ PARAMETER MATCHING VERIFICATION:")
    print("-" * 40)
    query_keywords = [word.lower() for word in query.split() if len(word) > 2]  # Filter short words
    print(f"Query keywords: {query_keywords}")
    
    for i, patent in enumerate(relevant_patents, 1):
        triz_params = json.loads(patent.metadata['triz_parameters'])
        matching_params = []
        for param in triz_params:
            for keyword in query_keywords:
                if keyword in param['name'].lower():
                    matching_params.append(f"{param['name']} (score: {param['score']:.3f})")
                    break
        
        if matching_params:
            print(f"{i}. ID: {patent.metadata['id']} - Matching TRIZ params: {matching_params}")
        else:
            print(f"{i}. ID: {patent.metadata['id']} - No direct keyword matches in TRIZ params")
    
    # 3. Content Relevance Check
    print("\n3. CONTENT RELEVANCE VERIFICATION:")
    print("-" * 40)
    for i, patent in enumerate(relevant_patents, 1):
        content_lower = patent.page_content.lower()
        query_terms_found = [term for term in query_keywords if term in content_lower]
        coverage_ratio = len(query_terms_found) / len(query_keywords) if query_keywords else 0
        print(f"{i}. ID: {patent.metadata['id']} - Query terms found: {query_terms_found} ({coverage_ratio:.1%} coverage)")
    
    # 4. Statistical Verification
    print("\n4. STATISTICAL VERIFICATION:")
    print("-" * 40)
    print(f"Total patents retrieved: {len(relevant_patents)}")
    print(f"Unique patent IDs: {len(set(patent.metadata['id'] for patent in relevant_patents))}")
    
    # TRIZ parameter distribution in results
    all_triz_params = []
    all_scores = []
    for patent in relevant_patents:
        triz_params = json.loads(patent.metadata['triz_parameters'])
        for param in triz_params:
            all_triz_params.append(param['name'])
            all_scores.append(param['score'])
    
    param_counts = Counter(all_triz_params)
    print(f"\nMost common TRIZ parameters in results:")
    for param, count in param_counts.most_common(10):
        print(f"  - {param}: {count} occurrences")
    
    # Score distribution
    if all_scores:
        avg_score = sum(all_scores) / len(all_scores)
        max_score = max(all_scores)
        min_score = min(all_scores)
        print(f"\nTRIZ parameter scores in results:")
        print(f"  - Average score: {avg_score:.3f}")
        print(f"  - Max score: {max_score:.3f}")
        print(f"  - Min score: {min_score:.3f}")
    
    # 5. Query-Result Relevance Summary
    print("\n5. QUERY-RESULT RELEVANCE SUMMARY:")
    print("-" * 40)
    
    # Calculate average similarity score
    avg_similarity = sum(score for _, score in relevant_patents_with_scores) / len(relevant_patents_with_scores)
    print(f"Average similarity score: {avg_similarity:.4f}")
    
    # Count patents with keyword matches
    patents_with_matches = 0
    for patent in relevant_patents:
        content_lower = patent.page_content.lower()
        if any(term in content_lower for term in query_keywords):
            patents_with_matches += 1
    
    print(f"Patents with query keyword matches: {patents_with_matches}/{len(relevant_patents)} ({patents_with_matches/len(relevant_patents):.1%})")
    
    # Recommend query refinement if low relevance
    if avg_similarity < 0.3:
        print("\n⚠️  LOW RELEVANCE WARNING: Consider refining your query for better results.")
    elif avg_similarity > 0.7:
        print("\n✅ HIGH RELEVANCE: Query results show strong similarity to your search.")
    else:
        print("\n✓ MODERATE RELEVANCE: Query results show reasonable similarity to your search.")

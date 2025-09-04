import json
import time
from pathlib import Path
import yaml
import string
from collections import Counter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter


if __name__ == '__main__':
    # Load patent data
    patent_samples_path = Path(r'./data/patent_samples.json')
    labeled_results_path = Path(r'./results/labeled_results/patent_score_table.json')
    triz_parameters_path = Path(r'./data/keywords/triz_params.yaml')

    with open(patent_samples_path, 'r', encoding='utf-8') as f:
        patent_samples = json.load(f)

    with open(labeled_results_path, 'r', encoding='utf-8') as f:
        patent_score_data = json.load(f)

    with open(triz_parameters_path, 'r', encoding='utf-8') as f:
        triz_parameters = yaml.safe_load(f)
    
    # Create a lookup dictionary for patent scores
    score_lookup = {item['patent_num']: item['params_score'] for item in patent_score_data}
    
    # Initialize text splitter
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        add_start_index=True
    )

    # Create Document objects for patents with TRIZ parameter scores
    documents = []
    for patent in patent_samples:
        patent_num = patent.get('doc_num')  # Assuming patent ID matches patent_num
        
        if patent_num in score_lookup:
            scores = score_lookup[patent_num]
            
            # Get top 5 TRIZ parameters based on scores
            # sorted_params = sorted(scores.items(), key=lambda x: float(x[1]), reverse=True)[:5]
            sorted_params = sorted(
                [(param, score) for param, score in scores.items() if float(score) > 0.4],
                key=lambda x: float(x[1]),
                reverse=True
            )

            
            # Create top parameters list with names and scores
            top_params = []
            triz_param_names = []
            for param_id, score in sorted_params:
                param_name = triz_parameters.get(int(param_id), f"Parameter {param_id}")
                top_params.append({
                    'id': int(param_id),
                    'name': param_name,
                    'score': float(score),
                    'confidence': 'high' if float(score) > 0.7 else 'medium' if float(score) > 0.4 else 'low'
                })
                triz_param_names.append(param_name)
            
            # Split patent content into chunks
            patent_content = f"Abstract: {patent['abstract']}\nDescription: {patent['description']}\nClaims: {patent['claims']}"
            # patent_content = f"Title: {patent['title']}"
            chunks = text_splitter.split_text(patent_content)
            triz_params = ", ".join(triz_param_names)
            
            # Create a Document for each chunk
            for i, chunk in enumerate(chunks):
                doc = Document(
                    page_content=f"Chunk {i+1}: {chunk}\nTRIZ Parameters: {triz_params}",
                    metadata={
                        'id': patent['doc_num'],
                        'title': patent['title'],
                        'abstract': patent['abstract'],
                        'chunk_num': i+1,
                        'total_chunks': len(chunks),
                        'triz_parameters': json.dumps(top_params)
                    }
                )
                documents.append(doc)
        else:
            print(f"Warning: No score data found for patent {patent_num}")

    print(f"Loaded {len(documents)} patent documents")

    # Define query
    # query = "How can mechanical friction be reduced to improve efficiency?"
    query = input("Enter your question: ")

    # Open output file for writing results
    translator = str.maketrans('', '', string.punctuation)
    no_punc = query.translate(translator)
    filename = no_punc.replace(" ", "_") + ".txt"
    output_path = Path(r'./results/rag_results/w_unlabeled')
    # output_path = Path(r'./results/rag_results/w_labeled')
    output_filename = output_path / Path(filename)

    with open(output_filename, 'w', encoding='utf-8') as output_file:
        output_file.write(f"Patent Search Results\n")
        output_file.write(f"Query: {query}\n")
        output_file.write(f"Total documents loaded: {len(documents)}\n")

        # Use embedding model
        # thenlper/gte-small
        embedding_model = HuggingFaceEmbeddings(
            model_name="all-MiniLM-L6-v2",
            multi_process=True,
            model_kwargs={"device": "cuda"},
            encode_kwargs={"normalize_embeddings": True},
        )

        output_file.write("Setting retrieval model...\n")
        start_retrieval = time.time()
        # Create embeddings for patents
        vectorstore = Chroma.from_documents(documents=documents, embedding=embedding_model)

        # Create a retriever
        retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 10})
        end_retrieval = time.time()
        output_file.write(f"End of retrieval. Retrieval model setting time: {end_retrieval - start_retrieval:.2f} seconds\n")

        # Query top 10 patents based on TRIZ parameters
        output_file.write("Start querying...\n")
        start = time.time()
        relevant_patents = retriever.invoke(query)
        end = time.time()
        output_file.write(f"End of query. Query time: {end - start:.2f} seconds\n\n")

        reversed_relevant_patents = reversed(relevant_patents)

        output_file.write("Question: " + query + "\n")
        output_file.write(f"Top 10 most relevant patents:\n")
        output_file.write("=" * 50 + "\n")

        # Display results
        # for i, patent in enumerate(relevant_patents, 1):
        for i, patent in enumerate(reversed_relevant_patents, 1):
            result = f"{i}. ID: {patent.metadata['id']}\n"
            result += f"   Title: {patent.metadata['title']}\n"
            triz_params = json.loads(patent.metadata['triz_parameters'])
            result += f"   TRIZ Parameters: {', '.join([param['name'] for param in triz_params])}\n"
            result += f"   Abstract: {patent.metadata['abstract'][:200]}...\n\n"

            output_file.write(result)
            print(result.strip())

        # =================== VERIFICATION METHODS ===================
        output_file.write("\n" + "=" * 60 + "\n")
        output_file.write("VERIFICATION RESULTS\n")
        output_file.write("=" * 60 + "\n")
        print("\n" + "=" * 60)
        print("VERIFICATION RESULTS")
        print("=" * 60)

        # 1. Get results with similarity scores
        output_file.write("\n1. RELEVANCE SCORE VERIFICATION:\n")
        output_file.write("-" * 40 + "\n")
        print("\n1. RELEVANCE SCORE VERIFICATION:")
        print("-" * 40)

        relevant_patents_with_scores = vectorstore.similarity_search_with_score(query, k=10)
        reversed_relevant_patents_with_scores = reversed(relevant_patents_with_scores)
        for i, (patent, score) in enumerate(reversed_relevant_patents_with_scores, 1):
            score_result = f"{i}. ID: {patent.metadata['id']} (Similarity Score: {score:.4f})\n"
            output_file.write(score_result)
            print(score_result.strip())

        # 2. Statistical Verification
        output_file.write("\n2. STATISTICAL VERIFICATION:\n")
        output_file.write("-" * 40 + "\n")
        print("\n2. STATISTICAL VERIFICATION:")
        print("-" * 40)

        stats1 = f"Total patents retrieved: {len(relevant_patents)}\n"
        stats2 = f"Unique patent IDs: {len(set(patent.metadata['id'] for patent in relevant_patents))}\n"
        output_file.write(stats1)
        output_file.write(stats2)
        print(stats1.strip())
        print(stats2.strip())

        # TRIZ parameter distribution in results
        all_triz_params = []
        all_scores = []
        for patent in relevant_patents:
            triz_params = json.loads(patent.metadata['triz_parameters'])
            for param in triz_params:
                all_triz_params.append(param['name'])
                all_scores.append(param['score'])

        param_counts = Counter(all_triz_params)
        param_header = "\nMost common TRIZ parameters in results:\n"
        output_file.write(param_header)
        print(param_header.strip())

        for param, count in param_counts.most_common(10):
            param_line = f"  - {param}: {count} occurrences\n"
            output_file.write(param_line)
            print(param_line.strip())

        # Score distribution
        if all_scores:
            avg_score = sum(all_scores) / len(all_scores)
            max_score = max(all_scores)
            min_score = min(all_scores)

            score_header = "\nTRIZ parameter scores in results:\n"
            score_avg = f"  - Average score: {avg_score:.3f}\n"
            score_max = f"  - Max score: {max_score:.3f}\n"
            score_min = f"  - Min score: {min_score:.3f}\n"

            output_file.write(score_header)
            output_file.write(score_avg)
            output_file.write(score_max)
            output_file.write(score_min)
            print(score_header.strip())
            print(score_avg.strip())
            print(score_max.strip())
            print(score_min.strip())

        # 5. Query-Result Relevance Summary
        output_file.write("\n3. QUERY-RESULT RELEVANCE SUMMARY:\n")
        output_file.write("-" * 40 + "\n")
        print("\n3. QUERY-RESULT RELEVANCE SUMMARY:")
        print("-" * 40)

        # Calculate average similarity score
        avg_similarity = sum(score for _, score in relevant_patents_with_scores) / len(relevant_patents_with_scores)
        avg_sim_result = f"Average similarity score: {avg_similarity:.4f}\n"
        output_file.write(avg_sim_result)
        print(avg_sim_result.strip())

        # Recommend query refinement if low relevance
        if avg_similarity < 0.3:
            recommendation = "\n⚠️  LOW RELEVANCE WARNING: Consider refining your query for better results.\n"
        elif avg_similarity > 0.7:
            recommendation = "\n✅ HIGH RELEVANCE: Query results show strong similarity to your search.\n"
        else:
            recommendation = "\n✓ MODERATE RELEVANCE: Query results show reasonable similarity to your search.\n"

        output_file.write(recommendation)
        print(recommendation.strip())

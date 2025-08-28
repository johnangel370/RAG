import polars as pl
import numpy as np
import yaml
import pprint
from typing import List, Dict
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from langchain_huggingface import HuggingFaceEmbeddings
import json
import os
from pathlib import Path

class TRIZPatentLabeler:
    def __init__(self):

        self.triz_parameters = None
        self.parameter_keywords = None
        self.section_weights = None
        self.methods_weight = None
        
        # Initialize embeddings model
        self.embeddings_model = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True}
        )
        
        # Initialize TF-IDF vectorizer
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=5000,
            stop_words='english',
            ngram_range=(1, 2)
        )
    @staticmethod
    def load_yaml(file_path):
        with open(file_path, 'r', encoding='utf-8') as f:
            return yaml.load(f, Loader=yaml.FullLoader)

    @staticmethod
    def load_json(file_path):
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)

    @staticmethod
    def show_dict(data_dict):
        pprint.pprint(data_dict)
    
    def combine_patent_text(self, patent_doc: Dict) -> str:
        """Combine different sections of a patent document into a single text."""
        sections = []
        
        # Add title if available
        if 'title' in patent_doc and patent_doc['title']:
            sections.append(f"Title: {patent_doc['title']}")
        
        # Add abstract if available
        if 'abstract' in patent_doc and patent_doc['abstract']:
            sections.append(f"Abstract: {patent_doc['abstract']}")
        
        # Add description if available
        if 'description' in patent_doc and patent_doc['description']:
            sections.append(f"Description: {patent_doc['description']}")
        
        # Add claims if available
        if 'claims' in patent_doc and patent_doc['claims']:
            sections.append(f"Claims: {patent_doc['claims']}")
        
        return ' '.join(sections)
    
    def preprocess_text(self, text: str) -> str:
        """Preprocess patent text for analysis."""
        # Remove extra whitespace and newlines
        text = re.sub(r'\s+', ' ', text)
        # Convert to lowercase
        text = text.lower()
        # Remove special characters but keep periods and commas
        text = re.sub(r'[^\w\s.,]', '', text)
        return text.strip()
    
    def extract_patent_sections(self, text: str) -> Dict[str, str]:
        """Extract relevant sections from patent text."""
        sections = {
            'abstract': '',
            'claims': '',
            'description': ''
        }
        
        # Simple regex patterns to identify sections
        patterns = {
            'abstract': r'abstract[:\s]*(.*?)(?=\n\n|\nclaims?|background)',
            'claims': r'claims?[:\s]*(.*?)(?=\n\n|description|background)',
            'description': r'(?:detailed\s+)?description[:\s]*(.*?)(?=\n\n|claims?|abstract)'
        }
        
        for section, pattern in patterns.items():
            match = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
            if match:
                sections[section] = match.group(1).strip()
        
        # If no sections found, use entire text as description
        if not any(sections.values()):
            sections['description'] = text
            
        return sections
    
    def calculate_keyword_scores(self, text: str) -> Dict[int, float]:
        """Calculate scores for each TRIZ parameter based on keyword matching."""
        scores = {}
        text_lower = text.lower()
        
        for param_id, keywords in self.parameter_keywords.items():
            score = 0
            for keyword in keywords:
                # Count occurrences of keyword
                count = len(re.findall(r'\b' + re.escape(keyword) + r'\b', text_lower))
                score += count
            
            # Normalize by text length
            scores[param_id] = score / len(text.split()) if text.split() else 0
            
        return scores
    
    def calculate_semantic_similarity(self, text: str) -> Dict[int, float]:
        """Calculate semantic similarity between text and TRIZ parameters."""
        # Create embeddings for the text
        text_embedding = self.embeddings_model.embed_query(text)
        
        scores = {}
        for param_id, param_desc in self.triz_parameters.items():
            # Create embedding for parameter description + keywords
            param_text = param_desc + " " + " ".join(self.parameter_keywords[param_id])
            param_embedding = self.embeddings_model.embed_query(param_text)
            
            # Calculate cosine similarity
            similarity = cosine_similarity(
                [text_embedding], 
                [param_embedding]
            )[0][0]
            scores[param_id] = similarity
            
        return scores
    
    def calculate_tfidf_scores(self, text: str, reference_corpus: List[str] = None) -> Dict[int, float]:
        """Calculate TF-IDF based scores for TRIZ parameters."""
        if reference_corpus is None:
            # Use parameter descriptions as reference corpus
            reference_corpus = [
                f"{desc} {' '.join(self.parameter_keywords[param_id])}"
                for param_id, desc in self.triz_parameters.items()
            ]
        
        # Add the input text to corpus
        corpus = reference_corpus + [text]
        # print(corpus)
        
        # Fit TF-IDF vectorizer
        tfidf_matrix = self.tfidf_vectorizer.fit_transform(corpus)
        
        # Calculate similarity between text and each parameter
        text_vector = tfidf_matrix[-1]  # Last item is the input text
        param_vectors = tfidf_matrix[:-1]  # All others are parameter descriptions
        
        similarities = cosine_similarity(text_vector, param_vectors)[0]
        
        scores = {}
        for i, param_id in enumerate(self.triz_parameters.keys()):
            scores[param_id] = similarities[i]
            
        return scores
    
    def ensemble_scoring(self, text: str, weights: Dict[str, float] = None) -> Dict[int, float]:
        """Combine multiple scoring methods using ensemble approach."""
        if weights:
            weights = weights
        else:
            weights = self.methods_weight
        
        # Calculate scores using different methods
        keyword_scores = self.calculate_keyword_scores(text)
        semantic_scores = self.calculate_semantic_similarity(text)
        tfidf_scores = self.calculate_tfidf_scores(text)
        
        # Normalize scores to 0-1 range
        def normalize_scores(scores):
            max_score = max(scores.values()) if scores.values() else 1
            if max_score == 0.0:
                max_score = 1
            return {k: v / max_score for k, v in scores.items()}
        
        keyword_scores = normalize_scores(keyword_scores)
        # print("keyword_scores: ", keyword_scores)
        semantic_scores = normalize_scores(semantic_scores)
        # print("semantic_scores: ", semantic_scores)
        tfidf_scores = normalize_scores(tfidf_scores)
        # print("tfidf_scores: ", tfidf_scores)
        
        # Combine scores
        final_scores = {}
        for param_id in self.triz_parameters.keys():
            final_scores[param_id] = (
                weights['keyword'] * keyword_scores.get(param_id, 0) +
                weights['semantic'] * semantic_scores.get(param_id, 0) +
                weights['tfidf'] * tfidf_scores.get(param_id, 0)
            )
        
        return final_scores
    
    def label_patent_from_json(self, patent_doc: Dict, top_k: int = 5, threshold: float = 0.1) -> Dict:
        """Label a patent document from JSON data with TRIZ parameters."""
        # Combine patent text from JSON fields
        text = self.combine_patent_text(patent_doc)
        processed_text = self.preprocess_text(text)
        
        # Extract sections from the combined text
        sections = self.extract_patent_sections(processed_text)
        
        # Calculate scores for each section
        section_scores = {}
        for section_name, section_text in sections.items():
            if section_text:
                section_scores[section_name] = self.ensemble_scoring(section_text)
        
        # Calculate overall scores (weighted average of sections)
        section_weights = self.section_weights
        
        overall_scores = {}
        for param_id in self.triz_parameters.keys():
            weighted_score = 0
            total_weight = 0
            
            for section_name, scores in section_scores.items():
                if section_name in section_weights:
                    weight = section_weights[section_name]
                    weighted_score += weight * scores.get(param_id, 0)
                    total_weight += weight
            
            overall_scores[param_id] = weighted_score / total_weight if total_weight > 0 else 0
        
        # Filter and sort results
        filtered_scores = {
            param_id: score for param_id, score in overall_scores.items()
            if score >= threshold
        }
        
        # Get top-k parameters
        top_parameters = sorted(
            filtered_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )[:top_k]
        
        # Prepare results
        results = {
            'patent_id': patent_doc.get('id', 'unknown'),
            'doc_num': patent_doc.get('doc_num', 'unknown'),
            'title': patent_doc.get('title', 'unknown'),
            'top_parameters': [
                {
                    'id': param_id,
                    'name': self.triz_parameters[param_id],
                    'score': score,
                    'confidence': 'high' if score > 0.7 else 'medium' if score > 0.4 else 'low'
                }
                for param_id, score in top_parameters
            ],
            'all_scores': overall_scores,
            'section_analysis': {
                section: {
                    'top_parameters': sorted(scores.items(), key=lambda x: x[1], reverse=True)[:3],
                    'avg_score': np.mean(list(scores.values())) if scores else 0
                }
                for section, scores in section_scores.items()
            }
        }
        
        return results
    
    def batch_label_patents_from_json(self, json_file_path: str, output_file: str = None, limit: int = None) -> List[Dict]:
        """Label multiple patent documents from JSON file."""
        print(f"Loading patents from: {json_file_path}")
        patent_docs = self.load_json(json_file_path)
        
        if limit:
            patent_docs = patent_docs[:limit]
            print(f"Processing first {limit} patents...")
        
        results = []
        total_patents = len(patent_docs)
        
        for i, patent_doc in enumerate(patent_docs, 1):
            try:
                print(f"Processing patent {i}/{total_patents}: {patent_doc.get('title', 'Unknown title')[:50]}...")
                result = self.label_patent_from_json(patent_doc)
                results.append(result)
            except Exception as e:
                print(f"Error processing patent {patent_doc.get('id', 'unknown')}: {e}")
                continue
        
        # Save results if output file specified
        if output_file:
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            print(f"Results saved to: {output_file}")
        
        return results
    
    def export_to_csv(self, results: List[Dict], output_file: str):
        """Export results to CSV format using polars."""
        rows = []
        
        for result in results:
            # JSON-based result
            base_row = {
                'patent_id': result.get('patent_id', 'unknown'),
                'doc_num': result.get('doc_num', 'unknown'),
                'title': result.get('title', 'unknown')[:100] + '...' if len(result.get('title', '')) > 100 else result.get('title', 'unknown')
            }
            
            # Add top parameters
            for i, param in enumerate(result['top_parameters'][:5]):  # Top 5
                base_row[f'param_{i+1}_id'] = param['id']
                base_row[f'param_{i+1}_name'] = param['name']
                base_row[f'param_{i+1}_score'] = param['score']
                base_row[f'param_{i+1}_confidence'] = param['confidence']
            
            rows.append(base_row)
        
        # Create polars DataFrame and export to CSV
        df = pl.DataFrame(rows)
        df.write_csv(output_file)
        print(f"CSV exported to: {output_file}")


def main():
    """Main function to demonstrate usage."""
    labeler = TRIZPatentLabeler()

    params_data = r"./data/keywords/triz_params.yaml"
    keywords_data = r"./data/keywords/params_keywords.yaml"

    print("Loading triz parameters and keywords from yaml file...")
    labeler.triz_parameters = labeler.load_yaml(params_data)
    labeler.parameter_keywords = labeler.load_yaml(keywords_data)
    print("Loading data from yaml files COMPLETE!")

    print("\n ========== TRIZ PARAMETERS ==========")
    labeler.show_dict(labeler.triz_parameters)

    print("\n ========== TRIZ KEYWORDS ==========")
    labeler.show_dict(labeler.parameter_keywords)

    patent_file_path = Path(r"./data/patent_samples.json")
    output_folder = Path(r"./results/labeled_results")
    limit = 100

    sem_weights = [0.3, 0.3, 0.3, 0.5, 0.7]
    km_weights = [0.3, 0.2, 0.5, 0.2, 0.1]
    tfidf_weights = [0.4, 0.5, 0.2, 0.3, 0.2]

    section_weights = {
        'abstract': 0.2,
        'claims': 0.3,
        'description': 0.5
    }

    labeler.section_weights = section_weights

    # Method Weight Sensitivity Analysis
    for i, weight in enumerate(sem_weights):
        file_name = f"method_results_{i+1}.json"
        output_file = output_folder / file_name
        labeler.methods_weight = {
            'keyword': km_weights[i],
            'semantic': weight,
            'tfidf': tfidf_weights[i]
        }
        results = labeler.batch_label_patents_from_json(patent_file_path, output_file, limit)
        print(f"\nProcessed {len(results)} patents")
        print(f"Results saved to: {output_file}")

    method_weights = {
        'keyword': 0.2,
        'semantic': 0.5,
        'tfidf': 0.3
    }

    labeler.methods_weight = method_weights

    claim_weights = [0.3, 0.3, 0.3, 0.5, 0.7]
    abs_weights = [0.3, 0.2, 0.5, 0.3, 0.1]
    describe_weights = [0.4, 0.5, 0.2, 0.2, 0.2]

    # Section Weight Sensitivity Analysis
    for i, weight in enumerate(claim_weights):
        file_name = f"section_results_{i+1}.json"
        output_file = output_folder / file_name
        labeler.section_weights = {
            'abstract': abs_weights[i],
            'claims': weight,
            'description': describe_weights[i]
        }
        results = labeler.batch_label_patents_from_json(patent_file_path, output_file, limit)
        print(f"\nProcessed {len(results)} patents")
        print(f"Results saved to: {output_file}")


if __name__ == "__main__":
    main()

# References and Methodology for TRIZ Patent Labeler

## TRIZ 48 Engineering Parameters
- **Primary Source**: Altshuller, G. (1984). "Creativity as an Exact Science: The Theory of the Solution of Inventive Problems"
- **Reference**: Savransky, S. D. (2000). "Engineering of Creativity: Introduction to TRIZ Methodology of Inventive Problem Solving"
- **Online Resource**: The TRIZ Journal (https://triz-journal.com/) - Contains comprehensive lists of the 48 engineering parameters

## Ensemble Approach References

### 1. Multi-method Text Classification
- **Polikar, R.** (2006). "Ensemble based systems in decision making." IEEE Circuits and Systems Magazine, 6(3), 21-45.
- **Rokach, L.** (2010). "Ensemble-based classifiers." Artificial Intelligence Review, 33(1-2), 1-39.

### 2. Combining TF-IDF and Semantic Embeddings
- **Mikolov, T., et al.** (2013). "Efficient estimation of word representations in vector space." arXiv preprint arXiv:1301.3781.
- **Salton, G., & Buckley, C.** (1988). "Term-weighting approaches in automatic text retrieval." Information Processing & Management, 24(5), 513-523.

### 3. Patent Text Analysis
- **Trappey, A. J., et al.** (2006). "Clustering patents using non-exhaustive overlaps." Journal of Systems Science and Systems Engineering, 15(2), 162-181.
- **Yoon, B., & Park, Y.** (2004). "A text-mining-based patent network: Analytical tool for high-technology trend." The Journal of High Technology Management Research, 15(1), 37-50.

### 4. Keyword-based Classification
- **Sebastiani, F.** (2002). "Machine learning in automated text categorization." ACM Computing Surveys, 34(1), 1-47.
- **Yang, Y., & Pedersen, J. O.** (1997). "A comparative study on feature selection in text categorization." ICML, 97, 412-420.

## Methodology Justification

### Weight Distribution (Keyword: 0.3, Semantic: 0.4, TF-IDF: 0.3)
The weight distribution is based on:
1. **Semantic similarity (40%)**: Given higher weight because embeddings capture contextual meaning better than exact matches
2. **Keyword matching (30%)**: Important for domain-specific terminology but limited by exact matching
3. **TF-IDF (30%)**: Provides statistical significance but may miss semantic relationships

### References for Weight Selection:
- **Kittler, J., et al.** (1998). "On combining classifiers." IEEE Transactions on Pattern Analysis and Machine Intelligence, 20(3), 226-239.
- **Xu, L., et al.** (1992). "Methods of combining multiple classifiers and their applications to handwriting recognition." IEEE Transactions on Systems, Man, and Cybernetics, 22(3), 418-435.

## Limitations and Recommendations

### Current Limitations:
1. **Keywords**: Hand-crafted based on engineering knowledge rather than corpus analysis
2. **Weights**: Fixed weights rather than learned from training data
3. **Validation**: No ground truth dataset for validation

### Recommended Improvements:
1. **Corpus-based keyword extraction**: Use patent corpus to automatically extract relevant terms
2. **Supervised learning**: Train on labeled patent-TRIZ datasets to optimize weights
3. **Cross-validation**: Use k-fold cross-validation to evaluate performance

## Alternative Approaches in Literature

### 1. Deep Learning Approaches
- **Devlin, J., et al.** (2018). "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding." arXiv preprint arXiv:1810.04805.
- **Lee, J., et al.** (2020). "BioBERT: a pre-trained biomedical language representation model for biomedical text mining." Bioinformatics, 36(4), 1234-1240.

### 2. Topic Modeling
- **Blei, D. M., et al.** (2003). "Latent dirichlet allocation." Journal of Machine Learning Research, 3, 993-1022.
- **Griffiths, T. L., & Steyvers, M.** (2004). "Finding scientific topics." Proceedings of the National Academy of Sciences, 101(suppl 1), 5228-5235.

### 3. Graph-based Approaches
- **Mihalcea, R., & Tarau, P.** (2004). "TextRank: Bringing order into text." Proceedings of the 2004 conference on empirical methods in natural language processing, 404-411.

## Evaluation Metrics

For future validation, consider:
- **Precision/Recall**: Standard classification metrics
- **F1-Score**: Harmonic mean of precision and recall
- **Cohen's Kappa**: Inter-annotator agreement for human evaluation
- **NDCG**: Normalized Discounted Cumulative Gain for ranking evaluation

## Data Sources for Training/Validation

### Patent Databases:
- **USPTO**: United States Patent and Trademark Office database
- **EPO**: European Patent Office database
- **Google Patents**: Public patent search engine
- **WIPO**: World Intellectual Property Organization database

### TRIZ-Patent Datasets:
- **TRIZ-based patent analysis papers**: See Yoon & Park (2004), Trappey et al. (2006)
- **Patent classification datasets**: USPTO CPC (Cooperative Patent Classification)

https://chatgpt.com/share/68769f0b-a990-800c-8c73-62ef6654829e

[The Hidden Challenges of Domain-Adapting LLMs](https://www.arcee.ai/blog/the-hidden-obstacles-of-domain-adaptation-in-llms)
[Classifying the TRIZ Contradiction Problem of the Patents Based on Engineering Parameters](https://link.springer.com/chapter/10.1007/978-3-319-13987-6_32)
[整合TRIZ四十原則與演化趨勢探討產品創新之潛力](https://ir.lib.nycu.edu.tw/handle/11536/77077)

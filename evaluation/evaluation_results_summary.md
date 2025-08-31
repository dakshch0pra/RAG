# RAG System Evaluation Report  

## Answer Quality  
- *BLEU scores (0.07–0.36):* Show that generated answers often diverge in phrasing from the ground truth, reflecting paraphrasing rather than replication.  
- *ROUGE-L F1 (0.17–0.53):* Indicates moderate overlap in structure and content, confirming partial lexical alignment with references.  
- *Token-level Precision/Recall/F1 (0.34–0.59):* Demonstrates that the model consistently uses important terms, even when sentence forms differ.  
- *TF-IDF Cosine Similarity (0.46–0.73):* Strong scores highlight that the answers are semantically coherent with ground truths, despite low surface similarity.  

## Citation Accuracy  
- *Perfect citation precision and recall* on 4 out of 6 questions → clear evidence of strong grounding.  
- *Minor recall drops observed:*  
  - CNNs vs Transformers → recall = 0.8 (missed one relevant source).  
  - Evaluation metrics (ML vs CV) → recall = 0.5 (partial attribution).  
- *No hallucinated citations* → strengthens confidence in source reliability.  

## Strengths  
- ✅ *Citation reliability:* High precision and recall make the system trustworthy in grounding claims.  
- ✅ *Semantic alignment:* TF-IDF similarity demonstrates conceptual accuracy.  
- ✅ *Terminology retention:* Consistent use of domain-specific keywords supports technical correctness.  

## Weaknesses  
- ⚠ *Lexical mismatch:* Low BLEU and ROUGE show limited word-level overlap.  
- ⚠ *Detail recall gaps:* Some answers omit key information, reducing completeness.  
- ⚠ *Citation coverage inconsistencies:* Occasional missing references in otherwise grounded answers.  

## Conclusion  
- The RAG system excels at *providing semantically accurate, well-grounded responses with strong citation reliability*.  
- Its main limitations are in *verbatim correctness and completeness of detail*, which could impact use cases requiring exact phrasing or full coverage.  
- Overall, the system is *highly effective for knowledge support and explanatory tasks, and further improvements in **recall and lexical overlap* would enhance its suitability for academic and professional applications.

"""
Query Encoder Module
====================
Extended query feature extractor + embedding.
"""
import sys
import logging
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config

logger = logging.getLogger(__name__)

class QueryEncoder:
    def __init__(self):
        self.nlp = None
        try:
            import spacy
            self.nlp = spacy.load("en_core_web_sm")
        except Exception as e:
            logger.warning(f"spaCy not available: {e}")
            
        self.encoder = None
        try:
            from sentence_transformers import SentenceTransformer
            self.encoder = SentenceTransformer(config.EMBEDDING_MODEL)
        except Exception as e:
            logger.warning(f"SentenceTransformer not available: {e}")
            
    def encode(self, query: str) -> dict:
        features = {
            "entity_count": 0.0,
            "keyword_density": 0.0,
            "query_length": float(len(query.split())),
            "has_identifiers": 0.0,
            "has_numbers": 0.0,
            "question_type": 0.0,
            "avg_word_frequency": 0.0,
            "abstractness_score": 0.0,
            
            "query_embedding": np.zeros(384) if self.encoder else None,
            "query_complexity_score": 0.0,
            "estimated_retrieval_difficulty": 0.0,
            "has_temporal_reference": 0.0,
            "has_comparison": 0.0,
            "specificity_score": 0.0,
            "ambiguity_score": 0.0,
        }
        
        if self.encoder:
            try:
                features["query_embedding"] = self.encoder.encode(query)
            except Exception as e:
                logger.warning(f"Error encoding query: {e}")
                
        if self.nlp:
            try:
                doc = self.nlp(query)
                features["entity_count"] = float(len(doc.ents))
                nouns_and_propn = sum(1 for token in doc if token.pos_ in ["NOUN", "PROPN"])
                features["keyword_density"] = nouns_and_propn / max(1, len(doc))
                features["query_length"] = float(len(doc))
                features["has_numbers"] = float(any(t.like_num for t in doc))
                features["specificity_score"] = sum(1 for t in doc if t.pos_ == "PROPN") / max(1, len(doc))
            except Exception as e:
                logger.warning(f"Error processing query with spaCy: {e}")
                
        comp_words = ["vs", "compare", "difference", "better"]
        if any(w in query.lower() for w in comp_words):
            features["has_comparison"] = 1.0
            
        features["query_complexity_score"] = min(1.0, (features["query_length"] + features["entity_count"]*2) / 20.0)
        
        return features
        
    def to_feature_vector(self, features: dict) -> np.ndarray:
        keys = [
            "entity_count", "keyword_density", "query_length", "has_identifiers",
            "has_numbers", "question_type", "avg_word_frequency", "abstractness_score",
            "query_complexity_score", "estimated_retrieval_difficulty", "has_temporal_reference",
            "has_comparison", "specificity_score", "ambiguity_score"
        ]
        # Pad to exactly 15 features 
        keys.append("query_complexity_score") 
        vec = [float(features.get(k, 0.0)) for k in keys]
        return np.array(vec)
        
    def to_context_vector(self, features: dict) -> np.ndarray:
        return self.to_feature_vector(features)

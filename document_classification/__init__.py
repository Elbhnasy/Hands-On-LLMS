"""Document classification pipeline using embeddings."""

from .pipeline import EmbeddingClassifier, load_20newsgroups_data, ClassificationResult

__all__ = ['EmbeddingClassifier', 'load_20newsgroups_data', 'ClassificationResult']

"""Document classification pipeline using embeddings."""

import click
from loguru import logger
from typing import List, Dict, Any
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.datasets import fetch_20newsgroups
from sentence_transformers import SentenceTransformer
import json
import pickle
from pathlib import Path
from dataclasses import dataclass
from typing import Optional


@dataclass
class ClassificationResult:
    """Result of a classification prediction."""
    label: str
    confidence: float
    all_probabilities: Dict[str, float]


class EmbeddingClassifier:
    """
    Document classification pipeline using sentence embeddings.
    
    This class provides a complete pipeline for:
    1. Converting text documents to embeddings using SentenceTransformer models
    2. Training a classifier on the embeddings
    3. Making predictions on new documents
    4. Evaluating the model performance
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2", 
                 classifier_type: str = "logistic_regression",
                 random_state: int = 42):
        """
        Initialize the classifier.
        
        Args:
            model_name: Name of the SentenceTransformer model to use
            classifier_type: Type of classifier to use ('logistic_regression' or 'random_forest')
            random_state: Random seed for reproducibility
        """
        self.model_name = model_name
        self.classifier_type = classifier_type
        self.random_state = random_state
        
        # Will be initialized when needed
        self._embedding_model: Optional[SentenceTransformer] = None
        self._classifier: Optional[Any] = None
        self._label_names: Optional[List[str]] = None
        
        logger.info(f"Initialized EmbeddingClassifier with model={model_name}, "
                    f"classifier={classifier_type}")
    
    @property
    def embedding_model(self) -> SentenceTransformer:
        """Lazy-load the embedding model."""
        if self._embedding_model is None:
            logger.info(f"Loading embedding model: {self.model_name}")
            try:
                self._embedding_model = SentenceTransformer(self.model_name)
            except Exception as e:
                logger.error(f"Failed to load embedding model: {e}")
                raise
        return self._embedding_model
    
    def _get_classifier(self):
        """Create a new classifier instance."""
        if self.classifier_type == "logistic_regression":
            return LogisticRegression(max_iter=1000, random_state=self.random_state)
        elif self.classifier_type == "random_forest":
            return RandomForestClassifier(n_estimators=100, random_state=self.random_state)
        else:
            raise ValueError(f"Unsupported classifier type: {self.classifier_type}")
    
    def embed_documents(self, documents: List[str], 
                        batch_size: int = 32) -> np.ndarray:
        """
        Convert text documents to embeddings.
        
        Args:
            documents: List of text documents
            batch_size: Batch size for embedding generation
            
        Returns:
            Array of embeddings with shape (num_documents, embedding_dim)
        """
        logger.info(f"Embedding {len(documents)} documents with batch_size={batch_size}")
        model = self.embedding_model
        embeddings = model.encode(documents, batch_size=batch_size, 
                                  show_progress_bar=True)
        return np.array(embeddings)
    
    def train(self, documents: List[str], labels: List[Any]):
        """
        Train the classifier on document embeddings.
        
        Args:
            documents: List of text documents
            labels: List of labels corresponding to the documents
        """
        # Generate embeddings
        logger.info("Training: Generating document embeddings...")
        embeddings = self.embed_documents(documents)
        
        # Store label names for inference
        unique_labels = sorted(set(labels))
        self._label_names = [str(label) for label in unique_labels]
        
        # Train the classifier
        logger.info("Training classifier...")
        self._classifier = self._get_classifier()
        self._classifier.fit(embeddings, labels)
        
        logger.info(f"Training complete. Labels: {self._label_names}")
    
    def predict(self, documents: List[str]) -> List[ClassificationResult]:
        """
        Predict labels for new documents.
        
        Args:
            documents: List of text documents to classify
            
        Returns:
            List of ClassificationResult containing the predicted label and confidence
        """
        if self._classifier is None:
            raise ValueError("Classifier not trained. Call train() first.")
        
        # Generate embeddings
        logger.info(f"Predicting labels for {len(documents)} documents...")
        embeddings = self.embed_documents(documents)
        
        # Get predictions
        predictions = self._classifier.predict(embeddings)
        probabilities = self._classifier.predict_proba(embeddings)
        
        # Create results
        results = []
        for pred_idx, (pred, probs) in enumerate(zip(predictions, probabilities)):
            # Find the confidence for the predicted label
            if hasattr(self._classifier, 'classes_'):
                pred_class_idx = list(self._classifier.classes_).index(pred)
                confidence = probs[pred_class_idx]
            else:
                confidence = max(probs)
            
            # Create probability dict
            prob_dict = {}
            if self._label_names:
                for class_name, prob in zip(self._classifier.classes_, probs):
                    prob_dict[str(class_name)] = float(prob)
            
            results.append(ClassificationResult(
                label=str(pred),
                confidence=float(confidence),
                all_probabilities=prob_dict
            ))
        
        return results
    
    def evaluate(self, documents: List[str], 
                 true_labels: List[Any]) -> Dict[str, Any]:
        """
        Evaluate the classifier on test data.
        
        Args:
            documents: List of test documents
            true_labels: True labels for the test documents
            
        Returns:
            Dictionary with evaluation metrics
        """
        logger.info("Evaluating model performance...")
        results = self.predict(documents)
        predicted_labels = [r.label for r in results]
        
        accuracy = accuracy_score(true_labels, predicted_labels)
        report = classification_report(true_labels, predicted_labels, 
                                       output_dict=True)
        
        return {
            "accuracy": accuracy,
            "classification_report": report,
            "num_samples": len(documents)
        }
    
    def save(self, path: str):
        """
        Save the trained model to disk.
        
        Args:
            path: Directory path to save the model
        """
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        
        # Save classifier
        classifier_path = path / "classifier.pkl"
        with open(classifier_path, "wb") as f:
            pickle.dump(self._classifier, f)
        
        # Save config
        config = {
            "model_name": self.model_name,
            "classifier_type": self.classifier_type,
            "label_names": self._label_names,
        }
        config_path = path / "config.json"
        with open(config_path, "w") as f:
            json.dump(config, f, indent=2)
        
        logger.info(f"Model saved to {path}")
    
    def load(self, path: str):
        """
        Load a trained model from disk.
        
        Args:
            path: Directory path to load the model from
        """
        path = Path(path)
        
        # Load classifier
        classifier_path = path / "classifier.pkl"
        with open(classifier_path, "rb") as f:
            self._classifier = pickle.load(f)
        
        # Load config
        config_path = path / "config.json"
        with open(config_path, "r") as f:
            config = json.load(f)
        
        self.model_name = config["model_name"]
        self.classifier_type = config["classifier_type"]
        self._label_names = config["label_names"]
        
        logger.info(f"Model loaded from {path}")


def load_20newsgroups_data(categories: List[str] = None, 
                           subset: str = "train") -> Dict[str, Any]:
    """
    Load the 20 newsgroups dataset.
    
    Args:
        categories: List of categories to load (None for all)
        subset: 'train' or 'test'
        
    Returns:
        Dictionary with 'documents' and 'labels' keys
    """
    logger.info(f"Loading 20 newsgroups dataset (subset={subset})...")
    
    data = fetch_20newsgroups(subset=subset, categories=categories,
                              shuffle=True, random_state=42,
                              remove=('headers', 'footers', 'quotes'))
    
    return {
        "documents": data.data,
        "labels": data.target.tolist(),
        "target_names": data.target_names,
        "categories": categories
    }


@click.group()
def cli():
    """Document classification pipeline using embeddings."""
    pass


@cli.command()
@click.option('--categories', type=str, default='comp.graphics,sci.space,talk.politics.guns',
              help='Comma-separated list of categories to classify')
@click.option('--model', type=str, default='all-MiniLM-L6-v2',
              help='SentenceTransformer model name')
@click.option('--classifier', type=str, default='logistic_regression',
              type=click.Choice(['logistic_regression', 'random_forest']),
              help='Classifier type')
@click.option('--output', type=str, default='models/document_classifier',
              help='Output directory for the trained model')
@click.option('--num-samples', type=int, default=None,
              help='Number of samples to use (for testing, None for all)')
def train(categories: str, model: str, classifier: str, 
          output: str, num_samples: Optional[int]):
    """Train a document classifier."""
    # Parse categories
    category_list = [c.strip() for c in categories.split(',')]
    
    # Load data
    logger.info("Loading training data...")
    data = load_20newsgroups_data(categories=category_list, subset="train")
    
    documents = data["documents"]
    labels = data["labels"]
    
    if num_samples:
        documents = documents[:num_samples]
        labels = labels[:num_samples]
    
    logger.info(f"Training on {len(documents)} documents from categories: {category_list}")
    
    # Initialize and train classifier
    clf = EmbeddingClassifier(model_name=model, classifier_type=classifier)
    clf.train(documents, labels)
    
    # Save model
    clf.save(output)
    logger.info(f"Model saved to {output}")


@cli.command()
@click.option('--model-path', type=str, default='models/document_classifier',
              help='Path to the trained model')
@click.option('--categories', type=str, default='comp.graphics,sci.space,talk.politics.guns',
              help='Comma-separated list of categories')
@click.option('--num-samples', type=int, default=None,
              help='Number of samples to evaluate (None for all)')
def evaluate(model_path: str, categories: str, num_samples: Optional[int]):
    """Evaluate a trained document classifier."""
    # Parse categories
    category_list = [c.strip() for c in categories.split(',')]
    
    # Load model
    logger.info(f"Loading model from {model_path}...")
    clf = EmbeddingClassifier()
    clf.load(model_path)
    
    # Load test data
    logger.info("Loading test data...")
    data = load_20newsgroups_data(categories=category_list, subset="test")
    
    documents = data["documents"]
    labels = data["labels"]
    
    if num_samples:
        documents = documents[:num_samples]
        labels = labels[:num_samples]
    
    # Evaluate
    results = clf.evaluate(documents, labels)
    
    logger.info(f"Evaluation results:")
    logger.info(f"  Accuracy: {results['accuracy']:.4f}")
    logger.info(f"  Number of samples: {results['num_samples']}")
    
    # Print detailed report
    print("\nClassification Report:")
    print(classification_report(labels, 
                                [r.label for r in clf.predict(documents[:100])],
                                target_names=data["target_names"]))


@cli.command()
@click.option('--model-path', type=str, default='models/document_classifier',
              help='Path to the trained model')
@click.option('--text', type=str, required=True,
              help='Text to classify')
def predict(model_path: str, text: str):
    """Classify a single document."""
    # Load model
    logger.info(f"Loading model from {model_path}...")
    clf = EmbeddingClassifier()
    clf.load(model_path)
    
    # Predict
    results = clf.predict([text])
    
    result = results[0]
    print(f"\nPredicted label: {result.label}")
    print(f"Confidence: {result.confidence:.4f}")
    print("\nAll probabilities:")
    for label, prob in sorted(result.all_probabilities.items(), key=lambda x: x[1], reverse=True):
        print(f"  {label}: {prob:.4f}")


@cli.command()
def demo():
    """Run a complete demo: train, evaluate, and predict."""
    # Use a smaller subset for the demo
    categories = ['comp.graphics', 'sci.space', 'talk.politics.guns']
    
    logger.info("=" * 50)
    logger.info("Document Classification Pipeline Demo")
    logger.info("=" * 50)
    
    # 1. Load data
    logger.info("\n1. Loading training data...")
    train_data = load_20newsgroups_data(categories=categories, subset="train")
    
    # Use a subset for faster demo
    train_docs = train_data["documents"][:100]
    train_labels = train_data["labels"][:100]
    
    logger.info(f"   Loaded {len(train_docs)} training documents")
    
    # 2. Train
    logger.info("\n2. Training classifier...")
    clf = EmbeddingClassifier(model_name="all-MiniLM-L6-v2", 
                                classifier_type="logistic_regression")
    clf.train(train_docs, train_labels)
    
    # 3. Evaluate
    logger.info("\n3. Evaluating on test data...")
    test_data = load_20newsgroups_data(categories=categories, subset="test")
    test_docs = test_data["documents"][:50]
    test_labels = test_data["labels"][:50]
    
    results = clf.evaluate(test_docs, test_labels)
    logger.info(f"   Accuracy: {results['accuracy']:.4f}")
    
    # 4. Predict on example
    logger.info("\n4. Making predictions on sample documents...")
    sample_texts = [
        "The OpenGL graphics library provides a powerful API for rendering 2D and 3D graphics.",
        "The Hubble Space Telescope has revolutionized our understanding of the universe.",
        "The Second Amendment guarantees the right to keep and bear arms."
    ]
    
    for text in sample_texts:
        result = clf.predict([text])[0]
        print(f"\n   Text: {text[:80]}...")
        print(f"   Predicted: {result.label} (confidence: {result.confidence:.4f})")
    
    logger.info("\n" + "=" * 50)
    logger.info("Demo complete!")
    logger.info("=" * 50)


if __name__ == "__main__":
    cli()

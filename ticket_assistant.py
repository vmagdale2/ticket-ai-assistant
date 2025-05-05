# Ticket Categorization and Priority AI Assistant
# This project demonstrates a machine learning-based system for automatically
# classifying support tickets and assigning priority levels.

import pandas as pd
import numpy as np
import re
import pickle
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
from sklearn.pipeline import Pipeline

# Download necessary NLTK resources
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')


class TicketProcessor:
    """
    Class for preprocessing ticket text data
    """

    def __init__(self):
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))

    def clean_text(self, text):
        """Clean and normalize text data"""
        if not isinstance(text, str):
            return ""

        # Convert to lowercase
        text = text.lower()

        # Remove special characters and numbers
        text = re.sub(r'[^a-zA-Z\s]', '', text)

        # Tokenize
        tokens = word_tokenize(text)

        # Remove stop words and lemmatize
        cleaned_tokens = [
            self.lemmatizer.lemmatize(word) for word in tokens
            if word not in self.stop_words and len(word) > 2
        ]

        return " ".join(cleaned_tokens)


class TicketCategorizer:
    """
    Model for categorizing support tickets into predefined categories
    """

    def __init__(self):
        self.categories = [
            "Technical Issue",
            "Account Access",
            "Billing Question",
            "Feature Request",
            "Product Information",
            "General Inquiry"
        ]
        self.vectorizer = TfidfVectorizer(max_features=5000)
        self.model = OneVsRestClassifier(LogisticRegression(max_iter=1000))
        self.processor = TicketProcessor()
        self.pipeline = None

    def train(self, X, y):
        """Train the categorization model"""
        # Preprocess the text data
        X_cleaned = [self.processor.clean_text(text) for text in X]

        # Create and fit the pipeline
        self.pipeline = Pipeline([
            ('vectorizer', self.vectorizer),
            ('classifier', self.model)
        ])

        self.pipeline.fit(X_cleaned, y)
        return self

    def predict(self, text):
        """Predict category for a new ticket"""
        if not self.pipeline:
            raise ValueError("Model has not been trained yet")

        # Preprocess the input text
        cleaned_text = self.processor.clean_text(text)

        # Make prediction
        prediction = self.pipeline.predict([cleaned_text])[0]
        return prediction

    def save_model(self, filepath):
        """Save the trained model to a file"""
        with open(filepath, 'wb') as f:
            pickle.dump(self.pipeline, f)

    def load_model(self, filepath):
        """Load a trained model from a file"""
        with open(filepath, 'rb') as f:
            self.pipeline = pickle.load(f)
        return self


class PriorityAssigner:
    """
    Model for determining the priority level of a ticket
    """

    def __init__(self):
        self.priority_levels = ["Low", "Medium", "High", "Critical"]
        self.vectorizer = TfidfVectorizer(max_features=5000)
        self.model = LogisticRegression(max_iter=1000)
        self.processor = TicketProcessor()
        self.pipeline = None

        # Keywords that might indicate higher priority
        self.urgent_keywords = [
            "urgent", "critical", "emergency", "broken", "error",
            "crash", "down", "not working", "failed", "immediate"
        ]

    def train(self, X, y):
        """Train the priority prediction model"""
        # Preprocess the text data
        X_cleaned = [self.processor.clean_text(text) for text in X]

        # Create and fit the pipeline
        self.pipeline = Pipeline([
            ('vectorizer', self.vectorizer),
            ('classifier', self.model)
        ])

        self.pipeline.fit(X_cleaned, y)
        return self

    def predict(self, text, category=None):
        """Predict priority for a new ticket"""
        if not self.pipeline:
            raise ValueError("Model has not been trained yet")

        # Preprocess the input text
        cleaned_text = self.processor.clean_text(text)

        # Check for urgent keywords to potentially override model prediction
        text_lower = text.lower()
        urgent_count = sum(1 for keyword in self.urgent_keywords if keyword in text_lower)

        # Make base prediction
        prediction = self.pipeline.predict([cleaned_text])[0]

        # Adjust prediction based on urgency keywords and category
        if urgent_count >= 3 and prediction != "Critical":
            return "High"

        # Certain categories might inherently have higher priority
        if category == "Technical Issue" and prediction == "Low":
            return "Medium"

        return prediction

    def save_model(self, filepath):
        """Save the trained model to a file"""
        with open(filepath, 'wb') as f:
            pickle.dump(self.pipeline, f)

    def load_model(self, filepath):
        """Load a trained model from a file"""
        with open(filepath, 'rb') as f:
            self.pipeline = pickle.load(f)
        return self


class TicketAssistant:
    """
    Main class that combines ticket categorization and priority assignment
    """

    def __init__(self):
        self.categorizer = TicketCategorizer()
        self.priority_assigner = PriorityAssigner()

    def load_models(self, categorizer_path, priority_path):
        """Load trained models"""
        self.categorizer.load_model(categorizer_path)
        self.priority_assigner.load_model(priority_path)
        return self

    def process_ticket(self, ticket_text, additional_info=None):
        """Process a new ticket and return category and priority"""
        category = self.categorizer.predict(ticket_text)
        priority = self.priority_assigner.predict(ticket_text, category)

        return {
            "text": ticket_text,
            "category": category,
            "priority": priority,
            "additional_info": additional_info
        }


# Example usage
if __name__ == "__main__":
    # Sample training data (in a real scenario, you would load this from a file)
    sample_tickets = [
        "I cannot log into my account after resetting my password",
        "The application crashes whenever I try to upload a file",
        "How do I change my subscription plan?",
        "I would like to suggest adding a dark mode to the app",
        "What are the system requirements for running this software?",
        "My monthly bill seems incorrect, I was charged twice"
    ]

    sample_categories = [
        "Account Access",
        "Technical Issue",
        "Billing Question",
        "Feature Request",
        "General Inquiry",
        "Billing Question"
    ]

    sample_priorities = [
        "Medium",
        "High",
        "Medium",
        "Low",
        "Low",
        "Medium"
    ]

    # Train models
    categorizer = TicketCategorizer()
    categorizer.train(sample_tickets, sample_categories)

    priority_assigner = PriorityAssigner()
    priority_assigner.train(sample_tickets, sample_priorities)

    # Save models
    categorizer.save_model("categorizer_model.pkl")
    priority_assigner.save_model("priority_model.pkl")

    # Create assistant and load models
    assistant = TicketAssistant()
    assistant.load_models("categorizer_model.pkl", "priority_model.pkl")

    # Test with a new ticket
    new_ticket = "The website is completely down and customers cannot make purchases"
    result = assistant.process_ticket(new_ticket)

    print(f"Ticket: {result['text']}")
    print(f"Category: {result['category']}")
    print(f"Priority: {result['priority']}")
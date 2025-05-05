# ticket-ai-as# Ticket Categorization and Priority AI Assistant

A machine learning-based system for automatically classifying support tickets and assigning priority levels to improve support team efficiency.

## Project Overview

This portfolio project demonstrates my skills in:
- Natural Language Processing (NLP)
- Machine Learning Classification
- Text Processing and Feature Engineering
- Web Application Development
- Data Visualization

The Ticket AI Assistant helps support teams by:
1. Automatically categorizing incoming tickets into predefined categories
2. Assigning appropriate priority levels based on content analysis
3. Providing a clean interface for monitoring and managing tickets
4. Displaying analytics on ticket volume, categories, and AI performance

## Features

### Core ML Functionality
- **Text Preprocessing**: Cleans and normalizes ticket text using NLTK for tokenization, stop word removal, and lemmatization
- **Category Classification**: Uses TF-IDF vectorization and a multiclass classifier to categorize tickets into 6 predefined categories
- **Priority Assignment**: Analyzes text content and category to determine ticket urgency (Low, Medium, High, Critical)
- **Keyword Detection**: Recognizes urgent keywords that may indicate higher priority issues

### Web Application
- **Clean, Responsive UI**: Built with Flask, Bootstrap, and JavaScript
- **Real-time Processing**: Processes tickets asynchronously using AJAX
- **Ticket History**: View and filter previously processed tickets
- **Advanced Analytics**: Visualize ticket metrics with interactive charts

## Tech Stack

- **Backend**: Python, Flask
- **ML/NLP**: scikit-learn, NLTK
- **Frontend**: HTML, CSS, JavaScript, Bootstrap 5
- **Data Visualization**: Chart.js
- **Data Handling**: pandas, NumPy

## Installation and Setup

### Prerequisites
- Python 3.8+
- pip

### Setup Instructions

1. Clone the repository
```bash
git clone https://github.com/yourusername/ticket-ai-assistant.git
cd ticket-ai-assistant
```

2. Create and activate a virtual environment
```bash
python -m venv venv
source venv/bin/activate  # On Windows, use: venv\Scripts\activate
```

3. Install dependencies
```bash
pip install -r requirements.txt
```

4. Run the application
```bash
python app.py
```

5. Open your browser and navigate to http://localhost:5000

## Project Structure

```
ticket-ai-assistant/
├── app.py                  # Flask application
├── ticket_assistant.py     # Core ML modules
├── requirements.txt        # Project dependencies
├── static/                 # Static assets
│   ├── css/                # Stylesheets
│   ├── js/                 # JavaScript files
│   └── img/                # Images
├── templates/              # HTML templates
│   ├── index.html          # Home page
│   ├── history.html        # Ticket history page
│   └── analytics.html      # Analytics dashboard
├── models/                 # Saved ML models
│   ├── categorizer_model.pkl
│   └── priority_model.pkl
└── README.md               # Project documentation
```

## How It Works

### Text Processing Pipeline

1. **Text Cleaning**: Remove special characters, convert to lowercase
2. **Tokenization**: Split text into individual words
3. **Stop Word Removal**: Remove common words that don't add meaning
4. **Lemmatization**: Convert words to their base form

### Classification Process

1. **Feature Extraction**: Convert text to numerical features using TF-IDF
2. **Category Prediction**: Use a multi-class classifier to assign the most relevant category
3. **Priority Analysis**: Consider text content, urgent keywords, and category to determine priority
4. **Response Generation**: Return predictions with confidence scores

## License

This project is licensed under the MIT License - see the LICENSE file for details.sistant
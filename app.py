# app.py - Flask application for the Ticket Assistant
# This provides a simple web interface to demonstrate the ticket categorization and prioritization

from flask import Flask, render_template, request, jsonify
import os
import pandas as pd
from ticket_assistant import TicketAssistant, TicketCategorizer, PriorityAssigner

app = Flask(__name__)


# Initialize and train the models if they don't exist
def initialize_models():
    if not (os.path.exists("categorizer_model.pkl") and os.path.exists("priority_model.pkl")):
        # Load sample training data (in a real app, this would come from a database)
        # For demo purposes, we'll create a more extensive sample dataset
        sample_data = {
            "text": [
                "I cannot log into my account after resetting my password",
                "The application crashes whenever I try to upload a file",
                "How do I change my subscription plan?",
                "I would like to suggest adding a dark mode to the app",
                "What are the system requirements for running this software?",
                "My monthly bill seems incorrect, I was charged twice",
                "The system is giving me an error code 404 when I try to access my files",
                "I need to upgrade my storage plan",
                "How do I export my data from the platform?",
                "The mobile app is incredibly slow on my device",
                "I found a security vulnerability in your login page",
                "Can I get a refund for my recent purchase?",
                "The search function isn't returning relevant results",
                "I need to delete my account permanently",
                "Your service has been down for three hours now",
                "How do I connect my account to other applications?",
                "The confirmation emails aren't being delivered to my inbox",
                "Can I get documentation for the API?",
                "The graphs on the dashboard aren't displaying correctly",
                "I'm interested in enterprise pricing for our company"
            ],
            "category": [
                "Account Access",
                "Technical Issue",
                "Billing Question",
                "Feature Request",
                "General Inquiry",
                "Billing Question",
                "Technical Issue",
                "Account Access",
                "General Inquiry",
                "Technical Issue",
                "Technical Issue",
                "Billing Question",
                "Technical Issue",
                "Account Access",
                "Technical Issue",
                "General Inquiry",
                "Technical Issue",
                "Product Information",
                "Technical Issue",
                "Billing Question"
            ],
            "priority": [
                "Medium",
                "High",
                "Medium",
                "Low",
                "Low",
                "Medium",
                "Medium",
                "Low",
                "Low",
                "Medium",
                "Critical",
                "Medium",
                "Medium",
                "Medium",
                "High",
                "Low",
                "Medium",
                "Low",
                "Medium",
                "Medium"
            ]
        }

        df = pd.DataFrame(sample_data)

        # Train categorizer
        categorizer = TicketCategorizer()
        categorizer.train(df["text"], df["category"])
        categorizer.save_model("categorizer_model.pkl")

        # Train priority assigner
        priority_assigner = PriorityAssigner()
        priority_assigner.train(df["text"], df["priority"])
        priority_assigner.save_model("priority_model.pkl")

    # Create assistant and load models
    assistant = TicketAssistant()
    assistant.load_models("categorizer_model.pkl", "priority_model.pkl")

    return assistant


# Create our ticket assistant
ticket_assistant = initialize_models()


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/process_ticket', methods=['POST'])
def process_ticket():
    data = request.json
    ticket_text = data.get('text', '')
    customer_info = data.get('customer_info', '')

    # Process the ticket
    result = ticket_assistant.process_ticket(ticket_text, additional_info=customer_info)

    return jsonify(result)


@app.route('/history')
def history():
    # In a real application, this would fetch from a database
    # For demo purposes, we'll just show a static example
    return render_template('history.html')


@app.route('/analytics')
def analytics():
    # In a real application, this would calculate real analytics
    # For demo purposes, we'll just show a static example
    return render_template('analytics.html')


if __name__ == '__main__':
    app.run(debug=True)
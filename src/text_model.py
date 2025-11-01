
"""
ClinicalBERT Diabetes Text Classifier - Inference Script
========================================================

This script provides easy-to-use functions for predicting diabetes relevance
from medical text using the trained ClinicalBERT model.

Usage:
    from text_model import DiabetesTextClassifier
    
    classifier = DiabetesTextClassifier('models/clinical_bert_diabetes_classifier.pth')
    
    text = "Patient presents with hyperglycemia and requires insulin therapy..."
    prediction = classifier.predict(text)
    print(f"Prediction: {prediction['label']} (confidence: {prediction['confidence']:.3f})")
"""

import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel
import re
import numpy as np
import json

class ClinicalBERTClassifier(nn.Module):
    def __init__(self, model_name, num_classes=2, dropout_rate=0.3):
        super(ClinicalBERTClassifier, self).__init__()
        
        self.bert = AutoModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(dropout_rate)
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_classes)
        
    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls_output = outputs.last_hidden_state[:, 0, :]
        cls_output = self.dropout(cls_output)
        logits = self.classifier(cls_output)
        return logits

class DiabetesTextClassifier:
    def __init__(self, model_path, device=None):
        """
        Initialize the diabetes text classifier
        
        Args:
            model_path (str): Path to the saved model file
            device (str): Device to run inference on ('cuda' or 'cpu')
        """
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model_path = model_path
        
        # Load model checkpoint
        checkpoint = torch.load(model_path, map_location=self.device)
        
        # Extract configuration
        self.config = checkpoint['model_config']
        self.class_names = checkpoint['class_names']
        self.max_length = self.config['max_length']
        
        # Initialize tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(checkpoint['tokenizer_name'])
        
        # Initialize and load model
        self.model = ClinicalBERTClassifier(
            self.config['model_name'],
            self.config['num_classes'],
            self.config['dropout_rate']
        )
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)
        self.model.eval()
        
        print(f"✅ Model loaded successfully on {self.device}")
        print(f"   Classes: {self.class_names}")
    
    def preprocess_text(self, text):
        """Preprocess medical text"""
        if not text:
            return ""
        
        text = str(text)
        
        # Remove de-identified placeholders
        text = re.sub(r'\[\*\*[^]]*\*\*\]', '', text)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Preserve medical abbreviations
        medical_abbrevs = [
            'HTN', 'DM', 'T2DM', 'T1DM', 'CAD', 'CHF', 'COPD', 'CKD', 'CVD',
            'MI', 'PE', 'DVT', 'UTI', 'ICU', 'ER', 'OR', 'IV', 'PO', 'NPO',
            'BID', 'TID', 'QID', 'PRN', 'STAT', 'HbA1c', 'BMI', 'BP', 'HR',
            'RR', 'O2', 'CO2', 'EKG', 'ECG', 'CBC', 'BUN', 'GFR', 'ALT', 'AST'
        ]
        
        # Temporarily replace abbreviations
        abbrev_map = {}
        for i, abbrev in enumerate(medical_abbrevs):
            if abbrev in text:
                placeholder = f"__ABBREV_{i}__"
                abbrev_map[placeholder] = abbrev
                text = text.replace(abbrev, placeholder)
        
        # Convert to lowercase
        text = text.lower()
        
        # Restore abbreviations
        for placeholder, abbrev in abbrev_map.items():
            text = text.replace(placeholder, abbrev)
        
        # Clean punctuation
        text = re.sub(r'[^\w\s.,()-]', ' ', text)
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def predict(self, text, return_probabilities=False):
        """
        Predict diabetes relevance for input text
        
        Args:
            text (str): Input medical text
            return_probabilities (bool): Whether to return class probabilities
            
        Returns:
            dict: Prediction results with label, confidence, and optionally probabilities
        """
        # Preprocess text
        processed_text = self.preprocess_text(text)
        
        if not processed_text:
            return {
                'label': 'Unknown',
                'confidence': 0.0,
                'probabilities': [0.5, 0.5] if return_probabilities else None,
                'error': 'Empty or invalid text'
            }
        
        # Tokenize
        encoding = self.tokenizer(
            processed_text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        # Move to device
        input_ids = encoding['input_ids'].to(self.device)
        attention_mask = encoding['attention_mask'].to(self.device)
        
        # Predict
        with torch.no_grad():
            outputs = self.model(input_ids, attention_mask)
            probabilities = torch.softmax(outputs, dim=1)
            predicted_class = torch.argmax(probabilities, dim=1).item()
            confidence = probabilities[0, predicted_class].item()
        
        result = {
            'label': self.class_names[predicted_class],
            'confidence': confidence,
            'predicted_class': predicted_class
        }
        
        if return_probabilities:
            result['probabilities'] = probabilities[0].cpu().numpy().tolist()
        
        return result
    
    def predict_batch(self, texts, batch_size=8):
        """
        Predict for multiple texts
        
        Args:
            texts (list): List of input texts
            batch_size (int): Batch size for processing
            
        Returns:
            list: List of prediction results
        """
        results = []
        
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i+batch_size]
            batch_results = []
            
            for text in batch_texts:
                result = self.predict(text, return_probabilities=True)
                batch_results.append(result)
            
            results.extend(batch_results)
        
        return results

# Example usage and testing
def main():
    """Example usage of the classifier"""
    
    # Sample medical texts for testing
    test_texts = [
        "Patient presents with elevated HbA1c of 9.2% and requires insulin therapy adjustment. Blood glucose monitoring shows persistent hyperglycemia despite metformin treatment.",
        "This systematic review examines the prevalence of cardiovascular disease in different populations across multiple cohort studies.",
        "Clinical trial demonstrates efficacy of GLP-1 agonists in reducing HbA1c levels by 1.2% compared to placebo in patients with T2DM.",
        "Epidemiological analysis of risk factors associated with metabolic syndrome in the general population."
    ]
    
    # Initialize classifier
    try:
        classifier = DiabetesTextClassifier('models/clinical_bert_diabetes_classifier.pth')
        
        print("\n🧪 Testing classifier with sample texts:")
        print("="*60)
        
        for i, text in enumerate(test_texts, 1):
            result = classifier.predict(text, return_probabilities=True)
            
            print(f"\nSample {i}:")
            print(f"Text: {text[:100]}...")
            print(f"Prediction: {result['label']}")
            print(f"Confidence: {result['confidence']:.3f}")
            print(f"Probabilities: {[f'{p:.3f}' for p in result['probabilities']]}")
        
        print("\n✅ Classifier testing complete!")
        
    except FileNotFoundError:
        print("❌ Model file not found. Please train the model first.")
    except Exception as e:
        print(f"❌ Error loading classifier: {e}")

if __name__ == "__main__":
    main()

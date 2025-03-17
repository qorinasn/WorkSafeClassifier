# WorkSafeClassifier: Workplace Accident Report Classification Web App
WorkSafeClassifier is an interactive web application built with Streamlit that classifies workplace accident reports into categories defined by the International Labour Organization (ILO). This application brings the functionality of our Jupyter notebook classification system to a user-friendly interface.

## Key Features

- **Interactive Classification**: Upload or paste accident reports for instant classification
- **Detailed Insights**: View preprocessing steps and model calculations
- **Multiple Pages**: Home page for classification and About page with project details
- **Visual Explanations**: See how the model processes text and makes predictions
- **User-Friendly Interface**: Designed for both technical and non-technical users

## Technology Stack

- Python
- Streamlit for web interface
- pandas and numpy for data processing
- scikit-learn for machine learning
- NLTK and Sastrawi for text processing
- Jupyter Notebook for development

## Project Structure

```
WorkSafeClassifier/
├── Web/                   # Web application components
│   ├── app.py                 # Main Streamlit application file
│   ├── home.py                # Home page functionality
│   ├── tentang.py             # About page content
│   ├── utils.py               # Utility functions for text processing
│   ├── requirements.txt       # Project dependencies
├── Model/                 # Folder containing model files
│   ├── skripsiqorina.ipynb         # Main Jupyter notebook implementing the classification system
│   ├── custom_stemming.py          # Custom stemming implementation
│   ├── datasetfix.py               # Dataset processing and fixing
│   ├── hapus.py                    # Data cleaning utilities
│   ├── mnb.py                      # Multinomial Naive Bayes implementation
│   ├── preprocess.py               # Text preprocessing functions
│   ├── print_version.py            # Version information
│   ├── tfidf.py                    # TF-IDF vectorization utilities
│   ├── data/                       # Raw and processed datasets
│   ├── intermediate_csv_files/     # Intermediate CSV files generated during processing
│   ├── pickle_files/               # Saved models, vectorizers, and processed data
└── __pycache__/                # Python cache files
```

## Getting Started

### Prerequisites
- Python 3.7+
- Streamlit
- pandas
- numpy
- scikit-learn
- nltk
- matplotlib
- seaborn
- Sastrawi

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/worksafeclassifier.git
   ```

2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Run the Streamlit application:
   ```bash
   streamlit run app.py
   ```

## Usage

1. **Home Page**: 
   - Enter or paste the text of a workplace accident report
   - Click "Predict" to see the classification result
   - View detailed processing steps and model calculations

2. **About Page**:
   - Learn about the project background
   - Understand the classification methodology
   - See details about the dataset and model performance

## Development Process

This application was developed following these key steps:

1. **Data Collection and Preparation**: Gathered workplace accident reports and categorized them according to ILO standards
2. **Text Preprocessing**: Implemented cleaning, case folding, tokenization, stopword removal, and stemming
3. **Feature Extraction**: Converted text to numerical features using TF-IDF vectorization
4. **Model Development**: Implemented Multinomial Naive Bayes classifier using both scikit-learn and custom code
5. **Evaluation**: Assessed model performance using accuracy, precision, recall, and F1-score
6. **Web Integration**: Built a user-friendly Streamlit interface to make the classification system accessible

## License

This project is licensed under the Apache 2.0 License - see the [LICENSE](LICENSE) file for details

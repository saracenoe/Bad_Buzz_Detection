# Bad_Buzz_Detection with deep learning

## Context
We want to be able to detect quickly "bad buzz" on social networks, to be able to anticipate and address issues as fast as possible. 

Goal: Create an AI RestAPI that can detect "bad buzz" and predict the reason for it.

### Project modules
We will use the Python programming language, and present here the code and results in this Notebook JupyterLab file.

We will use the usual libraries for data exploration, modeling and visualisation :

- NumPy and Pandas : for maths (stats, algebra, ...) and large data manipulation
- Plotly : for interactive data visualization

We will also use libraries specific to the goals of this project :

- NLP Natural Language Processing
- NLTK and Spacy : for text processing

# Article
Blog Article talking about Word Embedding Techniques

# Presentation
Slides showing the workflow of the project. 

## Installation
  ### Prerequisites
  Python 3.9
  Azure subscription
  Azure ML Studio
  
  ### Virtual environment
      
      conda create -n badbuzz python=3.8 -y
      conda activate badbuzz
      
  ### Dependencies    
      pip install -r requirements.txt
        
## Usage
  ### Data
  Download and extract the files from [Kaggle](https://www.kaggle.com/datasets/kazanova/sentiment140?select=training.1600000.processed.noemoticon.csv).

  ### Exploratory Data Analysis
  The notebook named o_notebooks presents the results of the EDA
      
  ### Sentiment analysis
  "Off-the-shelf API" approach using the API of the cognitive service offered by Microsoft Azure for sentiment analysis. Notebook named 1_azure...
      
  ### Baseline model
  Logistic Regression model to predict tweets sentiment will be the baseline against which we will compare all other (more advanced) models.
      
  ### Word embedding
   we use the Glove, word2vec and fasttext LSTM model to compute Word Embedding on our tweets dataset, before training a classification model on the lower-dimension vector space.
  
  ### BERT
  Setting up a pre-trained BERT modedl for fine-tuning
  
  ### AzureML
  We use the AzureML Studio's Designer to design our data processing pipeline.
  
  ## Run flask app
    python myapp_LSTM.py
   
      
 ## Author
 
 **Ezequiel Saraceno**
 
 ## Show your support

Give a ⭐️ if this project helped you!
 

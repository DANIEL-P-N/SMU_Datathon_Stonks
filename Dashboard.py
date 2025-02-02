import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import NMF
from sklearn.pipeline import Pipeline
import topicwizard
import spacy

# Load spaCy's language model
nlp = spacy.load("en_core_web_sm")  # Replace with your custom model if needed

# Load the dataset
# df2 = pd.read_excel('news_excerpts_parsed.xlsx')
df2 = pd.read_excel('wikileaks_parsed.xlsx')

# Remove unnecessary columns
news = df2.copy()
news.drop(columns='PDF Path', inplace=True)

# Extract entities as full phrases using spaCy
def extract_entity_phrases(text):
    doc = nlp(text)
    # Extract entities of types "NORP", "LOC", and "GPE"
    entity_phrases = [ent.text for ent in doc.ents]# if ent.label_ in {"NORP", "LOC", "GPE"}]
    return " | ".join(entity_phrases)  # Use "|" as a delimiter to separate entities

# Apply entity extraction to the 'Text' column
news['Processed_Text'] = news['Text'].dropna().apply(extract_entity_phrases)

# Check the processed text output
print("Sample processed text:")
print(news['Processed_Text'].head())

# Step 1: Define a function to create a topic pipeline
def make_topic_pipeline(vectorizer, model):
    return Pipeline([
        ('vectorizer', vectorizer),  # Step to vectorize the text
        ('model', model)             # Step to apply the topic model
    ])

# Step 2: Preprocess the data and set up the pipeline
vectorizer = CountVectorizer(
    token_pattern=r"[^|]+",  # Treat phrases separated by "|" as tokens
    min_df=2,                # Lowered to include less common phrases
    max_df=0.9,              # Increased to include more frequent phrases
    stop_words=None          # No predefined stop words since entities are meaningful
)

# Dynamically adjust n_components based on token count
news_articles = news['Processed_Text'].dropna()  # Remove any null values
vectorizer.fit(news_articles)
num_tokens = len(vectorizer.get_feature_names_out())
n_topics = min(10, num_tokens)  # Ensure n_topics does not exceed the number of tokens

print(f"Number of unique tokens: {num_tokens}")
print(f"Number of topics set to: {n_topics}")

model = NMF(n_components=n_topics, random_state=42)

# Create the topic modeling pipeline
topic_pipeline = make_topic_pipeline(vectorizer, model)

# Step 3: Fit the pipeline to the processed entity phrases
topic_pipeline.fit(news_articles)

# Step 4: Transform the data (optional if you need the transformed matrix)
X_train = topic_pipeline.named_steps['vectorizer'].transform(news_articles)

# Step 5: Visualize the entity phrases using Topic Wizard
news_articles = news['Processed_Text'].dropna()
topic_pipeline.fit(news_articles)
topicwizard.visualize(news['Processed_Text'], model=topic_pipeline)

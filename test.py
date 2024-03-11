import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib

nb_classifier = joblib.load("naive_bayes_classifier.h5")

# Read the JSON file (insert your dataset in AI_PROJECT FOLDER)
dataset = pd.read_json('thai_sentiment_dataset.json', typ='series')

dfseq_story = []
dfseq_sentiments = []

# Iterate each topic and create a DataFrame
for topic_key, topic_value in dataset.items():

    if isinstance(topic_value, dict) and topic_key == 'story':
        df = pd.DataFrame.from_dict(topic_value, orient='index', columns=['story'])
        df['topic'] = topic_key
        dfseq_story.append(df)

    if isinstance(topic_value, dict) and topic_key == 'sentiment':
        df = pd.DataFrame.from_dict(topic_value, orient='index', columns=['sentiment'])
        df['topic'] = topic_key
        dfseq_sentiments.append(df)
    else:
        print(f"Skipping topic {topic_key} as it doesn't contain valid data.")

# Concatenate all single DataFrames
if dfseq_story:
    final_df_story = pd.concat(dfseq_story, ignore_index=True)
    print("DataFrame for 'story':")
    print(final_df_story)
else:
    print("No 'story' DataFrames to concatenate")

if dfseq_sentiments:
    final_df_sentiment = pd.concat(dfseq_sentiments, ignore_index=True)
    print("DataFrame for 'sentiments':")
    print(final_df_sentiment)
else:
    print("No 'sentiment' DataFrames to concatenate")

final_df = pd.concat([final_df_story, final_df_sentiment], axis=1)

# Preprocess text data
final_df['story'] = final_df['story'].str.lower().str.replace('[^\w\s]', '')  # Example preprocessing (lowercase and remove punctuation)

# Vectorize text data
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(final_df['story'])
y = final_df['sentiment']

# Predictions
predictions = nb_classifier.predict(X)

# Evaluate model
accuracy = accuracy_score(y, predictions)
print("Accuracy:", accuracy)

# Print classification report
print(classification_report(y, predictions))


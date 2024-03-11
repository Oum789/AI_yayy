import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import KFold
from sklearn.metrics import classification_report, confusion_matrix

# Read the JSON file
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

# Train Naive Bayes classifier using k-fold cross-validation
nb_classifier = MultinomialNB()
kf = KFold(n_splits=30, shuffle=True, random_state=42)  # Define KFold
cv_scores = []
fold_idx = 0
for train_idx, test_idx in kf.split(X):  # Loop over splits
    fold_idx += 1
    X_train_fold, X_test_fold = X[train_idx], X[test_idx]
    y_train_fold, y_test_fold = y[train_idx], y[test_idx]

    nb_classifier.fit(X_train_fold, y_train_fold)
    predictions_fold = nb_classifier.predict(X_test_fold)

    cv_scores.append(nb_classifier.score(X_test_fold, y_test_fold))  # Append accuracy for this fold

    print(f"\nClassification Report - Fold {fold_idx}:")
    print(classification_report(y_test_fold, predictions_fold))
    
    # Calculate confusion matrix
    conf_matrix = confusion_matrix(y_test_fold, predictions_fold)
    print(f"\nConfusion Matrix - Fold {fold_idx}:")
    print(conf_matrix)

print("Cross-Validation Scores:", cv_scores)
print("Mean Accuracy:", sum(cv_scores) / len(cv_scores))  # Calculate mean accuracy
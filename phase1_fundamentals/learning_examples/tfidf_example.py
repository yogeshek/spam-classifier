"""
Simple TF-IDF Example - Understanding how it works

TF-IDF = Term Frequency - Inverse Document Frequency
- Converts text into numbers that ML models can understand
- Gives high scores to important/distinctive words
- Gives low scores to common words
"""

# Fix encoding for Windows terminal to display emojis
import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# Import required libraries
from sklearn.feature_extraction.text import TfidfVectorizer  # Main TF-IDF tool
import pandas as pd  # For displaying results in table format

# Step 1: Create sample documents (our training data)
# These represent different movie reviews
documents = [
    "I love this movie",           # Positive review
    "I hate this movie",            # Negative review
    "This movie is amazing",        # Positive review
    "Great movie with excellent acting"  # Positive review
]

print("=" * 60)
print("SIMPLE TF-IDF EXAMPLE")
print("=" * 60)

# Step 2: Display our sample documents
print("\n📄 Documents:")
for i, doc in enumerate(documents, 1):
    print(f"{i}. {doc}")

# Step 3: Create TF-IDF Vectorizer object
# This is the tool that will convert text to numbers
vectorizer = TfidfVectorizer()
# Note: We're using default settings here
# In real projects, you can customize:
#   - max_features: limit vocabulary size
#   - stop_words: remove common words like "the", "is"
#   - ngram_range: include phrases like "not good"

# Step 4: Fit and transform the documents
# fit_transform does TWO things:
#   1. fit(): Learn the vocabulary from documents (which words exist)
#   2. transform(): Convert documents to TF-IDF numbers
tfidf_matrix = vectorizer.fit_transform(documents)
# Result: A matrix where each row is a document, each column is a word
# Each cell contains the TF-IDF score for that word in that document

# Step 5: Get the vocabulary (list of all unique words)
feature_names = vectorizer.get_feature_names_out()
# This returns all unique words found in the documents
# Words are automatically sorted alphabetically

print(f"\n📊 Vocabulary (unique words): {len(feature_names)}")
print(feature_names)
# Notice: Words are lowercase and sorted alphabetically

# Step 6: Convert the sparse matrix to a readable table
# tfidf_matrix is a "sparse matrix" (efficient storage format)
# We convert it to a DataFrame (table) for easy viewing
df = pd.DataFrame(
    tfidf_matrix.toarray(),  # Convert sparse matrix to regular array
    columns=feature_names,    # Column names = words
    index=[f"Doc {i+1}" for i in range(len(documents))]  # Row names = documents
)

print("\n🔢 TF-IDF Matrix (word importance scores):")
print(df.round(3))  # Round to 3 decimal places for readability
# How to read this table:
# - Rows = Documents (Doc 1, Doc 2, etc.)
# - Columns = Words (acting, amazing, etc.)
# - Values = TF-IDF scores (0.000 to ~1.000)

print("\n" + "=" * 60)
print("📖 How to read this:")
print("=" * 60)
print("• Higher number = more important word for that document")
print("• 0.000 = word doesn't appear or is too common")
print("• Common words (movie, this) get low scores")
print("• Unique words (love, hate, amazing) get high scores")

# Step 7: Analyze a specific document in detail
print("\n" + "=" * 60)
print("🔍 Analyzing Document 1: 'I love this movie'")
print("=" * 60)

# Extract TF-IDF scores for Document 1
doc1_scores = df.loc["Doc 1"]  # Get the row for Doc 1

# Sort words by their importance (highest score first)
top_words = doc1_scores.sort_values(ascending=False).head(5)  # Get top 5 words

print("\nTop 5 important words:")
for word, score in top_words.items():
    if score > 0:  # Only show words that actually appear (score > 0)
        print(f"  '{word}': {score:.3f}")

print("\n💡 Notice:")
print("  • 'love' has HIGH score → appears only in this doc")
print("    Why? It's UNIQUE/RARE, so it's very distinctive!")
print("  • 'movie' has LOW score → appears in all docs")
print("    Why? It's COMMON, so it doesn't help distinguish documents")
print("  • Common words are less important for classification!")
print("    TF-IDF automatically downweights common words!")

# Step 8: Understand why TF-IDF helps Machine Learning
print("\n" + "=" * 60)
print("🎯 Why TF-IDF is useful for ML:")
print("=" * 60)
print("✓ Positive reviews have words: love, amazing, great, excellent")
print("✓ Negative reviews have words: hate, bad, terrible")
print("✓ TF-IDF gives these distinctive words HIGH scores")
print("✓ Common words (movie, this, the) get LOW scores")
print("✓ ML model learns from these patterns!")
print("\n💡 The model learns: 'love' → positive, 'hate' → negative")
print("   Common words like 'movie' don't help, so they're ignored!")

# Step 9: Test with a NEW review (prediction scenario)
print("\n" + "=" * 60)
print("🧪 Test with new review:")
print("=" * 60)

# Create a new review that wasn't in training data
new_review = ["This movie is terrible and boring"]

# Transform the new review using the SAME vectorizer
# Important: We use transform() NOT fit_transform()
# Why? We already learned the vocabulary, just converting new text
new_tfidf = vectorizer.transform(new_review)

print(f"\nNew review: '{new_review[0]}'")
print("\nWords the model sees:")

# Convert the TF-IDF matrix to a DataFrame for easy viewing
new_df = pd.DataFrame(
    new_tfidf.toarray(),  # Convert to regular array
    columns=feature_names  # Use the same vocabulary we learned before
)

# Display only words that appear in the new review
for word in new_df.columns:
    if new_df[word][0] > 0:  # If score > 0, the word appears
        print(f"  '{word}': {new_df[word][0]:.3f}")

# Step 10: Important limitation to understand
print("\n⚠️ Note: 'terrible' and 'boring' are NOT in vocabulary")
print("   because they weren't in training documents!")
print("   This is why we need diverse training data.")
print("\n📌 Key Insight:")
print("   - Vectorizer only knows words from training data")
print("   - Unknown words are IGNORED during prediction")
print("   - Solution: Train on large, diverse datasets!")
print("\n🔄 In your spam classifier:")
print("   - Trained on 5000+ emails → learns many spam words")
print("   - New email with 'WINNER!' → recognized as spam")
print("   - New email with unknown word → ignored, other words used")

"""
Practical Example: Decoding Sparse Matrix Output
How to see actual words from (row, column) indices
"""

import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

print("=" * 70)
print("DECODING SPARSE MATRIX OUTPUT - PRACTICAL EXAMPLE")
print("=" * 70)

# Load your spam data
try:
    df = pd.read_csv('data/spam.csv', encoding='latin-1')
    df = df[['v1', 'v2']]
    df.columns = ['label', 'message']

    print("\n✅ Loaded spam dataset")
    print(f"   Total emails: {len(df)}")

    # Simple preprocessing
    import re
    def clean_text(text):
        text = text.lower()
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        text = ' '.join(text.split())
        return text

    df['message'] = df['message'].apply(clean_text)
    df['label'] = df['label'].map({'ham': 0, 'spam': 1})

    # Split data
    X = df['message']
    y = df['label']
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Vectorize
    vectorizer = TfidfVectorizer(max_features=3000, stop_words='english')
    X_train_vec = vectorizer.fit_transform(X_train)

    print(f"   Training samples: {len(X_train)}")
    print(f"   Vocabulary size: {len(vectorizer.get_feature_names_out())}")

    print("\n" + "=" * 70)
    print("📊 SPARSE MATRIX OUTPUT")
    print("=" * 70)

    print("\nWhen you print X_train_vec, you see:")
    print(X_train_vec[:3])  # Show first 3 documents

    print("\n" + "=" * 70)
    print("🔍 DECODING THE OUTPUT")
    print("=" * 70)

    # Get vocabulary
    vocab = vectorizer.get_feature_names_out()

    # Analyze first document
    print("\n1️⃣ First Training Email:")
    first_email = X_train.iloc[0]
    print(f"   Original text: '{first_email[:80]}...'")

    print("\n   Sparse representation:")
    first_doc_sparse = X_train_vec[0]
    print(first_doc_sparse)

    print("\n   Decoding to actual words:")
    # Convert to dense to easily iterate
    first_doc_dense = first_doc_sparse.toarray()[0]

    # Get top 10 words
    word_scores = []
    for idx, score in enumerate(first_doc_dense):
        if score > 0:
            word_scores.append((vocab[idx], score))

    # Sort by score
    word_scores.sort(key=lambda x: x[1], reverse=True)

    print("\n   Top 10 most important words:")
    for i, (word, score) in enumerate(word_scores[:10], 1):
        print(f"   {i:2}. '{word:15s}' → {score:.4f}")

    print("\n" + "=" * 70)
    print("🧪 UNDERSTANDING YOUR EXAMPLE")
    print("=" * 70)

    print("\nYour output: (0, 23)  0.269")
    print("            (0, 46)  0.152")
    print()
    print("This means:")
    print(f"  • Row 0 = First email in training set")
    print(f"  • Column 23 = Word '{vocab[23]}'")
    print(f"  • Column 46 = Word '{vocab[46]}'")
    print()
    print("So the first email contains:")
    print(f"  • '{vocab[23]}' with importance score 0.269")
    print(f"  • '{vocab[46]}' with importance score 0.152")

    print("\n" + "=" * 70)
    print("💡 PRACTICAL TIPS")
    print("=" * 70)

    print("\n1. To see vocabulary mapping:")
    print("   vocab = vectorizer.get_feature_names_out()")
    print("   print(vocab[23])  # See word at index 23")
    print()
    print("2. To see all words in a document:")
    print("   doc = X_train_vec[0].toarray()[0]")
    print("   for idx, score in enumerate(doc):")
    print("       if score > 0:")
    print("           print(vocab[idx], score)")
    print()
    print("3. To convert to readable DataFrame:")
    print("   df = pd.DataFrame(X_train_vec.toarray(), columns=vocab)")
    print("   print(df.head())  # View first 5 documents")
    print()
    print("4. For debugging only! Don't convert in production:")
    print("   • Sparse matrices are memory efficient")
    print("   • Only convert to dense for viewing/debugging")

    print("\n" + "=" * 70)
    print("🎯 VISUAL COMPARISON")
    print("=" * 70)

    # Show a small example
    print("\nExample with first 3 emails, first 10 words:")

    # Get first 3 documents, first 10 features
    small_matrix = X_train_vec[:3, :10].toarray()
    small_vocab = vocab[:10]

    df_small = pd.DataFrame(
        small_matrix,
        columns=small_vocab,
        index=['Email 0', 'Email 1', 'Email 2']
    )

    print(df_small.round(3))

    print("\n   • Each row = one email")
    print("   • Each column = one word")
    print("   • 0.000 = word not in that email")
    print("   • Higher values = more important words")

except FileNotFoundError:
    print("\n❌ Error: spam.csv not found!")
    print("   This example needs the spam dataset to run.")
    print("   Place spam.csv in the data/ folder.")

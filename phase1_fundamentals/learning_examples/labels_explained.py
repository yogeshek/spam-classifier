"""
Understanding Labels in Machine Learning
Where are the labels after vectorization?
"""

import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

print("=" * 70)
print("WHERE ARE THE LABELS? - UNDERSTANDING X and y")
print("=" * 70)

# Simple example first
print("\n📚 Simple Example:")
print("-" * 70)

documents = [
    "I love this movie",      # Positive
    "Great acting",           # Positive
    "I hate this film",       # Negative
    "Terrible movie",         # Negative
]

labels = [1, 1, 0, 0]  # 1 = positive, 0 = negative

print("\nOriginal Data:")
for i, (doc, label) in enumerate(zip(documents, labels)):
    sentiment = "Positive" if label == 1 else "Negative"
    print(f"  {i}. '{doc}' → {sentiment} (label={label})")

print("\n" + "=" * 70)
print("🔄 WHAT GETS VECTORIZED?")
print("=" * 70)

# Vectorize ONLY the text, NOT the labels
vectorizer = TfidfVectorizer()
X_vectorized = vectorizer.fit_transform(documents)  # Text → Numbers
y = np.array(labels)  # Labels stay as-is!

print("\n✅ TEXT (X) gets vectorized:")
print("   Before: ['I love this movie', 'Great acting', ...]")
print(f"   After:  Matrix shape {X_vectorized.shape}")
print(f"           (4 documents × {X_vectorized.shape[1]} words)")
print()
print("   Each document becomes a row of numbers:")
print(X_vectorized.toarray().round(3))

print("\n✅ LABELS (y) stay exactly the same:")
print(f"   Before: [1, 1, 0, 0]")
print(f"   After:  {y}")
print()
print("   Labels are ALREADY NUMBERS! No vectorization needed!")

print("\n" + "=" * 70)
print("🧠 WHY DON'T WE VECTORIZE LABELS?")
print("=" * 70)

print("\n1. Labels are already in numeric form")
print("   • Spam classifier: 0 (ham) or 1 (spam)")
print("   • Sentiment: 0 (negative), 1 (positive)")
print("   • Multi-class: 0, 1, 2, 3, ... (categories)")
print()
print("2. Labels are what we're trying to PREDICT")
print("   • X (features) → goes into the model")
print("   • y (labels) → tells the model the correct answer")
print()
print("3. Vectorization is only for TEXT → NUMBERS conversion")
print("   • Labels are already numbers!")

print("\n" + "=" * 70)
print("📊 VISUALIZING THE RELATIONSHIP")
print("=" * 70)

# Create a visual table
vocab = vectorizer.get_feature_names_out()
df_visual = pd.DataFrame(
    X_vectorized.toarray(),
    columns=vocab
)
df_visual['→ LABEL'] = y
df_visual.index = [f"Doc {i}" for i in range(len(documents))]

print("\n" + str(df_visual.round(3)))

print("\n📖 How to read this:")
print("   • Columns (acting, film, great...) = Features (X)")
print("   • Last column (→ LABEL) = Target (y)")
print("   • Model learns: 'love', 'great' → Label 1 (positive)")
print("   • Model learns: 'hate', 'terrible' → Label 0 (negative)")

print("\n" + "=" * 70)
print("🎯 IN YOUR SPAM CLASSIFIER")
print("=" * 70)

print("\nIn train.py, you have:")
print()
print("# Step 1: Split data")
print("X = df['message']          # Text messages (features)")
print("y = df['label']            # 0=ham, 1=spam (labels)")
print()
print("X_train, X_test, y_train, y_test = train_test_split(X, y, ...)")
print()
print("# Step 2: Vectorize ONLY X (the text)")
print("X_train_vec = vectorizer.fit_transform(X_train)  # Text → Numbers")
print("X_test_vec = vectorizer.transform(X_test)        # Text → Numbers")
print()
print("# y_train and y_test stay as-is! They're already numbers!")
print()
print("# Step 3: Train model")
print("model.fit(X_train_vec, y_train)")
print("          ↑             ↑")
print("          |             |")
print("    Vectorized      Original labels")
print("    text data       (NOT vectorized)")

print("\n" + "=" * 70)
print("🔍 PRACTICAL DEMONSTRATION")
print("=" * 70)

# Load actual spam data
try:
    df = pd.read_csv('data/spam.csv', encoding='latin-1')
    df = df[['v1', 'v2']]
    df.columns = ['label', 'message']

    print("\n✅ Loaded spam dataset")

    # Show original
    print("\nOriginal data (first 3 rows):")
    print(df.head(3).to_string())

    # Preprocess
    import re
    def clean_text(text):
        text = text.lower()
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        text = ' '.join(text.split())
        return text

    df['message'] = df['message'].apply(clean_text)
    df['label'] = df['label'].map({'ham': 0, 'spam': 1})

    print("\nAfter converting labels to numbers:")
    print(df[['message', 'label']].head(3).to_string())

    # Split
    X = df['message']
    y = df['label']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print(f"\n" + "=" * 70)
    print("📦 AFTER SPLITTING")
    print("=" * 70)

    print(f"\nX_train (text messages):")
    print(f"  Type: {type(X_train)}")
    print(f"  Shape: {X_train.shape}")
    print(f"  First message: '{X_train.iloc[0][:60]}...'")

    print(f"\ny_train (labels):")
    print(f"  Type: {type(y_train)}")
    print(f"  Shape: {y_train.shape}")
    print(f"  First 10 labels: {y_train.iloc[:10].tolist()}")
    print(f"  Value counts: {y_train.value_counts().to_dict()}")

    # Vectorize
    vectorizer = TfidfVectorizer(max_features=3000, stop_words='english')
    X_train_vec = vectorizer.fit_transform(X_train)

    print(f"\n" + "=" * 70)
    print("🔢 AFTER VECTORIZING X_train")
    print("=" * 70)

    print(f"\nX_train_vec (vectorized text):")
    print(f"  Type: {type(X_train_vec)}")
    print(f"  Shape: {X_train_vec.shape}")
    print(f"  Format: Sparse matrix (only non-zero values stored)")

    print(f"\ny_train (labels) - UNCHANGED:")
    print(f"  Type: {type(y_train)}")
    print(f"  Shape: {y_train.shape}")
    print(f"  First 10 labels: {y_train.iloc[:10].tolist()}")
    print(f"  Still the same! [0, 0, 1, 0, ...]")

    print(f"\n" + "=" * 70)
    print("✅ KEY INSIGHT")
    print("=" * 70)
    print("\nX_train_vec and y_train are PAIRED:")
    print()
    print(f"  X_train_vec[0] ← vectorized first message")
    print(f"  y_train.iloc[0] = {y_train.iloc[0]} ← label for first message")
    print()
    print(f"  X_train_vec[1] ← vectorized second message")
    print(f"  y_train.iloc[1] = {y_train.iloc[1]} ← label for second message")
    print()
    print("And so on...")
    print()
    print("The model learns the mapping:")
    print("  X_train_vec[i] → y_train[i]")
    print("  (message features) → (correct label)")

    print(f"\n" + "=" * 70)
    print("🎓 TRAINING THE MODEL")
    print("=" * 70)

    print("\nmodel.fit(X_train_vec, y_train)")
    print()
    print("What happens:")
    print("  1. Model looks at X_train_vec[0] (vectorized message)")
    print(f"     and y_train.iloc[0]={y_train.iloc[0]} (correct answer)")
    print()
    print("  2. Model looks at X_train_vec[1] (vectorized message)")
    print(f"     and y_train.iloc[1]={y_train.iloc[1]} (correct answer)")
    print()
    print("  3. Repeats for all 4,457 training samples")
    print()
    print("  4. Model learns patterns:")
    print("     'Words like free, winner, urgent → label 1 (spam)'")
    print("     'Words like meeting, tomorrow → label 0 (ham)'")

except FileNotFoundError:
    print("\n❌ spam.csv not found - showing conceptual explanation only")

print("\n" + "=" * 70)
print("📌 SUMMARY")
print("=" * 70)
print()
print("❓ Where are the labels after vectorization?")
print("✅ Right where they were! Labels DON'T get vectorized.")
print()
print("Only X (text) gets vectorized:")
print("  X_train → X_train_vec (via vectorizer.fit_transform)")
print()
print("Labels stay as numbers:")
print("  y_train → stays as y_train [0, 1, 0, 1, ...]")
print()
print("Both are used together in training:")
print("  model.fit(X_train_vec, y_train)")
print("            ↑             ↑")
print("       Features      Labels")
print("      (vectorized)   (not vectorized)")

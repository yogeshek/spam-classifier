"""
Understanding Sparse Matrix Output from TF-IDF
What does (0, 23) 0.2694659942809374 mean?
"""

import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
import numpy as np

print("=" * 70)
print("UNDERSTANDING SPARSE MATRIX OUTPUT")
print("=" * 70)

# Simple example
documents = [
    "I love this movie",
    "I hate this movie"
]

vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(documents)

print("\n📄 Documents:")
for i, doc in enumerate(documents):
    print(f"  {i}. {doc}")

print("\n" + "=" * 70)
print("🔍 SPARSE MATRIX FORMAT (what you see when you print)")
print("=" * 70)

print("\n1️⃣ Printing the sparse matrix directly:")
print(tfidf_matrix)

print("\n📖 How to read this:")
print("   Format: (row, column)    value")
print("   --------------------------------")
print("   (0, 1)  0.538 → Row 0, Column 1, Score = 0.538")
print("   (0, 3)  0.442 → Row 0, Column 3, Score = 0.442")
print()
print("   • Row = Document number (0 = first document)")
print("   • Column = Word index in vocabulary")
print("   • Value = TF-IDF score")
print()
print("   ⚠️ Only NON-ZERO values are shown!")
print("      (That's what makes it 'sparse')")

print("\n" + "=" * 70)
print("🔢 CONVERTING TO READABLE FORMAT")
print("=" * 70)

# Get vocabulary
vocab = vectorizer.get_feature_names_out()
print(f"\n2️⃣ Vocabulary (words found):")
print(f"   {list(vocab)}")
print(f"\n   Index mapping:")
for idx, word in enumerate(vocab):
    print(f"   Column {idx} = '{word}'")

# Convert to dense array
print("\n3️⃣ Converting sparse matrix to full array:")
dense_array = tfidf_matrix.toarray()
print(dense_array)
print("\n   This is a 2D array (2 rows × " + str(len(vocab)) + " columns)")
print("   • Each row = one document")
print("   • Each column = one word")
print("   • Most values are 0.000 (words not in that document)")

# Convert to DataFrame for clarity
print("\n4️⃣ Converting to readable table (DataFrame):")
df = pd.DataFrame(
    tfidf_matrix.toarray(),
    columns=vocab,
    index=[f"Doc {i}" for i in range(len(documents))]
)
print(df.round(3))

print("\n" + "=" * 70)
print("💡 WHY USE SPARSE MATRICES?")
print("=" * 70)

print("\n🎯 Real-world example:")
print("   • Your spam dataset: ~5000 emails")
print("   • Vocabulary size: 3000 words")
print("   • Full matrix size: 5000 × 3000 = 15,000,000 numbers!")
print()
print("   But each email only uses ~50-100 words")
print("   So ~99% of the matrix is zeros!")
print()
print("   📊 Storage comparison:")
print("      Full matrix: ~120 MB")
print("      Sparse matrix: ~2-5 MB")
print()
print("   ✅ Sparse matrices only store non-zero values")
print("   ✅ Much faster and uses less memory")

print("\n" + "=" * 70)
print("🔧 WORKING WITH SPARSE MATRICES")
print("=" * 70)

print("\n5️⃣ Accessing specific values:")

# Example: Get Document 0
doc_0_sparse = tfidf_matrix[0]
print(f"\nDocument 0 (sparse format):")
print(doc_0_sparse)

# Convert to array to see all values
doc_0_dense = doc_0_sparse.toarray()[0]
print(f"\nDocument 0 (full array):")
print(doc_0_dense.round(3))

# Show which words have non-zero scores
print(f"\nDocument 0 - Words with scores:")
for idx, score in enumerate(doc_0_dense):
    if score > 0:
        print(f"   '{vocab[idx]}': {score:.3f}")

print("\n" + "=" * 70)
print("📌 KEY TAKEAWAYS")
print("=" * 70)
print()
print("1. Sparse matrix format: (row, column) value")
print("   • Only shows NON-ZERO values")
print("   • Saves memory for large datasets")
print()
print("2. To see full matrix:")
print("   • Use .toarray() to convert to numpy array")
print("   • Use pandas DataFrame for readable table")
print()
print("3. ML models can work directly with sparse matrices")
print("   • model.fit(X_train_vec, y_train) ← works with sparse!")
print("   • No need to convert to dense unless viewing")
print()
print("4. In your spam classifier:")
print("   • X_train_vec is sparse matrix (memory efficient)")
print("   • model.fit() uses it directly")
print("   • Only convert to dense for visualization/debugging")

print("\n" + "=" * 70)
print("🧪 DECODING YOUR EXAMPLE")
print("=" * 70)
print()
print("Your output:")
print("  (0, 23)  0.2694659942809374")
print("  (0, 46)  0.1527004108592188")
print()
print("Meaning:")
print("  • Document 0 (first email in training set)")
print("  • Word at index 23 has TF-IDF score = 0.269")
print("  • Word at index 46 has TF-IDF score = 0.152")
print()
print("To see which words these are:")
print("  vocab = vectorizer.get_feature_names_out()")
print("  print(vocab[23])  # Shows the actual word")
print("  print(vocab[46])  # Shows the actual word")

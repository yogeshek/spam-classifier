# Understanding Labels in Machine Learning

## 🎯 Quick Answer

**Question:** Where are the labels (sentiments) after vectorization?

**Answer:** The labels **don't get vectorized**. They stay exactly as they are!

---

## 📊 Visual Representation

```
BEFORE SPLIT:
┌─────────────────────────────────────────┐
│ Original Data (DataFrame)               │
├─────────────────────────────────────────┤
│ message              │ label            │
├──────────────────────┼──────────────────┤
│ "Free prize!"        │ 1 (spam)         │
│ "Meeting at 3pm"     │ 0 (ham)          │
│ "Winner! Click now"  │ 1 (spam)         │
│ "How are you?"       │ 0 (ham)          │
└─────────────────────────────────────────┘

                ↓ train_test_split()

AFTER SPLIT:
┌───────────────────┐         ┌────────────┐
│ X_train           │         │ y_train    │
│ (TEXT)            │         │ (NUMBERS)  │
├───────────────────┤         ├────────────┤
│ "Free prize!"     │    ←→   │ 1          │
│ "How are you?"    │    ←→   │ 0          │
│ ...               │         │ ...        │
└───────────────────┘         └────────────┘
        ↓                            ↓
   VECTORIZE!                   STAYS THE SAME!
        ↓                            ↓
┌───────────────────┐         ┌────────────┐
│ X_train_vec       │         │ y_train    │
│ (NUMBERS)         │         │ (NUMBERS)  │
├───────────────────┤         ├────────────┤
│ [0.5, 0.8, 0...]  │    ←→   │ 1          │
│ [0.3, 0.0, 0...]  │    ←→   │ 0          │
│ ...               │         │ ...        │
└───────────────────┘         └────────────┘
        ↓                            ↓
        └────────────┬───────────────┘
                     ↓
            model.fit(X_train_vec, y_train)
```

---

## 🔍 Step-by-Step Breakdown

### Step 1: Original Data
```python
df = pd.read_csv('spam.csv')
# message                  label
# "Free prize!"            spam
# "Meeting at 3pm"         ham
```

### Step 2: Convert Labels to Numbers
```python
df['label'] = df['label'].map({'ham': 0, 'spam': 1})
# message                  label
# "Free prize!"            1
# "Meeting at 3pm"         0
```

### Step 3: Split into Features (X) and Labels (y)
```python
X = df['message']    # Text (features)
y = df['label']      # Numbers (labels)
```

### Step 4: Split into Train/Test
```python
X_train, X_test, y_train, y_test = train_test_split(X, y, ...)

# X_train = ["Free prize!", "Meeting at 3pm", ...]
# y_train = [1, 0, ...]  ← Already numbers!
```

### Step 5: Vectorize ONLY X (NOT y)
```python
vectorizer = TfidfVectorizer()
X_train_vec = vectorizer.fit_transform(X_train)  # ✅ Text → Numbers
# y_train stays as [1, 0, ...] ← ✅ No change!

# X_train_vec shape: (4457, 3000)
#   • 4457 emails
#   • 3000 features (words)
#   • Each email becomes a row of 3000 numbers

# y_train shape: (4457,)
#   • 4457 labels
#   • One label per email: 0 or 1
```

### Step 6: Train Model
```python
model.fit(X_train_vec, y_train)
#         ↑            ↑
#         |            |
#    Features      Labels
#   (vectorized)  (NOT vectorized)
```

---

## 💡 Key Insights

### Why Don't We Vectorize Labels?

1. **Labels are already numbers**
   - `0` (ham) or `1` (spam)
   - No need to convert!

2. **Vectorization is for TEXT → NUMBERS**
   - Labels are not text
   - They're categorical numbers

3. **They serve different purposes**
   - **X (features)**: The INPUT to the model (what it sees)
   - **y (labels)**: The OUTPUT the model learns to predict (correct answer)

---

## 🎓 The Training Process

```python
# For each training sample:
model.fit(X_train_vec, y_train)

# Iteration 1:
#   Input:  X_train_vec[0] = [0.5, 0.8, 0.0, ...] (vectorized "Free prize!")
#   Output: y_train[0] = 1 (spam)
#   Model learns: These word patterns → spam

# Iteration 2:
#   Input:  X_train_vec[1] = [0.3, 0.0, 0.6, ...] (vectorized "Meeting at 3pm")
#   Output: y_train[1] = 0 (ham)
#   Model learns: These word patterns → ham

# ... repeats for all 4,457 training samples
```

---

## 📌 Common Confusion

### ❌ WRONG Thinking:
"We vectorized X_train, so we also need to vectorize y_train"

### ✅ CORRECT Thinking:
- **X_train** = Text → needs vectorization → **X_train_vec**
- **y_train** = Numbers → already usable → **stays as y_train**

---

## 🔬 Verification Code

```python
# Check shapes match
print(f"X_train_vec shape: {X_train_vec.shape}")  # (4457, 3000)
print(f"y_train shape: {y_train.shape}")          # (4457,)
#                                                     ↑
#                              Same number of samples!

# Check labels are still numbers
print(f"First 10 labels: {y_train[:10]}")
# Output: [1, 0, 0, 1, 0, 0, 0, 1, 0, 0]
#         ↑ Still 0s and 1s!

# Check they're paired correctly
print(f"First email: {X_train.iloc[0]}")
# Output: "Free prize!"

print(f"Its label: {y_train.iloc[0]}")
# Output: 1 (spam)

print(f"Vectorized: {X_train_vec[0]}")
# Output: (0, 234) 0.5  (0, 678) 0.8  ...
#         ↑ Text converted to numbers

print(f"Label unchanged: {y_train.iloc[0]}")
# Output: 1 (still spam!)
```

---

## 🎯 Summary Table

| Component | Before Split | After Split | After Vectorization | Used In Training |
|-----------|-------------|-------------|---------------------|------------------|
| **Text Messages** | DataFrame column | X_train (text) | X_train_vec (numbers) | ✅ model.fit() |
| **Labels** | DataFrame column | y_train (0/1) | y_train (0/1) ← SAME | ✅ model.fit() |

**Key Point:** Only the text gets vectorized. Labels stay as-is!

---

## 🚀 For Sentiment Analysis

Same concept applies:

```python
# Movie Review Dataset
# review                           sentiment
# "Great movie!"                   positive  (1)
# "Terrible acting"                negative  (0)

# After vectorization:
X_train_vec = vectorizer.fit_transform(X_train)  # Reviews → Numbers
y_train = [1, 0, 1, 0, ...]                      # Labels stay as-is!

model.fit(X_train_vec, y_train)
```

The sentiment labels (0, 1) don't get vectorized — they're already in the format the model needs!

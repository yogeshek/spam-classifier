 # predict.py - Can run unlimited times
import pickle

  # Load saved models (takes milliseconds)
with open('models/spam_classifier.pkl', 'rb') as f:
      model = pickle.load(f)

with open('models/vectorizer.pkl', 'rb') as f:
      vectorizer = pickle.load(f)

  # Get user input
message = input("Enter a message to check: ")

  # Make prediction
message_vec = vectorizer.transform([message])
prediction = model.predict(message_vec)[0]
probability = model.predict_proba(message_vec)[0]

  # Display result
if prediction == 1:
      print(f"🚨 SPAM (Confidence: {probability[1]*100:.1f}%)")
else:
      print(f"✅ HAM (Confidence: {probability[0]*100:.1f}%)")

---
name: test-spam
description: Test if a message is spam or ham using the trained model
args:
  message:
    description: The message text to classify
    required: false
---

# Instructions

When this skill is invoked:

1. If a message argument is provided, use that message
2. If no message is provided, ask the user for the message to test
3. Look for the trained model in the 'models/' directory
4. Run the prediction using 'predict.py' or create a quick test script
5. Display the result showing:
    - The classification (spam/ham)
    - The confidence score if available
    - A brief explanation

Make sure the model exists before running. If not found, inform the user they need to train the model first.

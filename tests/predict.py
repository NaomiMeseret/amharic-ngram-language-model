from src.ngram_model import NGramModel

# Load the already-trained model
model = NGramModel.load("results/model")

# Test next-word prediction
while True:
    text = input("Enter Amharic text (or 'exit'): ")

    if text.lower() == "exit":
        break

    predictions = model.predict_next(text, k=5, method="backoff")

    for word, probability in predictions:
        print(f"{word}: {probability:.4f}")
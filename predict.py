import sys
from ml_utils import ensure_model

message = " ".join(sys.argv[1:]).strip()
if not message:
    message = input("Enter an SMS message: ").strip()

model = ensure_model()
prediction = model.predict([message])[0]
probabilities = model.predict_proba([message])[0]
classes = list(model.classes_)
spam_probability = probabilities[classes.index("spam")]

print(f"Prediction: {prediction.upper()}")
print(f"Spam probability: {spam_probability:.2%}")

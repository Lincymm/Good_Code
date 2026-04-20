from model import train_model

def main():
    model = train_model()
    sample = [[5.1, 3.5, 1.4, 0.2]]
    prediction = model.predict(sample)

    print(f"Prediction: {prediction}")

if __name__ == "__main__":
    main()

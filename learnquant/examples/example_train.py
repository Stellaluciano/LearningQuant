from learnquant.data.loader import DataLoader
from learnquant.features.feature import generate_features
from learnquant.ml.train import train

if __name__ == "__main__":
    loader = DataLoader("learnquant/data")
    df = loader.load_processed("example_market")
    features = generate_features(df)
    result = train(features, model_type="sklearn")
    print("Training metrics:", result.metrics)

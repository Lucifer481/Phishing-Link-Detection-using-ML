import pandas as pd
from feature_extraction import extract_features

def create_feature_dataframe(df, url_col='url'):
    feature_list = []
    for url in df[url_col]:
        features = extract_features(url)
        feature_list.append(features)
    feature_df = pd.DataFrame(feature_list)
    return pd.concat([df.reset_index(drop=True), feature_df], axis=1)

if __name__ == "__main__":
    df = pd.read_csv('data/processed/dataset_cleaned.csv')
    df_with_features = create_feature_dataframe(df)
    df_with_features.to_csv('data/processed/dataset_features.csv', index=False)
    print("Feature extraction completed. Saved to data/processed/dataset_features.csv")

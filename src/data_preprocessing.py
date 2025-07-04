import pandas as pd

def load_data(phishing_path_1, phishing_path_2, legitimate_path):
    # Load both phishing datasets
    phishing_df_1 = pd.read_csv(phishing_path_1)
    phishing_df_2 = pd.read_csv(phishing_path_2)

    # Ensure both have a consistent column name for URLs
    phishing_df_1.columns = [col.lower().strip() for col in phishing_df_1.columns]
    phishing_df_2.columns = [col.lower().strip() for col in phishing_df_2.columns]

    # Rename to 'url' if needed
    if 'url' not in phishing_df_1.columns:
        phishing_df_1.rename(columns={phishing_df_1.columns[0]: 'url'}, inplace=True)
    if 'url' not in phishing_df_2.columns:
        phishing_df_2.rename(columns={phishing_df_2.columns[0]: 'url'}, inplace=True)

    # Combine both phishing datasets
    phishing_df = pd.concat([phishing_df_1, phishing_df_2], ignore_index=True)
    phishing_df['label'] = 1

    # Load legitimate URLs
    legitimate_df = pd.read_csv(legitimate_path)
    legitimate_df.columns = [col.lower().strip() for col in legitimate_df.columns]
    if 'url' not in legitimate_df.columns:
        legitimate_df.rename(columns={legitimate_df.columns[0]: 'url'}, inplace=True)
    legitimate_df['label'] = 0

    return phishing_df, legitimate_df

def clean_urls(df, url_column='url'):
    df = df.dropna(subset=[url_column])
    df = df.drop_duplicates(subset=[url_column])
    return df

def merge_data(phishing_df, legitimate_df):
    combined_df = pd.concat([phishing_df, legitimate_df], ignore_index=True)
    combined_df = combined_df.sample(frac=1).reset_index(drop=True)
    return combined_df

def save_processed_data(df, save_path):
    df.to_csv(save_path, index=False)
    print(f" Processed data saved to {save_path}")

if __name__ == "__main__":
    phishing_path = "data/raw/phishing_urls.csv"
    phishing_site = "data/raw/phishing_site_urls.csv"
    legitimate_path = "data/processed/legitimate_cleaned.csv"
    save_path = "data/processed/dataset_cleaned.csv"

    phishing_df, legitimate_df = load_data(phishing_path, phishing_site, legitimate_path)
    phishing_df = clean_urls(phishing_df)
    legitimate_df = clean_urls(legitimate_df)

    combined_df = merge_data(phishing_df, legitimate_df)
    save_processed_data(combined_df, save_path)

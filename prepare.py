import argparse
import os
import json
import pandas as pd

GARBAGE_FEATURES = ['shortest_words_raw',
                    'nb_underscore',
                    'ratio_extErrors',
                    'nb_tilde',
                    'nb_percent',
                    'nb_star',
                    'nb_dollar',
                    'nb_redirection',
                    'random_domain',
                    'login_form',
                    'punycode',
                    'char_repeat',
                    'iframe',
                    'nb_comma',
                    'port',
                    'onmouseover',
                    'right_clic',
                    'nb_space',
                    'path_extension',
                    'nb_or',
                    'ratio_nullHyperlinks',
                    'ratio_intRedirection',
                    'ratio_intErrors',
                    'submit_email',
                    'sfh', ]


def parse_args():
    """
    Parses command-line arguments.
    """
    parser = argparse.ArgumentParser(description="Prepare raw dataset for training")
    parser.add_argument("-i", "--input", type=str, required=True,
                        help="Path to the raw dataset CSV")
    parser.add_argument("--output-csv", type=str, required=True,
                        help="Path to save the clean CSV (without demo rows)")
    parser.add_argument( "--output-demo", type=str, required=True,
                        help="Path to save the demo JSON file")
    return parser.parse_args()


def encode_status(df: pd.DataFrame) -> pd.DataFrame:
    """
    Encodes the 'status' column to binary format (0: legitimate, 1: phishing).

    Args:
        df (pd.DataFrame): Raw dataframe.

    Returns:
        pd.DataFrame: Dataframe with encoded target.
    """
    df['status'] = df['status'].map({'legitimate': 0, 'phishing': 1})
    return df


def extract_demo(df: pd.DataFrame, output_path: str, n_samples: int = 5) -> pd.DataFrame:
    """
    Extracts demo samples, saves them to JSON, and returns df without them.

    Args:
        df (pd.DataFrame): Input dataframe.
        output_path (str): Path to save demo JSON.
        n_samples (int): Number of samples per class.

    Returns:
        pd.DataFrame: Dataframe without demo samples.
    """
    demo_df = df.groupby('status').sample(n=n_samples, random_state=42)
    df_train = df.drop(demo_df.index)

    demo_data = {}
    for _, row in demo_df.iterrows():
        url = row['url']
        features = row.drop(['url', 'status']).to_dict()
        demo_data[url] = features

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(demo_data, f, indent=4)

    return df_train


def clean_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Drops constant features and features with low correlation to the target.

    Args:
        df (pd.DataFrame): Dataframe to clean.

    Returns:
        pd.DataFrame: Cleaned dataframe.
    """
    # Drop known constants
    existing_constants = [f for f in GARBAGE_FEATURES if f in df.columns]
    df = df.drop(columns=existing_constants)

    # Drop low correlation features (< 0.05)
    correlations = df.corr(numeric_only=True)['status'].abs().sort_values(ascending=False)
    low_corr_features = correlations[correlations < 0.05].index.tolist()
    df = df.drop(columns=low_corr_features)

    return df


def main():
    args = parse_args()

    print(f"1. Loading raw data from {args.input}...")
    df = pd.read_csv(args.input)

    print("2. Encoding target variable...")
    df = encode_status(df)

    print("3. Extracting demo samples...")
    df = extract_demo(df, args.output_demo)

    print("4. Cleaning features...")
    df = df.drop(columns=['url'])
    df = clean_features(df)

    print(f"5. Saving clean dataset to {args.output_csv}...")
    os.makedirs(os.path.dirname(args.output_csv), exist_ok=True)
    df.to_csv(args.output_csv, index=False)

    print("Data preparation completed successfully!")


if __name__ == "__main__":
    main()
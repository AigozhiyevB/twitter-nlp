import argparse
import os
import pandas as pd
from client import Predictor
from tqdm import tqdm
print('Libs loaded')

def plain2df(input_file):
    tweets = []
    with open(input_file, 'r', encoding='utf-8') as file:
        for line in file:
            tweets.append(str(line.strip()))

    df = pd.DataFrame({'Text': tweets})
    return df

def read_file_to_dataframe(input_file, use_csv=False):
    if use_csv:
        df = pd.read_csv(input_file)
    else:
        format_type = detect_format(input_file)

        if format_type == 'text_with_id':
            df = pd.read_csv(input_file, header=None, names=['ID', 'Text'], delimiter='|')
        elif format_type == 'plain_text':
            df = plain2df(input_file)
        else:
            raise ValueError("Unknown format detected!")

    return df

def detect_format(input_file):
    with open(input_file, 'r', encoding='utf-8') as file:
        first_line = file.readline().strip()

    if '|' in first_line:
        return 'text_with_id'
    else:
        return 'plain_text'

def main(input_file, output_file, use_csv=False):
    # Read the input file into a pandas DataFrame
    df = read_file_to_dataframe(input_file, use_csv)

    # Initialize the ML model (replace this with your actual Predictor class instantiation)
    model = Predictor()

    # Make predictions using the ML model
    print('Prediction started')
    inference = [text for text in df['Text'].values]
    preds = [model.predict([i], verbose = 0) for i in tqdm(inference)]
    print('Done')
    df['Sentiment'] = preds
    # Write the DataFrame with predictions to the output file
    df.to_csv(output_file, index=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Detect and process file formats, if ID is passed separate it with '|'")
    parser.add_argument("--input_file", required=True, help="Input file path")
    parser.add_argument("--output_file", required=True, help="Output file path")
    parser.add_argument("--csv", action="store_true", help="Process input file as CSV using pandas")

    args = parser.parse_args()

    if not os.path.isfile(args.input_file):
        print("Input file not found!")
    else:
        main(args.input_file, args.output_file, args.csv)
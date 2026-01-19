import pandas as pd
import argparse
import sys

def preprocess_bank_statements(input_file, output_file='bank_statements_processed.csv'):
    """
    Preprocess bank statement CSV by removing specified columns.
    
    Args:
        input_file (str): Path to the input CSV file
        output_file (str): Path for the output processed CSV file
    """
    try:
        # Read the CSV file
        df = pd.read_csv(input_file)
        
        # Define columns to remove
        columns_to_remove = ['Bank RTN', 'Account Number', 'Check Number', 'Account Running Balance']
        
        # Identify which columns exist in the dataset
        existing_columns = [col for col in columns_to_remove if col in df.columns]
        missing_columns = [col for col in columns_to_remove if col not in df.columns]
        
        if missing_columns:
            print(f"Warning: The following columns were not found in the input file: {missing_columns}")
        
        # Remove the specified columns if they exist
        df_processed = df.drop(columns=existing_columns)
        
        # Save the processed dataframe to a new CSV file
        df_processed.to_csv(output_file, index=False)
        
        print(f"Processing complete!")
        print(f"Input file: {input_file}")
        print(f"Output file: {output_file}")
        print(f"Removed columns: {existing_columns}")
        print(f"Remaining columns: {list(df_processed.columns)}")
        print(f"Number of rows processed: {len(df_processed)}")
        
    except FileNotFoundError:
        print(f"Error: Input file '{input_file}' not found.")
        sys.exit(1)
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess bank statements CSV file")
    parser.add_argument("input_file", help="Path to the input CSV file")
    parser.add_argument("-o", "--output", default="bank_statements_processed.csv",
                        help="Path for the output processed CSV file (default: bank_statements_processed.csv)")
    
    args = parser.parse_args()
    
    preprocess_bank_statements(args.input_file, args.output)
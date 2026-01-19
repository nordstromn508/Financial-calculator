import pandas as pd
import re
import sys

def identify_merchant(description):
    """
    Identify merchant from transaction description using pattern matching
    """
    # Convert description to lowercase for easier matching
    desc_lower = description.lower()
    
    # Common merchant patterns
    merchant_patterns = {
        'Grocery': ['grocery', 'market', 'supermarket', 'whole foods', 'safeway', 'kroger', 'walmart', 'target'],
        'Restaurant': ['restaurant', 'dining', 'cafe', 'bistro', 'grill', 'diner', 'steakhouse', 'pizza', 'taco', 'burger', 'chinese', 'italian'],
        'Gas Station': ['gas', 'shell', 'exxon', 'chevron', 'bp', 'mobil', 'speedway', 'circle k'],
        'Online Shopping': ['amazon', 'ebay', 'best buy', 'apple', 'netflix', 'spotify', 'itunes', 'google play'],
        'Entertainment': ['movie', 'cinema', 'theater', 'amc', 'disney', 'hulu', 'hbo'],
        'Healthcare': ['hospital', 'clinic', 'pharmacy', 'walgreens', 'cvs', 'doctor', 'dentist'],
        'Travel': ['airline', 'delta', 'southwest', 'united', 'hotel', 'marriott', 'airbnb', 'uber', 'lyft'],
        'Utilities': ['electric', 'gas company', 'water', 'comcast', 'att', 'verizon', 't-mobile'],
        'Insurance': ['insurance', 'geico', 'progressive', 'state farm', 'allstate'],
        'Education': ['college', 'university', 'school', 'textbook'],
        'Salary': ['salary', 'payroll', 'employer', 'work', 'income'],
        'Freelance': ['freelance', 'contract', 'consulting', 'independent', 'client']
    }
    
    for merchant_type, patterns in merchant_patterns.items():
        for pattern in patterns:
            if pattern in desc_lower:
                return merchant_type
    
    # If no pattern matches, return 'Other'
    return 'Other'

def add_merchant_column(input_file, output_file=None):
    """
    Add a merchant identification column to the processed CSV file
    """
    try:
        # Read the processed CSV file
        df = pd.read_csv(input_file)
        
        # Add the new Merchant column by analyzing the Description column
        df['Merchant'] = df['Description'].apply(identify_merchant)
        
        # Determine output filename
        if output_file is None:
            output_file = input_file.replace('.csv', '_with_merchants.csv')
        
        # Save the updated dataframe to a new CSV file
        df.to_csv(output_file, index=False)
        
        print(f"Successfully added merchant column to {input_file}")
        print(f"Output saved to {output_file}")
        print(f"Added {len(df)} rows with merchant identifications")
        
        # Print a summary of identified merchants
        print("\nMerchant type summary:")
        print(df['Merchant'].value_counts())
        
        return df
        
    except FileNotFoundError:
        print(f"Error: File {input_file} not found.")
        return None
    except Exception as e:
        print(f"Error processing file: {str(e)}")
        return None

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python add_merchant_column.py <input_file.csv> [output_file.csv]")
        sys.exit(1)
    
    input_file = sys.argv[1]
    output_file = sys.argv[2] if len(sys.argv) > 2 else None
    
    add_merchant_column(input_file, output_file)
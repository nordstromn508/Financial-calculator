import csv
import random
from datetime import datetime, timedelta

def generate_transaction_data(start_date, num_transactions=993):
    """Generate additional bank transaction data"""
    
    # Sample data for generating realistic transactions
    transaction_types = ['DEBIT', 'CREDIT']
    descriptions = [
        'Grocery Store', 'Gas Station', 'Restaurant', 'Online Purchase', 
        'Salary Deposit', 'Freelance Work', 'Investment Income', 'Tax Refund',
        'Rent Payment', 'Utility Bill', 'Insurance Premium', 'Credit Card Payment',
        'ATM Withdrawal', 'Bank Fee', 'Interest Earned', 'Medical Expense',
        'Education Expense', 'Entertainment', 'Travel Expense', 'Shopping'
    ]
    
    # Starting balance from the last transaction
    starting_balance = 2404.50
    
    with open('/workspace/bank_statements_raw.csv', 'a', newline='') as csvfile:
        writer = csv.writer(csvfile)
        
        # Generate 993 more transactions to reach 1000 total
        current_date = datetime.strptime(start_date, '%Y-%m-%d')
        current_balance = starting_balance
        
        for i in range(num_transactions):
            # Randomly select transaction type
            trans_type = random.choice(transaction_types)
            
            # Select a random description
            description = random.choice(descriptions)
            
            # Generate amounts based on transaction type
            if trans_type == 'CREDIT':
                credit_amount = round(random.uniform(20.0, 2000.0), 2)
                debit_amount = 0
                current_balance += credit_amount
            else:  # DEBIT
                debit_amount = round(random.uniform(5.0, 500.0), 2)
                credit_amount = 0
                current_balance -= debit_amount
            
            # Occasionally add a check number for debits
            check_number = f"{random.randint(1006, 9999):04d}" if trans_type == 'DEBIT' and random.random() > 0.7 else ""
            
            # Create transaction row
            transaction = [
                current_date.strftime('%Y-%m-%d'),
                '123456789',  # Bank RTN
                '987654321',  # Account Number
                trans_type,
                description,
                str(debit_amount),
                str(credit_amount),
                check_number,
                str(round(current_balance, 2))
            ]
            
            writer.writerow(transaction)
            
            # Move to next day occasionally (not every transaction)
            if i % 3 == 0:  # Every third transaction, advance date
                current_date += timedelta(days=1)

if __name__ == "__main__":
    generate_transaction_data('2023-08-14', 993)
    print("Added 993 transactions to reach 1000 total transactions.")
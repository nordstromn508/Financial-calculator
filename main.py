import subprocess
import sys
import os

def run_preprocessing():
    """Run the preprocessing script to clean the CSV file"""
    print("Running preprocessing...")
    try:
        result = subprocess.run([sys.executable, 'preprocess_csv.py'], 
                                capture_output=True, text=True, check=True)
        print(result.stdout)
        if result.stderr:
            print("Errors:", result.stderr)
    except subprocess.CalledProcessError as e:
        print(f"Preprocessing failed: {e}")
        print(f"Error output: {e.stderr}")
    except Exception as e:
        print(f"Unexpected error during preprocessing: {e}")

def run_dashboard():
    """Run the dashboard script"""
    print("Starting dashboard...")
    try:
        # Import and run the dashboard
        import dashboard
        dashboard.app.run(debug=True, port=8050)
    except ImportError as e:
        print(f"Could not import dashboard: {e}")
    except Exception as e:
        print(f"Dashboard error: {e}")

if __name__ == '__main__':
    # First run preprocessing
    run_preprocessing()
    
    # Then run the dashboard
    run_dashboard()
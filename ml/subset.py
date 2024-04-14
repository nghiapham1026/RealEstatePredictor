import pandas as pd

def load_and_subset_data(file_path, output_file_path, subset_size=10000, random_state=69):
    data = pd.read_csv(file_path)
    
    # Display basic information about the dataset
    print(data.info())
    
    # Create a random subset of the data
    subset_data = data.sample(n=subset_size, random_state=random_state)
    
    subset_data.to_csv(output_file_path, index=False)
    print(f"Subset saved to {output_file_path}")

# Example usage
file_path = '../realtor-data.zip.csv'
output_file_path = '../realtor-data_small.csv'
subset = load_and_subset_data(file_path, output_file_path)
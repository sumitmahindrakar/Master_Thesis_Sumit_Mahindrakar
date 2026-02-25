import pickle

# Load and print the pickle file
with open('test_files/PIGNN/PIGNN_V01/DATA/frame_dataset.pkl', 'rb') as f:
    data = pickle.load(f)

print(data)
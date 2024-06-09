import os
from datasets import Dataset, DatasetDict, Audio
import pandas as pd

def create_dataset(noisyspeech_dir):
    # List all noisy speech files
    noisy_files = [os.path.join(noisyspeech_dir, f) for f in os.listdir(noisyspeech_dir) if f.endswith('.wav')]

    # Create a dataframe with file paths and tags
    data = {'file': noisy_files, 'label': ['noisy_speech'] * len(noisy_files)}
    df = pd.DataFrame(data)

    # Convert dataframe to Hugging Face Dataset
    dataset = Dataset.from_pandas(df)

    # Define audio column
    dataset = dataset.cast_column("file", Audio())

    return dataset

def upload_dataset(dataset, dataset_name):
    # Create a DatasetDict
    dataset_dict = DatasetDict({"train": dataset})

    # Save to Hugging Face
    dataset_dict.push_to_hub(dataset_name)

if __name__ == "__main__":
    # Directory containing the noisy speech files
    noisyspeech_dir = 'NoisySpeech_training'

    # Dataset name on Hugging Face
    dataset_name = 'rfhuang/audio-quality'

    # Create dataset
    dataset = create_dataset(noisyspeech_dir)

    # Upload dataset
    upload_dataset(dataset, dataset_name)

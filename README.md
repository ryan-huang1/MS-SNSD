
# Noisy Speech Synthesizer

This repository contains a script to synthesize noisy speech data from clean speech and noise files. The script allows you to specify the number of hours of data to generate and the range of Signal-to-Noise Ratio (SNR) values.

## Prerequisites

- Python 3.x
- Required Python packages (install using `pip install -r requirements.txt`)

## Installation

1. Clone the repository:
    ```bash
    git clone <repository-url>
    cd <repository-directory>
    ```

2. Install the required packages:
    ```bash
    pip install -r requirements.txt
    ```

## Configuration

The script uses a configuration file (`noisyspeech_synthesizer.cfg`) to set various parameters. Make sure to update the configuration file as needed.

## Usage

To generate noisy speech data, run the following command:

```bash
python noisyspeech_synthesizer.py --cfg noisyspeech_synthesizer.cfg --total_hours <number_of_hours>
```

### Arguments

- `--cfg`: Path to the configuration file (default is `noisyspeech_synthesizer.cfg`).
- `--cfg_str`: Section in the configuration file to use (default is `noisy_speech`).
- `--total_hours`: Total hours of data to be created.

### Example

```bash
python noisyspeech_synthesizer.py --cfg noisyspeech_synthesizer.cfg --total_hours 100
```

This command will generate 100 hours of noisy speech data.

## Uploading to Hugging Face

To upload the generated noisy speech data to a Hugging Face dataset, use the following script:

```python
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
```

### Instructions

1. Ensure you are logged in to your Hugging Face account:
    ```bash
    huggingface-cli login
    ```

2. Run the script to upload the dataset:
    ```bash
    python upload_to_huggingface.py
    ```

Replace `'rfhuang/audio-quality'` with the appropriate dataset name on Hugging Face.

## License

This project is licensed under the MIT License.

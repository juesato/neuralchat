# Neural Conversational Model

A TensorFlow implementation of a sequence-to-sequence neural conversational model trained on the OpenSubtitles corpus.

## Overview

This project implements the neural conversational model described in [A Neural Conversational Model](http://arxiv.org/pdf/1506.05869.pdf) by Oriol Vinyals and Quoc V. Le (July 2015). The model uses a sequence-to-sequence architecture with LSTM layers to generate conversational responses.

## Background

This implementation is based on TensorFlow's seq2seq tutorial and adapts it for conversational modeling. The model learns to respond to conversational inputs by training on movie subtitle pairs from the OpenSubtitles dataset.

## Features

- Automatic download and preprocessing of OpenSubtitles corpus
- Configurable model architecture (layer size, number of layers, vocabulary size)
- Training with checkpoint saving
- Interactive decoding mode for chatting with the trained model
- Bucket-based batching for efficient training

## Requirements

- Python 2.7+ or Python 3.x
- TensorFlow
- NumPy
- Six

## Installation

1. Clone this repository:
```bash
git clone <repository-url>
cd neuralchat
```

2. Install dependencies:
```bash
pip install tensorflow numpy six
```

## Usage

### Training the Model

To train a new model from scratch:

```bash
cd src
python chatbot.py --data_dir ../data2/ --train_dir ../train_512_2/ --size 512 --num_layers 2 --max_train_data_size 1500000
```

**Training Parameters:**
- `--data_dir`: Directory for storing/loading the OpenSubtitles dataset
- `--train_dir`: Directory for saving model checkpoints
- `--size`: Size of each model layer (default: 4096)
- `--num_layers`: Number of LSTM layers (default: 2)
- `--vocab_size`: Vocabulary size (default: 100000)
- `--max_train_data_size`: Limit on training data size (0 = no limit)
- `--learning_rate`: Initial learning rate (default: 0.5)
- `--batch_size`: Batch size for training (default: 64)

On first run, the script will:
1. Download the OpenSubtitles corpus
2. Process and tokenize the data
3. Create training and validation sets
4. Build a vocabulary
5. Begin training

Training a model with 256 hidden units, 2 layers, and vocab size 40000 on 1M data points takes approximately 12 hours on a single GPU.

### Interactive Chat (Decoding)

To chat with a trained model:

```bash
cd src
python chatbot.py --data_dir ../data2/ --train_dir ../train_512_2/ --size 512 --num_layers 2 --decode
```

The `--decode` flag starts an interactive session where you can type messages and receive responses from the model.

**Note:** The checkpoint files are not included in this repository as they exceed GitHub's 100MB file size limit. You'll need to train your own model first.

## Project Structure

```
neuralchat/
├── src/
│   ├── chatbot.py       # Main training and decoding script
│   └── data_utils.py    # Data processing utilities
├── data2/               # Dataset directory (created on first run)
│   ├── train/          # Training data
│   ├── valid/          # Validation data
│   └── vocab*.en       # Vocabulary files
└── train_512_2/        # Model checkpoints (created during training)
```

## Model Architecture

The model uses a sequence-to-sequence architecture with:
- LSTM encoder-decoder layers
- Attention mechanism (from TensorFlow's seq2seq model)
- Bucketing for variable-length sequences: [(5,10), (10,15), (20,25), (40,50)]
- Greedy decoding for response generation

## Dataset

The model trains on the [OpenSubtitles](http://opus.lingfil.uu.se/) corpus, which contains movie and TV show subtitles. The data processing pipeline:
1. Downloads and extracts subtitle XML files
2. Parses consecutive utterances as input-output pairs
3. Tokenizes and creates vocabulary
4. Splits into 70% training / 30% validation sets

## Limitations

- Model checkpoints are not included due to file size constraints
- Training requires significant computational resources
- Response quality depends heavily on training duration and model size
- Uses greedy decoding (no beam search)

## Credits

- Based on TensorFlow's seq2seq translation tutorial
- Implements the approach from "A Neural Conversational Model" by Vinyals & Le
- Uses the OpenSubtitles corpus for training data

## License

This code is a modification of Google's TensorFlow translation tutorial, licensed under Apache License 2.0.

## References

- [A Neural Conversational Model](http://arxiv.org/pdf/1506.05869.pdf) - Vinyals & Le, 2015
- [TensorFlow Sequence-to-Sequence Tutorial](https://www.tensorflow.org/tutorials/seq2seq)
- [OpenSubtitles Corpus](http://opus.lingfil.uu.se/)


# A NER approach using a LSTM/CRF neural network approach

In this notebook, I implemented the neural model described in the paper [End-to-end Sequence Labeling via Bi-directional LSTM-CNNs-CRF](https://www.aclweb.org/anthology/P16-1101.pdf). This model benefits from both word- and character-level representations automatically, by using combination
of bidirectional LSTM (Bi-LSTM), CNN and CRF.

The first implementation rely on Keras/Tensorflow packages.

## Dependencies
See `environment.yml`. In general, I used `tensorflow.keras` and `scikit-learn` for my ML experiments :crystal_ball:.

## Setup
```bash
conda env create -f environment.yml
conda activate ner_lstm_crf_wikiner
```

## Project Structure
* `data/`: directory in which are saved all the dataset used in the notebooks. The dataset are:
    * [WikiNER](https://github.com/dice-group/FOX/tree/master/input/Wikiner);
* `environment.yml`: conda environment file in order to replicate the environment on your machine and reproduce the experiments I made.

## References
* [This repo](https://github.com/napsternxg/DeepSequenceClassification/blob/master/model.py) for implementing in Keras character-level embedding with CNN and word-level embedding with Glove;
* *X. Ma, E.Hovy*; [End-to-end Sequence Labeling via Bi-directional LSTM-CNNs-CRF](https://www.aclweb.org/anthology/P16-1101.pdf); for the suggestions about the model structure and hyperparameters;
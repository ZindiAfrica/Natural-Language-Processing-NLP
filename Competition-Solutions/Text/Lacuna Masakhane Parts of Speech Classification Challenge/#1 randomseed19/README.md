# Lacuna Masakhane Parts of Speech Classification Challenge


# [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ashatilov/zindi_masakhane_pos/blob/master/train.ipynb)

## Data

Only competition data from repo https://github.com/masakhane-io/masakhane-pos is used.

## Model

Pretrained NLLB-200 is an encoder-decoder multilanguage translation model (https://huggingface.co/facebook/nllb-200-distilled-600M).

We use only encoder part of it for POS tagging task, and define `M2M100ForTokenClassification` class, it consist of encoder part on NLLB model and token classification head on top of it. Class is defined here [masakhane_pos/m2m_100_encoder/modeling_m2m_100.py](masakhane_pos/m2m_100_encoder/modeling_m2m_100.py#L27)

## Training

LoRA approach is used for model fine-tuning.

All data processing and utility functions are defined in [masakhane_pos/utils.py](masakhane_pos/utils.py#L14).

Training script: [train.py](train.py#L64)

Notebook with full code to reproduce solution: [train.ipynb](train.ipynb) - open and run it in Google Colab, full training takes about 30 minutes.
 [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ashatilov/zindi_masakhane_pos/blob/master/train.ipynb)

## Finding the best solution process

Firstly, iteratively perform a greedy search for the best set of languages to train on based on the public score:

1. Train on a set of 1 language, validate on all the others, and find the one that gives the best public score.
2. Train on a set of 2 languages: the best from the previous step (lug) and iteratively select all others to find the best set based on the public score.
3. Train on a set of 3 languages: the best set from the previous step (lug, ibo) and iteratively select all others to find the best set based on the public score.
4. And so on.

The final set of languages that was used in the final submission is lug, ibo, mos, sna. A set of 5 languages gave a slightly worse result than a set of 4.

Secondly, using pseudo labels:

1. Train the model on the found best set of languages (lug, ibo, mos, sna) - it gave ~0.728 public score.
2. Predict labels for luo and tsn using the model from the previous step, and add them as training data. This increased the public score to ~0.742.

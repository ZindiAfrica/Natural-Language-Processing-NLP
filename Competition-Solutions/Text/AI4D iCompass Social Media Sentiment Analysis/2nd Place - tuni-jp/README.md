# Solution Summary

All models are trained in StratifiedKfold at 5fold.

## Model and Setting

The pre-trained model is trained with the following settings

- maxlen 198
- Domain-Adaptation
    - train+test MLM
- Label-smoothing
- dynamic padding
    - Padding during training can speed up the training.
    - In addition, we sorted by token length. This makes it even faster.
        - There was almost no degression in sorting.
- Colab P100 or V100

### Arabic-Bert (aubmindlab/bert-base-arabertv02)

cv: 0.809

- I used the translate script that was shared in the discussion
    - [https://zindi.africa/competitions/ai4d-icompass-social-media-sentiment-analysis-for-tunisian-arabizi/discussions/5256](https://zindi.africa/competitions/ai4d-icompass-social-media-sentiment-analysis-for-tunisian-arabizi/discussions/5256)
- The accuracy of the translation is not very good (my teammate confirmed this). But we can get the same or slightly better results than Vanilla-BERT

### Vanilla-BERT (bert-base-uncased)

cv: 0.805

- I tried the multilingual model, but the accuracy was almost the same. Therefore, we chose the lighter model.
- distillation model was a little underperforming.

We cannot use more than two pre-trained models (1GB is the limit).

However, we knew that the ensemble would score better and be more robust, so we used a traditional model that could be trained with only the given data set and was fast.

### LSTM

cv: 0.799

- sentencepiece (vocab 8000)
    - Subword Regularization
- gensim word2vec pretrain (train+test) (dim=300)

### Fasttext (lightgbm)

cv: 0.802

- fastext pretrain (train+test) (dim=300)
- stopword remove (en+fr)
- lightgbm train

### Catboost

cv: 0.789

- text_features=["text"]
    - no preprocessing

### MultinominalNB

cv: 0.795

- tfidf vectorizer
    - no preprocessing

## Ensemble
- Stacking the 1st model with oof
- 0.5 * Logistic Regression + 0.5 * lightgbm

## Didn't work

- French-BERT
    - I heard that Arabizi uses more French than English, but it was worse than Vanilla-BERT.
- Vocabulary Augmentation
- Pseudo Labeling
- MultiSample Dropout
- Predict long sentences separately
- Predict neutral separately

## Training Time, Resources

1. Arabic-BERT
    - pertaining model size 516M
    - execution time about < 2h
2. Vanilla-BERT
    - pertaining model size 418M
    - execution time about < 2h
3. fasttext+lightgbm
    - execution time about < 15min
4. LSTM
    - execution time about < 30min
5. Catboost
    - execution time about < 15min
6. MultinominalNB
    - execution time about < 5min

It meets all the rules.

- no external data
- total pretraining model size 934M < 1GB
- total execution time about ~5h < 10h(training+inferance time)

Finally

submissions was optional during the competition. This change is generally welcomed, as it excludes models that happen to fit the test. However, I don't think the rule should be changed during the competition.
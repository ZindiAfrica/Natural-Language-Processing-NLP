# Competition Summary

## Description

Part-of-speech (POS) tagging is a crucial step in natural language processing (NLP), as it allows algorithms to understand the grammatical structure and meaning of a text. This is especially important in creating the building blocks for preparing low-resource African languages for NLP tasks. The MaseakhaPOS dataset for 20 typologically diverse African languages, including benchmarks, was created with the help of Lacuna Fund to try and address this problem.

The objective of this challenge is to create a machine learning solution that correctly classifies 14 parts of speech for the unrelated Luo and Setswana languages. You will need to build one solution that applies to both languages, not two solutions, one for each language.

It is important that only one solution be built for both languages as this is a step in creating a solution that can be applied to many different languages, instead of having to create a model for each language.



## Competition Rules

Participation in this competition could be as an individual or in a team of up to four people.

Prizes are transferred only to the individual players or to the team leader.

Code was not shared privately outside of a team. Any code shared, was made available to all competition participants through the platform. (i.e. on the discussion boards).



## Datasets and packages

The training set of 19 languages is available at this repo: https://github.com/masakhane-io/masakhane-pos

Use this starter notebook to get started: https://github.com/masakhane-io/masakhane-pos/blob/main/train_pos.ipynb

The test set contains 17 parts of speech from Luo and 17 parts of speech from Setswana. Both these languages are unseen in the training set.

You can read more about the dataset and some idea that have worked in the past in this paper (https://arxiv.org/pdf/2305.13989.pdf). However, you are encouraged to come up with your own methods.

The solution must use publicly-available, open-source packages only.

You may use only the datasets provided for this competition. Automated machine learning tools such as automl are not permitted.

You may use pretrained models as long as they are openly available to everyone.

The license of the POS dataset is in CC-BY-4.0-NC, the monolingual data have difference licenses depending on the news website license.


## Submissions and winning

The top 3 solution placed on the final leaderboard were required to submit their winning solution code to us for verification, and thereby agreed to assign all worldwide rights of copyright in and to such winning solution to Zindi.



## Reproducibility

The full documentation was retrieved. This includes:
- All data used

- Output data and where they are stored

- Explanation of features used

- The solution must include the original data provided by Zindi and validated external data (no processed data)

- All editing of data must be done in a notebook (i.e. not manually in Excel)



## Data standards:

- The most recent versions of packages were used.

- Submitted code run on the original train, test, and other datasets provided.



## Evaluation:

The error metric for this competition is Accuracy.

For every row in the dataset, submission files should contain 2 columns: ID and Target.

There are 17 parts of speech for both Luo and Setswana.

Your submission file should look like this:

Id	        Pos
Id1698c74uqq_0	NOUN
Id1698c74uqq_1	ADP



## Prizes

1st place: $3 500 USD

2nd place: $2 100 USD

3rd place: $1 400 USD



## Benefits

This challenge is also important for Lacuna Fund, to help reach their goals of making ML-ready datasets available from low- and middle-income contexts.


[![CC BY 4.0][cc-by-shield]][cc-by]

This work is licensed under a
[Creative Commons Attribution 4.0 International License][cc-by].

[![CC BY 4.0][cc-by-image]][cc-by]

[cc-by]: http://creativecommons.org/licenses/by/4.0/
[cc-by-image]: https://i.creativecommons.org/l/by/4.0/88x31.png
[cc-by-shield]: https://img.shields.io/badge/License-CC%20BY%204.0-lightgrey.svg



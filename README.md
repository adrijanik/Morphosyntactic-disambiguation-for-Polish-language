### Morphosyntactic disambiguer for Polish language
#### POS tagging - morphosyntactic disambiguation and guessing for Polish language [PoleVal - Task 1A, http://poleval.pl/]
Solution is based on pretrained words embeddings coming from: http://publications.ics.p.lodz.pl/2016/word_embeddings/ generated using skip-gram method.

##### to install package:
```
$ git clone https://github.com/adrijanik/Morphosyntactic-disambiguation-for-Polish-language.git
$ cd Morphosyntactic-disambiguation-for-Polish-language
$ sudo pip install .
```
##### to run analysis on xcef cesAna data:
```
$ from morphosyntactic_disambiguer import pipeline
$ data_flow = pipeline.Pipeline()
$ data_flow.run('train_data.xml','gold_data.xml')
```
| Accuracy  | Train | Test
| ------------- | ------------- | ----- |
|   | 72% | 56%  |

# Rumour Detection via Zero-shot Cross-lingual Transfer Learning

## Paper
L. Tian, X. Zhang, and J.H. Lau (2021). Rumour Detection via Zero-shot Cross-lingual Transfer Learning. In Proceedings of The European Conference on Machine Learning and Principles and Practice of Knowledge Discovery in Databases (ECML PKDD 2021), pages 603—618.

##  Data and Resource
We use Twitter15/Twitter16, PHEME and WEIBO datasets for rumour detection.
The five-fold split is from [here](https://github.com/majingCUHK/Rumor_RvNN/tree/master/nfold)

In this repository, we do not provide you with the raw input data. Please download the datasets from the following links.

| Dataset | Link |
| --- | --- |
| T15/T16 | https://www.dropbox.com/s/7ewzdrbelpmrnxu/rumdetect2017.zip?dl=0 ) |
| PHEME | https://figshare.com/articles/dataset/PHEME_dataset_of_rumours_and_non-rumours/4010619 |
| WEIBO | https://alt.qcri.org/~wgao/data/rumdect.zip |
| Extra Chinese Pretraining Data | https://archive.ics.uci.edu/ml/datasets/microblogPCU |
| Extra English Pretraining Data | https://github.com/KaiDMML/FakeNewsNet |

## Dependencies
1. Python 3.6
2. Run `pip install -r requirements.txt`

## Adaptive Pretraining script:

Clone the transformer to local directory
```
git clone https://github.com/huggingface/transformers.git
```


For further pre-training the language models:
```
python transformers/examples/language-modeling/run_language_modeling.py ,
        --output_dir='ML_MBERT_DAPT',
        --model_type=bert ,
        --model_name_or_path=bert-base-multilingual-cased-freeze-we,
        --do_train,
        --overwrite_output_dir,
        --train_data_file='train.txt',
        --do_eval,
        --block_size=256,
        --eval_data_file='vali.txt',
        --mlm"
```




#If you find this code useful, please let us know and cite our paper.

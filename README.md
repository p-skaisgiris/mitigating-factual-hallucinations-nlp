# dl4nlp-text-summarization
Factual Hallucination Metrics for NLG Evaluation

## Table of Contents
- [Introduction](#introduction)
- [Methodology](#methodology)
- [Dataset](#dataset)
- [Models](#models)
- [Hallucination Metrics](#hallucination-metrics)

## Introduction

Recent advancements in Natural Language Generation (NLG) have improved the fluency and coherence of NLG outputs in tasks such as summarization and dialogue generation. However, these models are prone to generate content that is nonsensical, grammatically incorrect, or irrelevant to the topic, and are known as "hallucinations." Hallucinations, particularly factual inaccuracies, pose serious consequences such as spreading misinformation and violating privacy. To address this challenge, researchers have explored various measurement and mitigation methods. This paper provides an ensemble of metrics to measure whether the generated text is factually correct.

## Methodology

The methodology employed in this paper involves the use of various metrics to assess and address hallucinatory content effectively. In addition to traditional metrics like ROUGE, the following metrics are utilized:

- **QAGS (Question-Answering for Factual Consistency)**: This metric generates questions about the generated summary and evaluates factual consistency by comparing the generated answers to the expected ones.

- **BLEURT (Context-Aware Metric)**: BLEURT surpasses traditional metrics like BLEU and ROUGE by employing pre-trained transformers to gauge similarity between generated and reference text, capturing nuances in quality such as fluency and coherence.

- **FACT (Triple Relation-Based Metric)**: FACT leverages pre-trained models to extract factual triples from both the source document and the summary. Its output is a ratio of how many triples extracted from the summary are also found in the source document.

- **SUMMAC (Sentence-Level Metric)**: SUMMAC breaks down source documents and summaries into sentences and computes entailment probabilities between document and summary sentences using Natural Language Inference (NLI).

## Dataset

The XSum (Extreme Summarization) dataset is used for experimentation. This dataset consists of approximately 226,000 news articles from the BBC News website, each accompanied by a single-sentence summary. The summaries were written by professional editors and are considered to be high-quality references.

## Models

For most of the experiments, the T5 language model is used. Specific model variants include:
- **t5-small**: The smallest version of the t5 model.
- **t5-small-xsum**: The small version of the t5 model, fine-tuned on the XSUM dataset.
- **t5-large**: The t5 model with 770 million parameters.
- **t5-large-xsum**: The large version of t5 fine-tuned on XSum.
- **t5-large-xsum-cnn**: Based on the t5-large model, finetuned on the XSUM and CNN Daily Mail summarization datasets.

## Conclusion

This paper presents a comprehensive approach to evaluate and mitigate factual hallucinations in NLG. By utilizing an ensemble of metrics, analyzing different language models, and exploring various methods for mitigating hallucinations, the paper aims to contribute to the understanding and improvement of NLG systems.

For more details, refer to the complete paper. **LINK PAPER**



## Environment

```
conda create -n DL4NLP python=3.10
conda activate DL4NLP
pip install -r requirements.yml
```

Contain hallucationa metric such as factsumm and summac. See notebookon 'hallucination_metrics.ipynb' how to install and bug fix required for factsumm. In order to make use of GPU training do not use pip install factsumm, but clone from the original repository. For factsumm to get the package dir. Open python in terminal

    > import factsumm
    > factsumm.__file__


## Training
(See train.py for more arguments)

```
python train.py --wandb-mode disabled
```

## Hallucination Evaluation
(See train.py for more arguments)

```
python eval.py 
```

## Human judgment comparison 
Comparing our halluciniation metrics with human judgment of hallucination in XSUM using [google-research-dataset](https://github.com/google-research-datasets/xsum_hallucination_annotations/tree/master)

```
python human_judgement.py 
```

## Model checkpoints
Note however, that these model were only trained as a sanity check were not u
Bart-base and T5-large checkpoints can be found here: [checkpoints](https://drive.google.com/drive/folders/1IF9n4bljNzDUlAX7U54LgtBZWko8z0jO?usp=sharing)

## About
Paulius, Myrthe, Erencan, Luka

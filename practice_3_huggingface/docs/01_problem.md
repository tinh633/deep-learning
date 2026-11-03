# 01 — Problem statement, objectives and acceptance tests

## Scope

In this practice, we solve two connected but distinct NLP tasks. Exercise 1 applies a sentiment model trained by someone else, verifying the **tokenization → forward inference → sentiment label** sequence. Exercise 2 adapts a general pretrained language model to a labeled dataset through the **tokenization → Trainer fine-tuning → held-out evaluation** sequence. Both are binary English sentiment tasks, but their training status differs.

## Input/output definition

- Input: a Unicode English review sentence or full movie review.
- Output class encoding: 0 = NEGATIVE, 1 = POSITIVE.
- Ex1 evidence: tokenizer tokens, input IDs and attention mask, model identifier, predicted label and reported score.
- Ex2 evidence: dataset provenance, train/validation split, model/hyperparameters, epoch logs, model weights and independent final metrics.
- The score from a softmax-based classifier is a model probability estimate, not calibrated evidence of external truth.

## Success criteria

**Ex1:** code downloads/loads a Hub classifier and executes an unseen sample without training. A neutral or sarcastic sentence can still force a binary prediction; this does not imply the model understands all nuanced opinions.

**Ex2:** code obtains official IMDb dataset splits, makes independent stratified train/validation subsets of the train partition, trains a sequence classifier through Trainer, selects checkpoint using validation F1 and reports final reserved-test results without test-driven selection. The assignment does not require a particular accuracy threshold.

## Connection to Practice 1/2

Practice 1 used FashionMNIST pixels → CNN features → class logits; Practice 2 transferred pretrained image features to CIFAR-10. Practice 3 uses **tokens → Transformer encoder embeddings → classification logits**. The same supervised-learning principles apply: loss, gradient descent, overfitting, data leakage and split discipline. A pretrained Transformer usually learns language features before the task-specific labels are available.

## What is out of scope

Multiclass sentiment, multilingual benchmarking, neutral-class modeling, LLM prompt classification, hyperparameter grid search, production API and benchmark competition. Do not confuse successful code execution with proven real-world reliability.

## Terminology

- **Hub repository:** versioned location for model/tokenizer/dataset assets.
- **Tokenizer:** deterministic mapping from text to tokens and IDs (given configuration).
- **Forward pass:** computation of logits from IDs/attention masks, without updating weights.
- **Fine-tuning:** training with task labels to update a pretrained model's trainable parameters.
- **Trainer:** HF training loop orchestrator (forward/loss/backward/optimizer/logging).
- **Held-out test:** data not consulted for training or validation selection.

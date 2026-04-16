# Ambirig: Ordinal Distributional Modelling for Word Sense Plausibility

This repository contains the code for **Ambirig**, our system for **SemEval-2026 Task 5: Word Sense Plausibility Estimation**.

We model plausibility as an **ordinal distribution prediction problem** and optimise using **Earth Mover’s Distance (EMD)** to better align with human annotator disagreement.

## Overview

Traditional Word Sense Disambiguation (WSD) assumes a single correct sense. However, in narrative contexts, multiple interpretations can remain plausible.

Our approach:
- Predicts a **distribution over Likert scores (1–5)** instead of a single label
- Uses a **GlossBERT-style cross-encoder**
- Optimises an **ordinal loss (EMD)** instead of cross-entropy

## Model

- Backbone: `microsoft/deberta-v3-base`
- Architecture: Cross-encoder (context + gloss)
- Output: 5-class probability distribution
- Loss: **Squared Earth Mover’s Distance (EMD)**

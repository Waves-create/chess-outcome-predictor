# ♟️ chess-outcome-predictor

A machine learning project that predicts the outcome of a chess game — **White win or Black win** — using historical match data and Python-based analysis.

---

## Overview

This project focuses on understanding how chess game progression and player ratings influence match outcomes.  
By varying the number of moves considered (`max_moves`), the model analyzes how prediction accuracy improves as more game information becomes available.

The goal is not deployment, but **data analysis, feature experimentation, and model evaluation**.

---

## Tech Stack

- Python  
- pandas, numpy  
- scikit-learn  
- python-chess  
- matplotlib  

---

## Project Structure
```
chess-outcome-predictor/  
├── data/  
│   └── games.csv  
├── train.py        # model training and evaluation  
├── train1.py       # accuracy vs max_moves experiment  
├── requirements.txt  
└── README.md 
```
---

## Setup

Install dependencies:
```bash
pip install -r requirements.txt
```
---

## Usage

Train the model:
```bash
python train.py
```
Run experiments for different max_moves values and generate accuracy graph:
```bash
python train1.py
```
---

## Results

Model accuracy increases as more moves are considered:

- 20 moves → ~0.68  
- 40 moves → ~0.74  
- 80 moves → ~0.85  
- 150 moves → ~0.87  

This indicates that later stages of a chess game provide stronger signals for predicting the final outcome.

---

## Features Used

- Number of moves considered  
- White player rating  
- Black player rating  
- Game outcome labels (White / Black)

---

## Purpose

This project was built as a learning-focused machine learning analysis to strengthen skills in:
- Data preprocessing  
- Feature engineering  
- Model evaluation  
- Experiment-driven ML development  

---

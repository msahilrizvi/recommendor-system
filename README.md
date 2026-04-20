# 🎬 Movie Recommendation System

## 📌 Overview
This project builds an end-to-end recommendation system that suggests movies to users based on past interactions. It includes data preprocessing, collaborative filtering, evaluation, and API-ready architecture.

---

## 🎯 Problem Statement
Given a user's past interactions, recommend top-N movies they are likely to enjoy.

---

## 📂 Dataset
- MovieLens 100K dataset
- Contains user-item interactions with ratings

---

## 🧠 Approach

### 1. Data Preprocessing
- Filtered users and items (where applicable)
- Created train-test split per user

### 2. Baseline Model
- Popularity-based recommendation
- Recommends globally popular items

### 3. Collaborative Filtering
- Item-based collaborative filtering
- Cosine similarity between item vectors

---

## 📊 Evaluation

Metrics used:
- Precision@K
- Recall@K

### Results:
- Precision@10: ~0.10
- Recall@10: ~0.06

---

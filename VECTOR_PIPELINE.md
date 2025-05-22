# Vector-Based Malware Detection Pipeline
This document outlines a simple workflow for detecting malicious files using feature embeddings and a vector database like Pinecone.

## Pipeline Steps
1. **악성/정상 샘플 수집** - Gather both malicious and benign samples for training.
2. **Feature 추출 및 벡터 임베딩** - Extract features from each sample and convert them into vector embeddings.
3. **Pinecone 저장 (벡터 인덱스)** - Store these embeddings in a Pinecone index for fast similarity search.
4. **신규 페이로드 유입 → 동일 임베딩** - When a new payload arrives, apply the same feature extraction and embedding process.
5. **Pinecone 유사도 검색** - Query Pinecone to find similar embeddings.
6. **임계치 초과 시 탐지 및 알림** - If the similarity score exceeds a threshold, flag the payload and generate an alert.

This approach allows quick comparison of new samples against known data. Adjust the threshold based on validation results to balance detection rate and false positives.

# Spotify Music Recommender

A content-based song recommender over 114,000 Spotify tracks. Rather than comparing every track against every other, it partitions the catalog with K-means and searches only within a cluster using an approximate nearest-neighbor index, then re-ranks the shortlist with exact cosine similarity.

**Stack:** Python · pandas · scikit-learn · Annoy · matplotlib/seaborn · Jupyter

---

## Problem

Recommending by audio similarity is a nearest-neighbor problem, and the naive version doesn't scale: an exact similarity matrix over 114K tracks is roughly 13 billion pairs. This project builds a recommender that keeps similarity search tractable by shrinking the search space twice — first by dimensionality, then by cluster — and only paying for exact similarity on a small candidate set.

## Data

[Spotify Tracks Dataset](https://www.kaggle.com/datasets/maharshipandya/-spotify-tracks-dataset) (Kaggle) — 114,000 tracks across 21 columns, including 12 numeric audio features (danceability, energy, key, loudness, mode, speechiness, acousticness, instrumentalness, liveness, valence, tempo, time signature).

Audio features had no missing values. Features were min-max scaled before dimensionality reduction, since the raw columns span wildly different ranges (tempo in BPM vs. mode as a binary).

## Approach

**1. Dimensionality reduction.** PCA on the scaled audio features. The variance curve is gradual — the first two components capture 52%, and 8 components are needed to reach 95%:

| Components | 1 | 2 | 4 | 6 | 8 |
|---|---|---|---|---|---|
| Cumulative variance | 30.5% | 52.3% | 78.3% | 90.8% | 95.8% |

**2. Clustering.** MiniBatchKMeans with k=6, chosen from the elbow curve. The initial fit was badly unbalanced, so oversized clusters (>1.5× the ideal size) were re-partitioned into subclusters, yielding 7 clusters of more comparable size.

**3. Approximate search.** A separate Annoy index per cluster, built on the first two principal components. Four distance metrics were compared (cosine, euclidean, manhattan, dot); cosine was selected. Final indices use 50 trees.

**4. Exact re-ranking.** For a query song, the Annoy index returns 3× the requested number of candidates from within its cluster; those candidates are then scored with exact cosine similarity on the full audio-feature vectors, and the top N are returned.

## Results

| Measure | Value |
|---|---|
| PCA components retained (95% variance) | 8 |
| Clusters after rebalancing | 7 |
| Silhouette score (on PCA-reduced features) | 0.202 |
| Intra-list cosine similarity, sample query | 0.994 |

**Cluster structure is weak but real.** A silhouette score of 0.202 says the clusters are only modestly separated — unsurprising, since audio features form a continuum rather than discrete genres. That's acceptable here because the clusters are used to prune the search space, not to serve as meaningful categories in themselves.

**The retrieved songs are acoustically near-identical, and that's the interesting failure.** Querying "Blinding Lights" returns tracks with 0.994 mean pairwise similarity — but they span Mexican rock, thrash metal, breakbeat, and J-rock. The pipeline does exactly what it was built to do and still produces recommendations no listener would want, which is the real lesson: audio-feature proximity is not musical similarity. Nothing in the feature set encodes genre, era, language, or artist, so the model has no way to know that acoustically adjacent tracks can be culturally unrelated.

**Two data issues surfaced through the recommendations.** The same track appeared twice in one result set — the dataset carries one row per track–genre pairing, so tracks tagged with multiple genres are duplicated and can be recommended more than once. Deduplication on `track_id` would fix this. Separately, the 0.994 figure is inflated: cosine similarity on min-max scaled, all-positive vectors compresses toward 1, so it's useful for comparing candidates against each other but shouldn't be read as an absolute quality score.

## Limitations and next steps

- **Indexing on 2 of 8 components.** The Annoy indices are built on `pca1`/`pca2` only, which together hold 52% of variance — the retrieval stage is discarding information the re-ranking stage then uses. Rebuilding indices on all 8 components is the highest-value fix.
- **The hit-rate metric doesn't discriminate.** The metric comparison scores each query as `neighbors_returned / 5`, which is 1.0 whenever the index returns a full set, so it can't actually separate cosine from euclidean. A ranking metric against held-out similar pairs would be needed to justify the metric choice.
- **No genre or metadata signal.** Adding `track_genre` as a constraint or a hybrid re-ranking term would address the cross-genre problem directly.
- **No user data.** This is purely item–item; there's no personalization, and no way to learn from what anyone actually listened to.

## Repo structure

```
├── Final_Project.ipynb    # end-to-end analysis
└── README.md
```

## Running it

```bash
pip install pandas numpy scikit-learn annoy matplotlib seaborn plotly
jupyter notebook Final_Project.ipynb
```

The dataset is pulled via `kagglehub` in the first cell and requires Kaggle credentials.

## Context

Final project for DTSA 5510 (Unsupervised Algorithms in Machine Learning), M.S. Data Science, University of Colorado Boulder.

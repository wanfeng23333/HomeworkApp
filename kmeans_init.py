import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler

df = pd.read_csv("assistments.csv", encoding="latin1", low_memory=False)
df = df.head(3000)

grouped = df.groupby("user_id").agg({
    "correct": "mean",
    "attempt_count": "mean",
    "hint_count": "mean",
    "problem_id": "count"
}).reset_index()

grouped.columns = ["user_id", "accuracy", "avg_attempt", "avg_hint", "task_count"]

scaler = MinMaxScaler()
grouped["completion"] = scaler.fit_transform(grouped[["task_count"]])

grouped["efficiency"] = 1 / (1 + grouped["avg_attempt"] + grouped["avg_hint"])

data = grouped[["accuracy", "completion", "efficiency"]].dropna()

kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
kmeans.fit(data)

print(kmeans.cluster_centers_)
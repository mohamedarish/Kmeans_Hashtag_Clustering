from pyspark.sql import SparkSession

spark = SparkSession.builder.appName("TFIDF-KMeans-Clustering").getOrCreate()

data = (
    spark.read.option("header", "false")
    .csv("assets/tester.csv")
    .toDF("tweets")
)
data = data.na.fill("")

from pyspark.ml.feature import Tokenizer, Word2Vec

tokenizer = Tokenizer(inputCol="tweets", outputCol="words")
wordsData = tokenizer.transform(data)

word2Vec = Word2Vec(
    vectorSize=100, minCount=0, inputCol="words", outputCol="features"
)
model = word2Vec.fit(wordsData)
result = model.transform(wordsData)

from pyspark.sql.types import DoubleType
from pyspark.sql import functions as F
from pyspark.ml.feature import ElementwiseProduct
from pyspark.ml.linalg import DenseVector

unitVector = DenseVector([1.0] * 100)
dot_udf = F.udf(lambda x: float(DenseVector(x).dot(unitVector)), DoubleType())

# Calculate the dot product and cosine similarity
dotProduct = ElementwiseProduct(
    scalingVec=unitVector, inputCol="features", outputCol="dotProduct"
)
result = dotProduct.transform(result)

result = result.withColumn("cosineSimilarity", dot_udf("dotProduct"))

from pyspark.ml.clustering import KMeans

kmeans = KMeans(k=69, featuresCol="features", predictionCol="cluster")
model = kmeans.fit(result)
result = model.transform(result)

import matplotlib.pyplot as plt
import pandas as pd

# Collect the necessary data to plot
local_result = result.select("features", "cluster").toPandas()

# Extract the X and Y coordinates for plotting (assuming vectorSize=2)
local_result["x"] = local_result["features"].apply(lambda v: v[0])
local_result["y"] = local_result["features"].apply(lambda v: v[1])

# Plot the clusters
plt.figure(figsize=(10, 6))
for cluster_id in range(69):  # Adjust based on your actual number of clusters
    cluster_data = local_result[local_result["cluster"] == cluster_id]
    plt.scatter(
        cluster_data["x"], cluster_data["y"], label=f"Cluster {cluster_id}"
    )

plt.title("K-means Clustering")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.legend()
plt.show()

import networkx as nx
from pyspark.sql.window import Window
from pyspark.sql.functions import desc

# Assuming result is the DataFrame obtained after K-means clustering
clusters = (
    result.select("cluster").distinct().rdd.flatMap(lambda x: x).collect()
)

for cluster_id in clusters:
    print(cluster_id)
    # Select tweets in the current cluster
    cluster_data = result.filter(result["cluster"] == cluster_id)

    # Extract tweet IDs and cosine similarities
    similarity_df = cluster_data.select("cosineSimilarity").withColumn(
        "tweet_id", F.monotonically_increasing_id()
    )
    windowSpec = Window.orderBy(desc("cosineSimilarity"))
    similarity_df = similarity_df.withColumn(
        "rank", F.dense_rank().over(windowSpec)
    )
    similarity_df = similarity_df.select("tweet_id", "cosineSimilarity")
    # .filter("rank <= 5")  # You can adjust the number of top similarities to show

    # Convert the Spark DataFrame to Pandas
    similarity_pd = similarity_df.toPandas()

    # Plot the similarity graph using networkx
    G = nx.Graph()

    # Add nodes
    for row in similarity_pd.itertuples():
        G.add_node(
            row.tweet_id,
            label=f"{row.tweet_id}\nSimilarity: {row.cosineSimilarity:.4f}",
        )

    # Add edges
    for i in range(len(similarity_pd)):
        for j in range(i + 1, len(similarity_pd)):
            G.add_edge(
                similarity_pd.at[i, "tweet_id"],
                similarity_pd.at[j, "tweet_id"],
                weight=similarity_pd.at[i, "cosineSimilarity"],
            )

    # Plot the graph
    pos = nx.spring_layout(G)  # You can use other layout algorithms as well
    labels = nx.get_edge_attributes(G, "weight")
    edges = G.edges()
    weights = [G[u][v]["weight"] for u, v in edges]

    plt.figure(figsize=(12, 12))
    nx.draw_networkx_nodes(G, pos, node_size=100, node_color="skyblue")
    nx.draw_networkx_edges(
        G, pos, edgelist=edges, width=weights, edge_color="gray"
    )
    nx.draw_networkx_labels(
        G, pos, font_size=8, font_color="black", font_family="sans-serif"
    )
    nx.draw_networkx_edge_labels(G, pos, edge_labels=labels, font_color="red")

    plt.title(f"Similarity Graph - Cluster {cluster_id}")
    plt.show()

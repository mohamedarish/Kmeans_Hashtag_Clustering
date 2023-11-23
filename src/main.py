from pyspark.sql import SparkSession
from pyspark.sql.functions import col, explode, split

spark = SparkSession.builder.appName("WordOccurrence").getOrCreate()

csv_file_path = "assets/test.csv"
df = spark.read.option("header", "false").csv(csv_file_path)

print(df.count())

words_df = df.withColumn("words", split(col("_c2"), " "))

exploded_df = words_df.select(
    col("_c0").alias("ID"), explode(col("words")).alias("word")
)

word_occurrences = exploded_df.groupBy("word").count().orderBy("count")

word_occurrences.show(word_occurrences.count(), truncate=False)

print(word_occurrences.count())

spark.stop()

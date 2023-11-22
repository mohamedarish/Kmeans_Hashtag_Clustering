from pyspark.sql import SparkSession
from pyspark.sql.functions import col, explode, split

# Create a Spark session
spark = SparkSession.builder.appName("WordOccurrence").getOrCreate()

# Read the CSV file into a Spark DataFrame
csv_file_path = "assets/init_data.csv"
df = spark.read.option("header", "false").csv(csv_file_path)

# Extract the words from the 4th column using the split function
words_df = df.withColumn("words", split(col("_c4"), " "))

# Explode the array of words into separate rows
exploded_df = words_df.select(
    col("_c0").alias("ID"), explode(col("words")).alias("word")
)

# Group by the word and count its occurrences
word_occurrences = exploded_df.groupBy("word").count().orderBy("count")

# Show the results
word_occurrences.show(word_occurrences.count(), truncate=False)

print(word_occurrences.count())

# Stop the Spark session
spark.stop()

from pyspark.sql import SparkSession
from pyspark.sql.types import StructType
from pyspark.sql.functions import explode, split

spark = SparkSession.builder.appName(
    "StructuredNetworkWordCount"
).getOrCreate()


userSchema = (
    StructType()
    .add("id1", "string")
    .add("id2", "string")
    .add("time", "string")
    .add("NO_QUERY", "string")
    .add("username", "string")
    .add("content", "string")
)

# Create DataFrame representing the stream of input lines from connection to localhost:9999
lines = spark.read.csv("assets/data.csv", header=True, inferSchema=True)

# Split the lines into words
words = lines.select(explode(str(split(lines, " "))).alias("word"))

# Generate running word count
wordCounts = words.groupBy("word").count()

# Start running the query that prints the running counts to the console
query = wordCounts.writeStream.outputMode("complete").format("console").start()

query.awaitTermination()

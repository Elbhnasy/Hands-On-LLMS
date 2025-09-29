from pyspark.sql import SparkSession
from pyspark.sql.functions import avg

# Create Spark session
spark = SparkSession.builder.appName('SparkWithCSV').getOrCreate()

# Read CSV (assumes header row)
df = spark.read.csv('people.csv', header=True, inferSchema=True)

# Show the DataFrame
df.show()

# Quick analysis: Average age by city? (Group and aggregate)
df.groupBy('City').agg(avg('Age').alias('Avg Age')).show()

# Stop the session
spark.stop()

from pyspark.sql import SparkSession
from pyspark.sql.functions import length

spark = SparkSession.builder.appName("DataFrame").getOrCreate()

# df = spark.read.text("words.txt")

## first 5 words starting with b, ending with t
# df.select('value').where('value like "b%t"').show(5)

## last 10 longest words
# df = df.withColumn("lengths", length("value"))
# df = df.sort(df.lengths.desc())
# df.select('value','lengths').show(10,False)

## Calculate the number of lines and the number of distinct words from file1
df = spark.read.text("file1.txt")
print('NUMBER OF TOTAL LINES ARE - ',df.count())

word_count = df.rdd.flatMap(lambda x: x.split(' ')).distinct().count()
print("NUMBER OF DISTINCT WORDS: ", word_count)
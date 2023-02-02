words = spark.read.text("words.txt")

words_b_t = words.select('value').where('value like "b%t"')
words_b_t = words_b_t.withColumnRenamed('value','words')
words_b_t.show()
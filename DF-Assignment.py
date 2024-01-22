#!/usr/bin/env python
# coding: utf-8

# # PySpark DF Assignment(Group 8)
# ## 21522557	Trần Thanh Sơn
# ## 21522678	Phạm Trung Tín
# ## 21522515	Nguyễn Việt Quang

# In[1]:


import findspark
findspark.init()


# In[2]:


from pyspark.sql import SparkSession
from pyspark.sql.functions import col, split, explode, avg, max, first, desc
from pyspark.sql.types import *
from pyspark.sql.window import Window
from pyspark.sql.functions import rank, regexp_extract, udf, row_number
from pyspark.sql.functions import concat_ws, slice, collect_list,window


# Create a Spark session
spark = SparkSession.builder.appName("GenreRatings").getOrCreate()

# Load data from CSV files
movies_df = spark.read.csv("movies_small.csv", header=True, inferSchema=True)
ratings_df = spark.read.csv("ratings_small.csv", header=True, inferSchema=True)

# Extract year from title column
movies_df = movies_df.withColumn("year", regexp_extract(col("title"), "\((\d{4})\)", 1).cast(IntegerType()))
movies_df = movies_df.filter(col("year").isNotNull())
movies_df = movies_df.withColumn("genre", explode(split("genres", "\\|")))

# Join the ratings and movies dataframes
joined_df = ratings_df.join(movies_df, ratings_df.movieId == movies_df.movieId)


# ## 1. Show the number of movies made in each year. The results are sorted by year.

# In[3]:


movies_per_year = movies_df.groupBy("year").count().orderBy("year")
movies_per_year.show(100, truncate=False)


# ## 2. Show the number of movies belonging to each genre made in each year. The results are sorted by year. The movie count of each genre is displayed in each column.

# In[4]:


genre_count_per_year = (
    movies_df.groupBy("year", "genre")
    .count()
    .groupBy("year")
    .pivot("genre")
    .sum("count")
    .na.fill(0)
    .orderBy("year")
)
genre_count_per_year.show(100, truncate=False)


# ## 3. For each userID, show the average rating of that user for each genre. The results are sorted by userID. The result of each genre is displayed in each column.

# In[5]:


# Question 1
genre_avg_df = joined_df.groupBy("userId", "genre").agg(avg("rating").alias("avg_rating"))

# Pivot the DataFrame so that each genre becomes a column
pivoted_df = genre_avg_df.groupBy("userId").pivot("genre").agg(avg("avg_rating")).na.fill(0).orderBy("userId")

# Show the pivoted DataFrame
pivoted_df.show(100, False)


# ## 4. For each movie, show the name, the year, the number of ratings, and the average rating (from all users) of each movie. The results are sorted by the years and then by the names of the movies.

# In[6]:


from pyspark.sql.functions import count
# Window to partition by user ID and order by rating descending 
movie_stats_df = joined_df.groupBy("title").agg(count("rating").alias("Num_rating"), 
                                               avg("rating").alias("Avg_rating"))
#movie_stats_df = movie_stats_df.orderBy("year", "title")
movie_stats_df.show(100)


# In[7]:


movie_stats_df.join(joined_df, movie_stats_df.title == joined_df.title).select(joined_df["year"],movie_stats_df["*"]).distinct().orderBy("year", "title").withColumnRenamed("title","Movie_name").show(100, False)


# ## 5. For each user ID, show the genre that received highest average rating from that user and the list of top 5 movies belonging to that genre that receive highest average rating from all user and haven’t been rated by that user (For example, ‘Action’ is the genre that received highest average rating from user ID X. Among ‘Action’ movies that hasn’t been rated by user ID X, you are supposed to show top 5 movies that received highest average rating from all users). The results are sorted by the user ID.

# In[8]:


# Calculate the average rating for each genre-user combination
avg_rating_df = joined_df.groupBy("userId", "genre").agg(avg("rating").alias("avg_rating"))

# Identify the genre with the highest average rating for each user
user_top_genre_df = avg_rating_df.withColumn("top_genre_rank", rank().over(Window.partitionBy("userId").orderBy(col("avg_rating").desc())))
user_top_genre_df = user_top_genre_df.filter(col("top_genre_rank") == 1)

# Join the movie dataframe to get movie titles and year
movie_titles_df = movies_df.select("movieId", "title", "year")

# Define a window specification partitioned by userId and ordered by avg_rating descending
window_spec = Window.partitionBy("userId").orderBy(desc("avg_rating"))

# Add a column with the rank of each genre within each user
user_top_genre_df = user_top_genre_df.withColumn("top_genre_rank", row_number().over(window_spec))

# Filter out only the rows where the rank is 1
user_top_genre_df = user_top_genre_df.filter("top_genre_rank = 1")

# Show the final result
user_top_genre_df.show(100,False)



# In[9]:


# Filter movies that haven't been rated by the user
unrated_movies_df = ratings_df.join(user_top_genre_df, (ratings_df.userId == user_top_genre_df.userId))
unrated_movies_df = unrated_movies_df.drop("userId")

# Calculate the average rating for each genre-movie combination
avg_movie_rating_df = unrated_movies_df.groupBy("movieId", "genre").agg(avg("rating").alias("avg_movie_rating"))

# Identify the top 5 movies for each user's top genre
user_top_5_movies_df = avg_movie_rating_df.withColumn("movie_rank", rank().over(Window.partitionBy("genre").orderBy(col("avg_movie_rating").desc())))
user_top_5_movies_df = user_top_5_movies_df.filter(col("movie_rank") <= 5)

# Join the movie titles dataframe to get movie titles and year
user_top_5_movies_with_titles_df = user_top_5_movies_df.join(movie_titles_df, user_top_5_movies_df.movieId == movie_titles_df.movieId)

user_top_5_movies_with_titles_df.show(1000)


# In[10]:


# Join the user_top_genre_df with user_top_5_movies_with_titles_df to get the final result
final_df = user_top_genre_df.join(user_top_5_movies_with_titles_df,  (user_top_genre_df.genre == user_top_5_movies_with_titles_df.genre)).drop(user_top_genre_df.genre).distinct()
# Show the final result sorted by user ID
final_df = final_df.select("userId","genre", "title").withColumnRenamed("genre","Highest_rated_genre_name").orderBy("userId")


final_df.groupBy("userId","Highest_rated_genre_name").agg(concat_ws("|", slice(collect_list("title"), 1, 5))\
                          .alias("Top_5_unrated_movies_with_highest_rating")).show(100, False)


# ## 6. For each user ID, show the two genres that received highest rating from that user, and the list of top 5 highest rated movies that have both genres and hasn’t been rated by that user. The results are sorted by the user ID.

# In[11]:


# Calculate the average rating for each genre-user combination
avg_rating_df = joined_df.groupBy("userId", "genre").agg(avg("rating").alias("avg_rating"))
# Identify the genre with the highest average rating for each user
user_top_2_genre_df = avg_rating_df.withColumn("top_genre_rank", rank().over(Window.partitionBy("userId").orderBy(col("avg_rating").desc())))
user_top_2_genre_df = user_top_2_genre_df.filter(col("top_genre_rank") <= 2)

# Join the movie dataframe to get movie titles and year
movie_titles_df = movies_df.select("movieId", "title", "genre")

# Define a window specification partitioned by userId and ordered by avg_rating descending
window_spec = Window.partitionBy("userId").orderBy(desc("avg_rating"))

# Add a column with the rank of each genre within each user
user_top_2_genre_df = user_top_2_genre_df.withColumn("top_genre_rank", row_number().over(window_spec))

# Filter out only the rows where the rank is 1
user_top_2_genre_df = user_top_2_genre_df.filter("top_genre_rank <= 2")

#
# Show the final result
user_2_genre_df = user_top_2_genre_df.select('userId', 'genre').groupBy("userId").agg(concat_ws("|", slice(collect_list("genre"), 1, 2)).alias("Two_highest_rated_genre_names"))
user_top_2_genre_df = user_top_2_genre_df.select('userId', 'genre').withColumnRenamed("genre","top_rated_genre_name").orderBy("userId")
user_2_genre_df .show(100, False)


# In[12]:


# Filter movies that haven't been rated by the user
unrated_movies_df = ratings_df.join(user_top_2_genre_df, (ratings_df.userId == user_top_2_genre_df.userId)).drop(ratings_df.userId)
unrated_movies_df = unrated_movies_df.join(movie_titles_df, movie_titles_df.movieId == unrated_movies_df.movieId).distinct().drop("movieId","rating","timestamp") 

unrated_movies_df = unrated_movies_df.filter(col("top_rated_genre_name") == col("genre"))
unrated_movies_df =unrated_movies_df.groupBy(col("userId"), col("title")).agg(count("*").alias("count")).filter(col("count") == 2)
unrated_movies_df.show(100)


# In[13]:


unrated_2_movies_df = user_2_genre_df.join(unrated_movies_df, (unrated_movies_df.userId == user_2_genre_df.userId)).drop(unrated_movies_df.userId, "count")

unrated_2_movies_df.groupBy("userId","Two_highest_rated_genre_names").agg(concat_ws("|", slice(collect_list("title"), 1, 5))\
                          .alias("Top_5_unrated_movies_with_highest_rating")).show(100, False)


# ## 7. Show the years of the first_appearance of each genre.

# In[14]:


from pyspark.sql.functions import min

genre_year_df = joined_df.groupBy("genre").agg(min("year").alias("first_appearance_year"))

genre_year_df.show(100, False)


# ## 8. For each user ID, show the list of top 5 movies made after 2000 that received highest rating from that user. The results are sorted by user ID.

# In[15]:


from pyspark.sql import functions as F

# Filter for movies after 2000
after_2000_df = joined_df.filter(col("year") > 2000)

# Window to partition by user ID and order by rating descending 
window = Window.partitionBy("userId").orderBy(col("rating").desc())

# Get top 5 rows in each partition and select only title
top5_titles = after_2000_df.select("*", rank().over(window).alias("rank")) \
                    .filter(col("rank") <= 5) \
                    .select("userId", "title")

# Concatenate top 5 movie titles for each user
top5_concat = top5_titles.groupBy("userId").agg(
    concat_ws("|", slice(collect_list("title"), 1, 5)).alias("Top_5_movie_after_2000_rated_by_this_user")
)
# Show the results
top5_concat.show(100, truncate=False)


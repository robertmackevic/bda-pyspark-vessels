from pyspark.sql import SparkSession
from pyspark.sql.functions import to_timestamp, col, lag, radians, sin, cos, sqrt, atan2, sum as pyspark_sum
from pyspark.sql.types import DoubleType
from pyspark.sql.window import Window
from src.paths import DATASET_CSV
import math
from pyspark.sql.types import StructType, StructField, StringType, DoubleType, TimestampType, LongType, IntegerType
import folium

def haversine_distance(lat1:float, lon1:float, lat2:float, lon2:float):
    # Radius of the Earth in km
    R = 6371.0  

    # Convert latitude and longitude from degrees to radians
    lat1_rad = radians(lat1)
    lon1_rad = radians(lon1)
    lat2_rad = radians(lat2)
    lon2_rad = radians(lon2)

    # Calculate the change in coordinates
    dlon = lon2_rad - lon1_rad
    dlat = lat2_rad - lat1_rad

    # Haversine formula
    a = sin(dlat / 2) ** 2 + cos(lat1_rad) * cos(lat2_rad) * sin(dlon / 2) ** 2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))

    # Calculate the distance
    distance = R * c
    return distance



def run():
    spark = SparkSession.builder\
        .appName("Load CSV into PySpark DataFrame") \
        .config("spark.driver.memory", "12g") \
        .config("spark.executor.memory", "12g") \
        .config("spark.memory.offHeap.enabled","true") \
        .config("spark.memory.offHeap.size","12g")\
        .config("spark.delta.catalog.update.enabled", "false") \
        .getOrCreate()

    df = spark.read.csv(str(DATASET_CSV), header=True)
    df = df.withColumn("timestamp", to_timestamp(col("# Timestamp"), "dd/MM/yyyy HH:mm:ss"))
    columns_to_keep = ["timestamp", "MMSI", "latitude", "longitude"]
    df = df.select(*columns_to_keep)
    df = df.dropna(subset=columns_to_keep)
    df = df.withColumn("timestamp", df["timestamp"].cast(TimestampType()))\
           .withColumn("MMSI", df["MMSI"].cast(IntegerType()))\
           .withColumn("latitude", df["latitude"].cast(StringType()))\
           .withColumn("longitude", df["longitude"].cast(StringType()))

    windowSpec = Window.partitionBy("MMSI").orderBy("timestamp")
    df = df.withColumn("prev_latitude", lag("latitude").over(windowSpec))
    df = df.withColumn("prev_longitude", lag("longitude").over(windowSpec))
    df = df.dropna(subset=["prev_latitude", "prev_longitude"])

    df = df.filter((df["prev_latitude"] != df["latitude"]) | (df["prev_longitude"] != df["longitude"]))
    
    df = df.withColumn("distance", haversine_distance(
        col("prev_latitude"), col("prev_longitude"), col("latitude"), col("longitude")
    ))

    grouped_df = df.groupBy("MMSI").agg(pyspark_sum("distance").alias("total_distance"))

    top_5_mmsi = grouped_df.orderBy(grouped_df["total_distance"].desc()).limit(5)

    first_mmsi = top_5_mmsi.select("MMSI").first()
    first_mmsi = first_mmsi["MMSI"]
    first_mmsi_df = df.where(df["MMSI"] == first_mmsi)
    coordinates = first_mmsi_df.select("latitude", "longitude").collect()


    m = folium.Map(location=[float(coordinates[0]["latitude"]), float(coordinates[0]["longitude"])], zoom_start=11)

    trail_coordinates = [(float(coord["latitude"]), float(coord["longitude"])) for coord in coordinates]
    
    folium.PolyLine(trail_coordinates, tooltip="Coast").add_to(m)
    m.save("map.html")
    m

    print("Top 5 MMSI by Distance Sum:")
    top_5_mmsi.show()

    df.show(10)
    spark.stop()

if __name__ == "__main__":
    run()

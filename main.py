import pyspark.sql.functions as f
from folium import Map, PolyLine
from pyspark.sql import SparkSession
from pyspark.sql.types import StringType, IntegerType
from pyspark.sql.window import Window
from matplotlib import pyplot as plt
from src.distance import haversine_distance
from src.paths import DATASET_CSV, DATA_DIR


def run() -> None:
    spark = SparkSession.builder \
        .appName("PySpark Vessels") \
        .config("spark.driver.memory", "12g") \
        .config("spark.executor.memory", "12g") \
        .config("spark.memory.offHeap.enabled", "true") \
        .config("spark.memory.offHeap.size", "12g") \
        .config("spark.delta.catalog.update.enabled", "false") \
        .getOrCreate()

    # Loading the data, converting to timestamp datatype
    df = spark.read.csv(str(DATASET_CSV), header=True)
    df = df.withColumn("timestamp", f.to_timestamp(f.col("# Timestamp"), "dd/MM/yyyy HH:mm:ss"))

    # Dropping unnecessary columns
    columns_to_keep = ["timestamp", "MMSI", "latitude", "longitude"]
    df = df.select(*columns_to_keep)
    df = df.dropna(subset=columns_to_keep)

    # Make sure the other columns have appropriate typing
    df = df.withColumn("MMSI", df["MMSI"].cast(IntegerType())) \
        .withColumn("latitude", df["latitude"].cast(StringType())) \
        .withColumn("longitude", df["longitude"].cast(StringType()))

    # Find the previous locations of vessels
    windowSpec = Window.partitionBy("MMSI").orderBy("timestamp")
    df = df.withColumn("prev_latitude", f.lag("latitude").over(windowSpec))
    df = df.withColumn("prev_longitude", f.lag("longitude").over(windowSpec))
    df = df.withColumn("prev_time", f.lag("timestamp").over(windowSpec))
    df = df.dropna(subset=["prev_latitude", "prev_longitude", "prev_time"])

    df = df.filter((df["prev_latitude"] != df["latitude"]) | (df["prev_longitude"] != df["longitude"]))

    df = df.withColumn("distance", haversine_distance(
        f.col("prev_latitude"), f.col("prev_longitude"), f.col("latitude"), f.col("longitude")
    ))
    df = df.withColumn("time_diff", f.col("timestamp").cast("long") - f.col("prev_time").cast("long"))
    df = df.withColumn("speed", f.col("distance") / f.col("time_diff") * 3600)
    
    df = df.filter(df["speed"] < 50)

    # sample_df = df.sample(False, 100000 / df.count(), seed=42)
    # samples = [i["speed"] for i in sample_df.select("speed").collect() if i["speed"] is not None]
    # plt.figure(figsize=(10, 6))
    # plt.hist(samples, bins=500, color='skyblue', edgecolor='black')
    # plt.xlabel('Speed (km/h)')
    # plt.ylabel('Frequency')
    # plt.title('Speed Distribution')
    # plt.grid(True)
    # plt.show()



    
    grouped_df = df.groupBy("MMSI").agg(f.sum("distance").alias("total_distance"))
    top_5_mmsi = grouped_df.orderBy(grouped_df["total_distance"].desc()).limit(5)
    print("Top 5 Distance Traveled by Vessel (MMSI):")
    top_5_mmsi.show()

    # Plot the journey of a vessel onto a map
    # The map is saved in data/map.html
    first_mmsi = top_5_mmsi.select("MMSI").first()
    first_mmsi = first_mmsi["MMSI"]
    first_mmsi_df = df.where(df["MMSI"] == first_mmsi)
    coordinates = first_mmsi_df.select("latitude", "longitude").collect()

    _map = Map(location=[float(coordinates[0]["latitude"]), float(coordinates[0]["longitude"])], zoom_start=11)
    trail_coordinates = [(float(coord["latitude"]), float(coord["longitude"])) for coord in coordinates]
    PolyLine(trail_coordinates, tooltip="Coast").add_to(_map)
    _map.save(DATA_DIR / "map.html")

    spark.stop()


if __name__ == "__main__":
    run()

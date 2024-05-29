from argparse import Namespace, ArgumentParser

import pyspark.sql.functions as f
from folium import Map, PolyLine
from matplotlib import pyplot as plt
from pyspark.sql import SparkSession
from pyspark.sql.types import StringType, IntegerType
from pyspark.sql.window import Window

from src.distance import haversine_distance
from src.paths import DATASET_CSV, DATA_DIR


def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument("-v", "--visualize", required=False, action="store_true")
    return parser.parse_args()


def run(visualize: bool) -> None:
    spark = SparkSession.builder.appName("PySpark Vessels").getOrCreate()

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

    # Filter out the entries that show no movement
    df = df.filter((df["prev_latitude"] != df["latitude"]) | (df["prev_longitude"] != df["longitude"]))

    # Compute the distance between consecutive positions, time difference and speed
    df = df.withColumn("distance", haversine_distance(
        f.col("prev_latitude"), f.col("prev_longitude"), f.col("latitude"), f.col("longitude")
    ))
    df = df.withColumn("time_diff", f.col("timestamp").cast("long") - f.col("prev_time").cast("long"))
    df = df.withColumn("speed", f.col("distance") / f.col("time_diff") * 3600)

    # Optionally, plot the distribution of speed to visualize outliers
    if visualize:
        df = df.filter("speed < 100")
        sample_df = df.sample(False, 10000 / df.count(), seed=42)
        samples = [i["speed"] for i in sample_df.select("speed").collect() if i["speed"] is not None]
        plt.figure(figsize=(10, 6))
        plt.hist(samples, bins=100, color="skyblue", edgecolor="black")
        plt.xlabel("Speed (km/h)")
        plt.ylabel("Frequency")
        plt.title("Speed Distribution")
        plt.grid(True)
        plt.savefig(DATA_DIR / "speed-distribution.png")

    # Based on the speed distribution plot we can see that max speed is around 50 km/h
    # Anything more than that can be filtered out to get rid of the noise from the data.
    df = df.filter("speed < 50")

    # Find the top vessels by distance traveled
    grouped_df = df.groupBy("MMSI").agg(f.sum("distance").alias("total_distance"))
    top_5_mmsi = grouped_df.orderBy(grouped_df["total_distance"].desc()).limit(5)

    top_5_mmsi.show()
    first_row = top_5_mmsi.collect()[0]
    first_mmsi = first_row["MMSI"]
    total_distance = first_row["total_distance"]

    # Optionally, Plot the journey onto a map of the vessel that traveled the longest distance
    # The map is saved in data/map.html
    if visualize:
        first_mmsi_df = df.where(df["MMSI"] == first_mmsi)
        coordinates = first_mmsi_df.select("latitude", "longitude").collect()

        _map = Map(location=[float(coordinates[0]["latitude"]), float(coordinates[0]["longitude"])], zoom_start=11)
        trail_coordinates = [(float(coord["latitude"]), float(coord["longitude"])) for coord in coordinates]
        PolyLine(trail_coordinates, tooltip="Coast").add_to(_map)
        _map.save(DATA_DIR / "map.html")

    spark.stop()

    print("MMSI:", first_mmsi)
    print("Distance:", total_distance)


if __name__ == "__main__":
    run(**vars(parse_args()))

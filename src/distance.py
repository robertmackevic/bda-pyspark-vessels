import pyspark.sql.functions as f
from pyspark.sql.column import Column


def haversine_distance(lat1: Column, lon1: Column, lat2: Column, lon2: Column) -> Column:
    lat1_rad = f.radians(lat1)
    lon1_rad = f.radians(lon1)
    lat2_rad = f.radians(lat2)
    lon2_rad = f.radians(lon2)

    delta_lon = lon2_rad - lon1_rad
    delta_lat = lat2_rad - lat1_rad

    a = f.sin(delta_lat / 2) ** 2 + f.cos(lat1_rad) * f.cos(lat2_rad) * f.sin(delta_lon / 2) ** 2
    c = 2 * f.atan2(f.sqrt(a), f.sqrt(1 - a))

    return f.lit(6371.0) * f.lit(c)

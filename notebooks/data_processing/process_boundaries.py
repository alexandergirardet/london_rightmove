from pyrosm import OSM, get_data
import pandas as pd
import requests
import os
pd.options.mode.chained_assignment = None
import tempfile


def process_geodata():
    # file_path = f"../resources/boundary_date/data/{file_name}"
    file_name = "greater-london-latest.osm.pbf"
    file_path = "/Users/alexander.girardet/Code/Personal/projects/rightmove_project/data/greater-london-latest.osm.pbf"
    print(f"processing: {file_name}")

    osm = OSM(file_path)

    boundary_name = file_name.split(".osm")[0]

    boundary_df = osm.get_boundaries()
    boundary_df = boundary_df.rename(columns={"id": "boundary_id"})

    output_filename = f'geodata/{boundary_name}.geojson'  # Specifying the path in writable storage
    boundary_df.to_file(output_filename, driver='GeoJSON')
    print(f"Loaded {output_filename}")

# already_processed = os.listdir("geodata")
# processed_names = [name.split(".geojson")[0] for name in already_processed]
# files = os.listdir("../resources/boundary_date/data")

# for file_name in files:
#     name = file_name.split(".osm.pbf")[0]
#     if name not in processed_names:
#         if name != "scotland-latest":
#             try:
#                 process_geodata(file_name)
#             except:
#                 print(f"Failed to process: {name}")
process_geodata()
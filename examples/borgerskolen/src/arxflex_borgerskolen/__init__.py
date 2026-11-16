"""Example data for arxflex: room temperatures in Borgerskolen, March 2021.

30-min data for 16 rooms with thermostat set-points, plus outdoor temperature
and global radiation from DMI. How the data were made: see the README of
examples/borgerskolen in the arxflex repository. The data are under CC BY 4.0.

    from arxflex_borgerskolen import load_processed_data, available_rooms
    data = load_processed_data()
"""

from .load_processed_data import load_processed_data, DATA_FILE
from .temperature_column import temperature_column
from .available_rooms import available_rooms

__all__ = ["load_processed_data", "DATA_FILE", "temperature_column", "available_rooms"]

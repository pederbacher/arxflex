def temperature_column(data, roomname):
    """Name of the room temperature column, e.g. roomname "1.09".
    (The notebook assumed "__elsys"; room 0.01 has "__icmeter".)"""
    cols = [c for c in data.columns if c.startswith(f"Room_C{roomname}_Temperature__")]
    if len(cols) != 1:
        raise ValueError(f"no unique temperature column for room {roomname}: {cols}")
    return cols[0]

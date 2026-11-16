import re


def available_rooms(data, model_columns_only=True):
    """Names of the rooms in the data (without "C", e.g. "1.09"), sorted.

    model_columns_only=True : only rooms that can be modelled, i.e. rooms with a
        temperature column and all the model columns Tset_minus_Ti_C<room>,
        Tset_C<room>, Ta_minus_Ti_C<room> and Tfor_minus_Ti_C<room>
        (16 rooms in borgerskolen_experiment.csv).
    model_columns_only=False : all rooms with a temperature measurement
        (18 rooms; C0.07A and C1.07 lack the model columns, so they can't be
        modelled with this data).
    """
    pattern = re.compile(r"Room_C(.+?)_Temperature__")
    rooms = sorted({pattern.match(c).group(1) for c in data.columns if pattern.match(c)})
    if model_columns_only:
        prefixes = ("Tset_minus_Ti_C", "Tset_C", "Ta_minus_Ti_C", "Tfor_minus_Ti_C")
        rooms = [r for r in rooms if all(p + r in data.columns for p in prefixes)]
    return rooms

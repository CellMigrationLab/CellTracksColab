#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from copy import deepcopy
import glob
from pathlib import Path
from typing import TypeAlias

from lxml import etree as ET
import pandas as pd
from tqdm.notebook import tqdm



# Notes about TrackMate CSVs:
# - CSVs keep spots with no TRACK_ID. In that case the TRACK_ID cell is empty.
# - CSVs only keep spots with a 1 VISIBILITY. So spots with a 0 VISIBILITY
#   are removed from the spots dataframe.
# - CSVs only keep filtered tracks. So these tracks are removed from the dataframe
#   as well as the corresponding spots in the spot dataframe.

Features: TypeAlias = dict[str, dict[str, str]]

def check_calibration(units_list):
    """Check if all units calibration data match and print them."""
    if all(u == units_list[0] for u in units_list):
        print(f"All files have consistent calibration: {units_list[0]}")
    else:
        print("Warning: Inconsistent calibration data across files!")

def get_features_dict(
    iterator: ET.iterparse,
    ancestor: ET._Element,
) -> Features:
    """
    Get all the features of ancestor and return them as a dictionary.

    Parameters
    ----------
    iterator : ET.iterparse
        XML element iterator.
    ancestor : ET._Element
        Element encompassing the information to add.

    Returns
    -------
    Features
        A dictionary of features contained in the ancestor element.
    """
    features = dict()
    event, element = next(iterator)  # Feature.
    while (event, element) != ("end", ancestor):
        if element.tag == "Feature" and event == "start":
            attribs = deepcopy(element.attrib)
            current_feat = attribs["feature"]
            # We don't need all the feature declaration, just the part
            # about dimension and isint. So we can delete the rest.
            to_keep = ["dimension", "isint"]
            for key in [k for k in attribs if k not in to_keep]:
                attribs.pop(key)
            # Adding the feature to the dictionary.
            try:
                features[current_feat] = attribs
            except KeyError as err:
                print(
                    f"No key {err} in the attributes of "
                    f"current element '{element.tag}'. "
                    f"Not adding this feature to the features dictionary."
                )
        element.clear()
        event, element = next(iterator)
    return features


def get_all_features(
    iterator: ET.iterparse,
    ancestor: ET._Element,
) -> dict[str, Features]:
    """
    Create a dictionary of all features with their dimension and if they are integer.

    The model features are divided in 3 categories: spots, edges and tracks features.
    Those features are regrouped under the tag FeatureDeclarations.

    Parameters
    ----------
    iterator : ET.iterparse
        XML element iterator.
    ancestor : ET._Element
        Element encompassing the information to add.

    Returns
    -------
    dict[str, Features]
        All the features contained in the ancestor element.
    """
    feat_dict = dict()
    event, element = next(iterator)
    while (event, element) != ("end", ancestor):
        features = get_features_dict(iterator, element)

        feat_dict[element.tag] = features
        element.clear()
        event, element = next(iterator)
    return feat_dict


def convert_attributes(
    attributes: dict[str, str],
    features: Features,
):
    """
    Convert the values of `attributes` from string to int or float.

    The type to convert to is given by the dictionary of features with
    the key 'isint'.

    Parameters
    ----------
    attributes : dict[str, str]
        The dictionary whose values we want to convert.
    features : dict[str, dict[str, str]]
        The dictionary holding the type information to use.

    Raises
    ------
    KeyError
        If the 'isint' feature attribute doesn't exist.
    ValueError
        If the value of the 'isint' feature attribute is invalid.
    """
    for key in attributes:
        if key == "ID":
            attributes[key] = int(attributes[key])  # IDs are always integers.
        elif key in features:
            if "isint" not in features[key]:
                raise KeyError("No 'isint' feature attribute in FeatureDeclarations.")
            if features[key]["isint"].lower() == "true":
                attributes[key] = int(attributes[key])
            elif features[key]["isint"].lower() == "false":
                try:
                    attributes[key] = float(attributes[key])
                except ValueError:
                    pass  # Not an int nor a float so we let it be.
            else:
                raise ValueError(
                    f"'{features[key]['isint']}' is an invalid"
                    f" feature attribute value for 'isint'."
                )


def extract_spots_df(
    iterator: ET.iterparse,
    ancestor: ET._Element,
    features: Features,
) -> pd.DataFrame:
    """
    Extract a dataframe of spots.

    All the elements that are descendants of `ancestor` are explored.

    Parameters
    ----------
    iterator : ET.iterparse
        XML element iterator.
    ancestor : ET._Element
        Element encompassing the information to add.
    features: dict[str, dict[str, str]]
        Features of the spots.

    Returns
    -------
    pd.DataFrame
        A dataFrame of spots.
    """
    spot_list = []
    event, element = next(iterator)
    while (event, element) != ("end", ancestor):
        event, element = next(iterator)
        if element.tag == "Spot" and event == "end":
            # All items in element.attrib are parsed as strings but most
            # of them (if not all) are numbers. So we need to do a
            # conversion based on these attributes type (attribute `isint`)
            # as defined in the FeaturesDeclaration tag.
            attribs = deepcopy(element.attrib)
            try:
                convert_attributes(attribs, features)
            except ValueError as err:
                print(f"ERROR: {err} Please check the XML file.")
                raise
            except KeyError as err:
                print(f"ERROR: {err} Please check the XML file.")
                raise
            if "ROI_N_POINTS" in attribs:
                # We don't need this attribute but it's not always there
                # depending on how the spot detection was done.
                attribs.pop("ROI_N_POINTS")
            spot_list.append(attribs)
            element.clear()
    return pd.DataFrame(spot_list)


def update_spot_track_id_dict(
    spot_track_id_dict: dict[int, int],
    element: ET._Element,
    current_track_id: int,
    features: Features,
):
    """
    Update the spots dict with the track ID of the spots of the current edge.

    An edge is defined by two spots: the source and the target.

    Parameters
    ----------
    spot_track_id_dict : dict[int, int]
        Spots dictionary to updtae.
    element : ET._Element
        Element holding the information to be added.
    current_track_id : int
        Track ID of the spots of the edge.
    features: dict[str, dict[str, str]]
        Features of the edges.
    """
    attribs = deepcopy(element.attrib)
    try:
        convert_attributes(attribs, features)
    except ValueError as err:
        print(f"ERROR: {err} Please check the XML file.")
        raise
    except KeyError as err:
        print(f"ERROR: {err} Please check the XML file.")
        raise
    try:
        entry_spot = attribs["SPOT_SOURCE_ID"]
        exit_spot = attribs["SPOT_TARGET_ID"]
    except KeyError as err:
        print(
            f"No key {err} in the attributes of current element '{element.tag}'."
            f" Not adding these spots to the spots dictionary."
        )
    else:
        # Adding the current track ID to the spots of the edge. This will be
        # useful later to reinject the track ID into the spots dataframe.
        spot_track_id_dict[entry_spot] = current_track_id
        spot_track_id_dict[exit_spot] = current_track_id
    finally:
        element.clear()


def extract_tracks_df(
    iterator: ET.iterparse,
    ancestor: ET._Element,
    track_feats: Features,
    edge_feats: Features,
) -> tuple[pd.DataFrame, dict[int, int]]:
    """
    Extract a dataframe of tracks and a dictionary of spots TRACK_ID.

    All the elements that are descendants of `ancestor` are explored.

    Parameters
    ----------
    iterator : ET.iterparse
        XML element iterator.
    ancestor : ET._Element
        Element encompassing the information to add.
    features : dict[str, dict[str, str]]
        Features of the tracks.

    Returns
    -------
    tuple[pd.DataFrame, dict[int, int]]
        A dataframe of tracks and a dictionary with spots ID as keys and
        the associated TRACK_ID as value.
    """
    track_list = []
    event, element = next(iterator)
    spot_track_id_dict = dict()
    current_track_id = None

    while (event, element) != ("end", ancestor):
        # Saving the current track information.
        if element.tag == "Track" and event == "start":
            attribs = deepcopy(element.attrib)
            try:
                convert_attributes(attribs, track_feats)
            except ValueError as err:
                print(f"ERROR: {err} Please check the XML file.")
                raise
            except KeyError as err:
                print(f"ERROR: {err} Please check the XML file.")
                raise
            track_list.append(attribs)
            try:
                current_track_id = attribs["TRACK_ID"]
            except KeyError as err:
                print(
                    f"No key {err} in the attributes of "
                    f"current element '{element.tag}'. "
                )
                current_track_id = None

        # Saving the spots TRACK_ID for later reinjection into the spots dataframe.
        if element.tag == "Edge" and event == "start":
            update_spot_track_id_dict(
                spot_track_id_dict, element, current_track_id, edge_feats
            )

        event, element = next(iterator)

    return pd.DataFrame(track_list), spot_track_id_dict


def get_filtered_tracks_ID(
    iterator: ET.iterparse,
    ancestor: ET._Element,
) -> list[str]:
    """
    Get the list of IDs of the tracks to keep.

    Parameters
    ----------
    iterator : ET.iterparse
        XML element iterator.
    ancestor : ET._Element
        Element encompassing the information to add.

    Returns
    -------
    list[str]
        List of tracks ID to keep.
    """
    filtered_tracks_ID = []
    event, element = next(iterator)
    while (event, element) != ("end", ancestor):
        if element.tag == "TrackID" and event == "start":
            attribs = deepcopy(element.attrib)
            try:
                filtered_tracks_ID.append(int(attribs["TRACK_ID"]))
            except KeyError as err:
                print(
                    f"No key {err} in the attributes of current element "
                    f"'{element.tag}'. Ignoring this track."
                )
        event, element = next(iterator)
    return filtered_tracks_ID


def trackmate_xml_to_df(xml_path: str) -> tuple[pd.DataFrame, pd.DataFrame, dict]:

    """
    Read a XML file and extract spots, tracks dataframes, and calibration units.

    Parameters
    ----------
    xml_path : str
        Path of the XML file to process.

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame]
        A spots and a tracks dataframes.
    """
    # So as not to load the entire XML file into memory at once, we're
    # using an iterator to browse over the tags one by one.
    # The events 'start' and 'end' correspond respectively to the opening
    # and the closing of the considered tag.
    it = ET.iterparse(xml_path, events=["start", "end"])
    _, root = next(it)  # Saving the root of the tree for later cleaning.

    for event, element in it:
        # Getting spatial and temporal units.
        if element.tag == "Model" and event == "start":  # Add units.
            units = element.attrib
            print(f"Units: {units}")
            root.clear()  # Cleaning the tree to free up some memory.
            # All the browsed subelements of `root` are deleted.

        # Getting dimension and isint information about spot, edge and track features.
        if element.tag == "FeatureDeclarations" and event == "start":
            features = get_all_features(it, element)
            root.clear()

        # Creation of the spots dataframe.
        if element.tag == "AllSpots" and event == "start":
            spot_df = extract_spots_df(it, element, features["SpotFeatures"])
            root.clear()

        # Creation of the tracks dataframe.
        if element.tag == "AllTracks" and event == "start":
            track_df, spot_track_id_dict = extract_tracks_df(
                it, element, features["TrackFeatures"], features["EdgeFeatures"]
            )
            root.clear()

        # Extracting ID of filtered tracks.
        if element.tag == "FilteredTracks" and event == "start":
            id_to_keep = get_filtered_tracks_ID(it, element)
            root.clear()

        if element.tag == "Model" and event == "end":
            break  # We are not interested in the following data.

    # Now that we have finished parsing the XML file, we can modify
    # the spots and tracks dataframes so it matches the CSV outputs of TrackMate.

    # Adding the track ID to the spots dataframe.
    spot_df["TRACK_ID"] = spot_df["ID"].map(spot_track_id_dict)
    # Spots with no match have a NaN TRACK_ID.
    # However, NaN are considered as floats so we need to convert them to integers
    # thanks to the Int64 type, specific to pandas.
    spot_df["TRACK_ID"] = spot_df["TRACK_ID"].astype("Int64")

    # Removing filtered tracks and the corresponding nodes.
    track_df = track_df[track_df["TRACK_ID"].isin(id_to_keep)]
    spot_df = spot_df[spot_df["TRACK_ID"].isin(id_to_keep)]

    # Removing spots with a 0 VISIBILITY.
    spot_df = spot_df[spot_df["VISIBILITY"] != 0]

    # Reordering and renaming columns.
    spot_df.rename(columns={"name": "LABEL"}, inplace=True)
    track_df.rename(columns={"name": "LABEL"}, inplace=True)
    spot_column_order = ["LABEL", "ID", "TRACK_ID"] + [
        k for k in features["SpotFeatures"]
    ]
    spot_df = spot_df.reindex(columns=spot_column_order)
    track_column_order = ["LABEL"] + [k for k in features["TrackFeatures"]]
    track_df = track_df.reindex(columns=track_column_order)

    return spot_df, track_df, units


def load_and_populate_from_TM_XML(folder_path):

    track_dfs, spot_dfs = [], []
    units_list = []
    xml_files = list(glob.glob(f"{folder_path}/*/*/*.xml"))
    for filepath in tqdm(xml_files, desc="Processing XML Files"):
        #print(filepath)
        # Loading a TrackMate XML as a dataframe.
        spot_df, track_df, units = trackmate_xml_to_df(filepath)
        units_list.append(units)

        # Adding CellTracksCollab features in spots dataframe.
        filename = Path(filepath).stem
        filename_parts = Path(filepath).parts
        condition = filename_parts[-3]
        repeat = filename_parts[-2]
        spot_df["File_name"] = filename
        spot_df["Condition"] = condition
        spot_df["experiment_nb"] = repeat
        # "Repeat" and "Unique_ID" are already added in the CellTracksColab.
        # spot_df["Repeat"] = repeat
        # spot_df["Unique_ID"] = filename + "_" + spot_df["TRACK_ID"].astype(str)
        spot_dfs.append(spot_df)

        # In tracks dataframe.
        track_df["File_name"] = filename
        track_df["Condition"] = condition
        track_df["experiment_nb"] = repeat
        # "Repeat" and "Unique_ID" are already added in the CellTracksColab.
        # track_df["Repeat"] = repeat
        # track_df["Unique_ID"] = filename + "_" + track_df["TRACK_ID"].astype(str)
        track_dfs.append(track_df)

    # Creating the final dataframes.
    merged_spots_df = pd.concat(spot_dfs, ignore_index=True)
    merged_tracks_df = pd.concat(track_dfs, ignore_index=True)
    check_calibration(units_list)
    return merged_spots_df, merged_tracks_df

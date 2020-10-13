#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import os
import pandas as pd
import numpy as np
import rasterio
from rasterio.windows import Window
import csv
import datetime as dt
import logging

logger = logging.getLogger(__name__)
# Override the default severity of logging.
logger.setLevel('INFO')
# Use StreamHandler to log to the console.
stream_handler = logging.StreamHandler()
# Don't forget to add the handler.
logger.addHandler(stream_handler)


def _get_pattern(oneFullpath):
    """ Sources the date extracted from Sentinel-2 fullpath filenames.
    Uses the .SAFE format of Sentinel-2 data to be functional.
    Args:
        oneFullpath (string): Fullpath.
    Returns:
        datetime object
    """
    return dt.datetime.strptime(oneFullpath.split('.SAFE')[0].split('_')[-1][0:8], '%Y%m%d')



def cbdf2cbarr(cbdf, metadata):
    """ Convert dataframe of cube to corresponding 3d cube array.
    Args:
        cbdf (pandas dataframe): Indexed as (rows:bands, row wise read, columns:individual pixels)
        metadata (dictionary): Containing all cube metadata, as returned from rasterio lib
    Return:
        cbarr (3d array): Indexed as tensor (count:bands, height:rows, width:columns)
    """

    # Convert dataframe to array
    temp = cbdf.to_numpy(dtype=metadata['dtype'])
    # Create axis, from 2D to 3D
    temp = temp[np.newaxis, :, np.newaxis]
    # Reshape array
    cbarr = np.reshape(temp, (metadata['count'], metadata['height'],  metadata['width']))

    return cbarr


def cbarr2cbdf(cbarr, metadata):
    """ Convert 3d cube array to corresponding dataframe of cube.
    Args:
        cbarr (3d array): Indexed as tensor (count:bands, height:rows, width:columns)
        metadata (dictionary): Containing all cube metadata, as returned from rasterio lib
    Return:
        cbdf (pandas dataframe): Indexed as (rows:bands, row wise read, columns:individual pixels)
    """

    # Drop array to 2D.
    temp = np.reshape(cbarr, (metadata['count'], metadata['height'] *  metadata['width']))
    # Convert array to dataframe.
    cbdf = pd.DataFrame(
        temp, columns=["pix_"+str(i) for i in range(0, metadata['height'] *  metadata['width'])])

    return cbdf


def cubePart(imPath, row_start, row_stop, col_start, col_stop, band_start, band_stop, **kwargs):
    """ Returns part of cube data as 3D array, dataframe and metadata of returned subset.
    Dataframe rows correspond to images / bands. Dataframe column corresponds to one pixel's depth.

    Args:
        imPath (string): Cube's fullpath.
        row_start, row_stop, col_start, col_stop, band_start, band_stop (int): Image coordinates,
                                                                    starting counting from zero.
    Returns:
        cube (numpy array): 3D numpy array.
        cube_df (pandas dataframe): Every row is one cube's image, every column is a pixel.
        metadata (dictionary): New image's updated metadata.
    """

    # Construct a window by image coordinates.
    win = Window.from_slices(slice(row_start, row_stop), slice(col_start, col_stop))

    rows = row_stop-row_start
    cols = col_stop-col_start
    bands = [i for i in range(band_start+1, band_stop+1)]

    # Check if imPath is string.
    assert isinstance(imPath, str)

    with rasterio.open(imPath) as src:
        # Find top-left x, y coordinates of new image, from selected starting row, col.
        new_left_top_coords = src.xy(row_start, col_start, offset='ul')
        # Original image metadata.
        metadata = src.meta
        # Assume that pixel is cubic.
        pixelSize = list(metadata['transform'])[0]
        # Create the new transformation.
        transf = rasterio.transform.from_origin(
            new_left_top_coords[0], new_left_top_coords[1], pixelSize, pixelSize)

        # Update metadata. Can be used to save new geolocated image.
        metadata.update(height=rows , width=cols, count=len(bands), transform=transf)

        # Read image as 3d cube
        cube = src.read(bands, window=win)
    # Convert array to dataframe
    cube_df = cbarr2cbdf(cube, metadata)

    return cube, cube_df, metadata




def readCube(imPath, **kwargs):
    """ Read an image as 3D array, dataframe & corresponding metadata.

    Args:
        imPath (string): Fullpath to multiband image.

    Returns:
        cube (numpy ndarray): 3D array.
        cube_df (pandas dataframe): Every row is one cube's image, every column is a pixel's depth.
        metadata (dictionary): Metadata of original cube.
    """

    with rasterio.open(imPath) as src:
        metadata = src.meta
        bands = [i for i in range(1, metadata['count']+1)]

        # Read image as 3d cube
        cube = src.read(bands)
    # Convert array to dataframe
    cube_df = cbarr2cbdf(cube, metadata)

    return cube, cube_df, metadata




def dataframe2tifCube(df, metadata, newFilename, searchPath, **kwargs):
    """ Writes a dataframe on disk, with georeference.

    Args:
        df (pandas dataframe): Every row is one cube's image, every column is a pixel.
        metadata (dictionary): Metadata of original cube.
        newFilename (string): Not a full path. Only the filename, without format ending.
        searchPath (string): Fullpath, where the result will be saved.
    Return:
        None
    """

    bands = [i for i in range(1, metadata['count']+1)]
    # Convert dataframe to array
    arr = cbdf2cbarr(df, metadata)

    # New filename.
    cubeName = os.path.join(searchPath, str(newFilename) + '.tif')
    # Write to disk timeseries cube.
    if len(bands) == 1:
        with rasterio.open(cubeName, 'w', **metadata) as dst:
            dst.write(arr)
    else:
        with rasterio.open(cubeName, 'w', **metadata) as dst:
            for id, _ in enumerate(bands, start=1):
                print(id, sep=' ', end=' ', flush=True)
                k = id-1
                dst.write_band(id, arr[k, :, :].astype(metadata['dtype']))
    return None





def writeCube(listOfPaths, searchPath, newFilename, dtype, sort=False, **kwargs):
    """ Stack satellite images (FROM DIFFERENT FILES) as timeseries cube, without loading them in memory.
    If there is a datetime field in filename, could enable sort=True, to sort cube layers by date, ascending.
    Also, if sort=True, dates are written at .txt file which will be saved with the same output name, as cube.

    Args:
        listOfPaths (list of strings): Paths of images which will participate in cube.
        searchPath (string): Where the result will be saved. Fullpath, ending to dir.
        newFilename (string): Not a full path. Only the filename, without format ending.
        dtype (string): Destination datatype.
        sort (bool (optional)): If True, sorts cube layers by date, extracted from paths. By default is None.
    Return:
        datetimes (list of dates): Dates in stacked order.
        metadata (dictionary): Metadata of written cube.
    """

    # Open a random image from images to keep metadata.
    with rasterio.open(listOfPaths[0]) as src:
        # Image metadata.
        metadata = src.meta
        # Update third dimension in metadata, as expected for cube.
        metadata.update({'dtype': dtype, 'count': len(listOfPaths), 'driver':'GTiff'})

    if sort == False:
        datetimes = None
        pass
    else:
        # Correctly sorted dates. 
        listOfPaths = sorted(listOfPaths, key=_get_pattern)

        # Keep datetimes in list & write to file.
        datetimes = [_get_pattern(path).date() for path in listOfPaths]
        # Export datetimes to file.
        with open(os.path.join(searchPath, str(newFilename) + '.txt') , 'w') as myfile:
            myfile.write('\n'.join([item.strftime('%Y-%m-%d') for item in datetimes]))

    # New filename.
    cubeName = os.path.join(searchPath, str(newFilename) + '.tif')
    # Stack products as timeseries cube.
    with rasterio.open(cubeName, 'w', **metadata) as dst:
        for id, layer in enumerate(listOfPaths, start=1):
            # Print cube layer ID and corresponding date -or not-.
            if not datetimes:
                print(id, sep=' ', end=' ', flush=True)
            else:
                print(id, datetimes[id-1])

            with rasterio.open(layer) as src:
                dst.write_band(id, src.read(1).astype(dtype))
                band_name = os.path.split(src.name)[-1].split('.')[0]
                dst.set_band_description(id, band_name)

    logging.info("Metadata of written cube are:\n{}".format(metadata))
    return datetimes, metadata




def cbInMem(listOfPaths, sort=False):
    """ Create 3d cube in memory from paths of different bands.
    Args:
        listOfPaths (list of strings): Fullpaths of individual bands.
        sort (boolean, optional): Sort fullpaths by date.
    Return:
        cbarr (3d array): Indexed as tensor (count:bands, height:rows, width:columns)
    """

    # Correctly sorted fullpaths, by date. 
    if sort == False:
        pass
    else:
        listOfPaths = sorted(listOfPaths, key=_get_pattern)

    # Read metadata of random image
    with rasterio.open(listOfPaths[0], 'r') as src:
        metadata = src.meta

    # Preallocate a zero array with corresponding dimensions
    temp = np.zeros((1, metadata['height'], metadata['width']))

    # Stack arrays as cube
    for bandpath in listOfPaths[0:5]:
        with rasterio.open(bandpath, 'r') as src:
            arr = src.read()
        
        cbarr = np.concatenate([temp, arr])
        temp = cbarr

    return cbarr[1:, :, :]




def extremeDOY(cbdf, dates, mode='max'):
    """ Compute DOYs of min or max value for every pixel's depth.
    Args:
        cbdf (pandas dataframe): Cube dataframe. Indexed as
                                (rows:bands, row wise read, columns:individual pixels)
        dates (list of strings): List containing dates from which is cube constructed.
                        This information is produced from writeCube() function.
        mode (string, optional): By default computes DOYs for 'max' values. Set to 'min' to compute
                        DOYs for min values.
    Return:
        res (pandas series): Day of year of correspoding value.
    """
    if mode == 'max':
        # Find index of max value, for every pixel. Starts from zero, skips NaN.
        res = cbdf.idxmax()
    elif mode == 'min':
        # Find index of min value, for every pixel. Starts from zero, skips NaN.
        res = cbdf.idxmin()
    else:
        print("mode = 'min' OR 'max'")

    print("Dates found for {} values from whole image: {}".format(mode, res.unique()))
    # Replace all index-of-week with corresponding date from .txt file, converted to datetime object
    w = res.unique()
    # Drop NaN values. Cuz idxmax() returns NaN if all entries are NaN.
    w = w[~np.isnan(w)]
    for value in w:
        res.replace(value, dt.datetime.strptime(dates[int(value)], '%Y-%m-%d'), inplace=True)

    # Convert datetime objects to day of year
    return res.dt.dayofyear
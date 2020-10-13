#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import os
import rasterio
from rasterio.features import shapes
import cv2
import fiona
import datetime as dt
import logging

logger = logging.getLogger(__name__)
# Override the default severity of logging.
logger.setLevel('INFO')
# Use StreamHandler to log to the console.
stream_handler = logging.StreamHandler()
# Don't forget to add the handler.
logger.addHandler(stream_handler)


def resampleBand(input_im_full_path, before, after, output_name=None, **kwargs):
    """ Upsample one-band image, to half pixelsize (e.g. from 20m to 10m).
        Save result to the same folder of input image.
    Args:
        input_im_full_path (string): Fullpath to input-image.
        before (int): Pixel resolution before resize.
        after (int): Pixel resolution after resize.
        output_name (string, optional): Filename for output, not a fullpath. Without format ending.
    Return:
        None
    """

    _splitted_path = os.path.split(input_im_full_path)
    # If filename not given by user.
    if output_name == None:
        output_name = _splitted_path[-1].split('.')[0] + str(after) + "m"
    else:
        pass
    
    # Construct new filname.
    nfilename = os.path.join(_splitted_path[0], output_name + ".tif")

    if os.path.exists(nfilename) == True and os.stat(nfilename).st_size != 0:
        # Pass if file already exists & it's size is not zero.
        return

    # Read input-image as array.
    with rasterio.open(input_im_full_path) as _src:
        metadata = _src.meta
        minx, maxy = (metadata['transform'][2], metadata['transform'][5])
        ratio = before//after
        arr = _src.read(1)

    # Just to be sure resize is correct.
    if int(metadata['transform'][0]) != before:
        return

    # Upsample.
    out_img = cv2.resize(arr, (ratio*arr.shape[0], ratio*arr.shape[1]), interpolation=cv2.INTER_LINEAR)

    # Create the new transformation.
    transf = rasterio.transform.from_origin(minx, maxy, after, after)

    # Update metadata.
    metadata.update(driver='GTiff', height=out_img.shape[0], width=out_img.shape[1], transform=transf)

    # Just to be sure resize is correct.
    if int(metadata['transform'][0]) != after:
        return

    # Write to disk resampled-image.
    with rasterio.open(nfilename, "w", **metadata) as dest:
        dest.write(out_img.astype(metadata['dtype']), 1)
        
    return None




def normalizeCommonLayers(listOfPaths, destDtype, overwrite=False, **kwargs):
    """  Normalize common bands of different dates, from different files,
        to selected dtype range, and save to disk.

    Args:
        listOfPaths (list of strings): Fullpaths of common bands.
        destDtype (string): Destination data type name supported by rasterio lib.
        overwrite (boolean, optional): If False, output has new filename &
                        is written to directory where input lives. If True,
                        input is overwritten by output.
    Return:
        None
    """

    dtype_ranges = {
    'int8': (-128, 127),
    'uint8': (0, 255),
    'uint16': (0, 65535),
    'int16': (-32768, 32767),
    'uint32': (0, 4294967295),
    'int32': (-2147483648, 2147483647),
    'float32': (-3.4028235e+38, 3.4028235e+38),
    'float64': (-1.7976931348623157e+308, 1.7976931348623157e+308)}

    # Find global min & max from every index.
    mns = []
    mxs = []
    for im in listOfPaths:
        with rasterio.open(im) as src:
            arr = src.read(1)
            metadata = src.meta
            _mn = arr.min()
            _mx = arr.max()
        mns.append(_mn)
        mxs.append(_mx)

    globmin = min(mns)
    globmax = max(mxs)

    metadata.update(dtype=destDtype)
    # Normalize every image of current indice to range of selected dtype, based on min(mins) and max(maxes).
    for im in listOfPaths:
        with rasterio.open(im) as src:
            arr = src.read(1)
            # MinMax Normalization Formula.
            normarr=(dtype_ranges[destDtype][1]-dtype_ranges[destDtype][0])/(globmax-globmin)*(arr-globmax)+dtype_ranges[destDtype][1]
            # New filename, if overwrite=False.
            if not overwrite:
                _p1, _p2, _p3 = os.path.split(im)[0], os.path.splitext(os.path.basename(im))[0], os.path.splitext(os.path.basename(im))[1]
                im = os.path.join(_p1, _p2 + '_norm_'+ destDtype + _p3)
            with rasterio.open(im, "w", **metadata) as dest:
                dest.write_band(1, normarr.astype(destDtype))
    return None




def vectorize(raster_file, metadata, vector_file, driver, mask_value=None, **kwargs):
    """ Extract vector from raster. Vector propably will include polygons with holes.
    
    Args:
        raster_file (ndarray): raster image.
        src (DatasetReader type): Keeps path to filesystem.
        vector_file (string): Pathname of output vector file.
        driver (string): Kind of vector file format.
        mask_value (float or integer): No data value.
    
    Returns:
        None. Saves folder containing vector shapefile to cwd or to given path.
    """
    start = dt.datetime.now()

    if mask_value is not None:
        mask = raster_file == mask_value
    else:
        mask = None
    
    logging.debug("Extract id, shapes & values...")
    features = ({'properties': {'raster_val': v}, 'geometry': s} for i, (s, v) in enumerate(
            # The shapes iterator yields geometry, value pairs.
            shapes(raster_file, mask=mask, connectivity=4, transform=metadata['transform'])))

    logging.debug("Save to disk...")
    with fiona.Env():
        with fiona.open(
                vector_file, 'w', 
                driver = driver,
                crs = metadata['crs'],
                schema = {'properties': [('raster_val', 'int')], 'geometry': 'Polygon'}) as dst:
            dst.writerecords(features)

    end = dt.datetime.now()
    logging.info("Elapsed time to vectorize raster to shp {}:\n{} mins".format(
        vector_file, (int((end-start).seconds/60))))
    return None




def gml2shp(sourcedataset, outputname=None, **kwargs):
    """ Convert format, from file.gml to file.shp & save to disk.

    Args:
        wdir (string): Fullpath of containing folder. Path to find source dataset & to save results.
        outnameshp (string, optional): Not fullpath. New filename for output shapefile.
                            Without format ending. By default uses the source filename.
    Return:
        None
    """
    # Path to save results.
    savepath = sourcedataset.split('/')
    # This maybe will be usefull in casse of a new filename.
    output = savepath[-1].split('.')[0]
    # 'splat' operator converts list-of-strings to path.
    savepath = "/" + os.path.join(*savepath[0:-1])
    # Change current working dorection.
    os.chdir(savepath)

    # New file's name.
    if outputname == None:
        outputname = output + '.shp'
    else:
        outputname = outputname + '.shp'

    # Execute command to terminal.
    cmd = "ogr2ogr -f 'ESRI SHAPEFILE' -a_srs 'EPSG:32634' " + outputname + " " + sourcedataset
    os.system(cmd)
    return None
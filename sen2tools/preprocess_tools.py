#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import os
import rasterio
from rasterio.features import shapes
from rasterio.plot import show
from rasterio.windows import Window
from shapely.geometry import box
import geopandas as gpd
from fiona.crs import from_epsg
import cv2
import fiona
import datetime as dt
from typing import List
import logging


logger = logging.getLogger(__name__)
# Override the default severity of logging.
logger.setLevel('INFO')
# Use StreamHandler to log to the console.
stream_handler = logging.StreamHandler()
# Don't forget to add the handler.
logger.addHandler(stream_handler)




class GeoImClip:
    def __init__(self, im:str, newname_flag:str, write:bool=True, dstdir:str=None):
        """Clip one band or multiband image, using three methods. By image coordinates,
        by coordinates in the same CRS as the image is, or by selecting a quarter of the image
        counting clockwise.

        Args:
            im (str): Fullpath to an image or cube-image.
            newname_flag (str): A flag which will be added to the end of the filename,
                in order to form the new filename.
            write (bool, optional): Save the image. Destination path is required. Defaults to True.
            dstdir (str, optional): Destination path. Defaults to None.
        """
        self.im = im
        self.newname_flag = newname_flag
        self.write = write
        self.dstdir = dstdir

    def byImCoords(self, imCoords:List[int]):
        """Clip one band or multiband image by image-coordinates.

        Args:
            imCoords (List[int]): [row_start, row_stop, col_start, col_stop, band_start, band_stop]

        Returns:
            tuple: The final result as array, corresponding metadata as rasterio-dictionary-metadata
                and the band name as string.
        """

        if len(imCoords) != 6:
            logger.exception(f"Six coordinates should be given in a list. E.g. [row_start, row_stop, col_start, col_stop, band_start, band_stop]")

        # Image coordinates
        row_start, row_stop, col_start, col_stop, band_start, band_stop = imCoords

        # Construct a window by image coordinates.
        win = Window.from_slices(slice(row_start, row_stop), slice(col_start, col_stop))     

        # Image dimensions
        rows = row_stop-row_start
        cols = col_stop-col_start
        bands = [i for i in range(band_start+1, band_stop+1)]

        with rasterio.open(self.im) as src:
        
            # Find top-left x, y coordinates of new image, from selected starting row, col
            new_left_top_coords = src.xy(row_start, col_start, offset='ul')
            
            # Original image metadata
            metadata = src.meta
            band_name = src.name
            
            # Assume that pixel is cubic
            pixelSize = list(metadata['transform'])[0]
            
            # Create the new transformation
            transf = rasterio.transform.from_origin(
                new_left_top_coords[0], new_left_top_coords[1], pixelSize, pixelSize)

            # Update metadata. Can be used to save new geolocated image
            metadata.update(height=rows , width=cols, count=len(bands), transform=transf)
            
            # Load image as array
            out_img = src.read(bands, window=win)

        if self.write == True:

            # Reshape as rasterio needs the shape.
            temp = out_img.reshape(1, out_img.shape[0], out_img.shape[1])

            # Output filename
            out_tif = os.path.join(self.dstdir, band_name +'_'+ self.newname_flag + '.tif')

            # Write output image to disk
            with rasterio.open(out_tif, "w", **metadata) as dest:
                dest.write(temp)
            
        return out_img, metadata, band_name



    def byXYcoords(self, coords:List[float]):
        """Clip one band or multiband image by coordinates. Image and coordinates should have the same CRS.

        Args:
            coords (List[float]): [minx, maxx, miny, maxy]

        Returns:
            tuple: The final result as array, corresponding metadata as rasterio-dictionary-metadata
                and the band name as string.
        """
        
        if len(coords) != 4:
            logger.exception(f"Six coordinates should be given in a list. E.g. [minx, maxx, miny, maxy]")

        # Coordinates
        minx, miny, maxx, maxy = coords
    
        with rasterio.open(self.im) as src:

            # Convert x,y to row, col. Image and coordinates should have the same CRS.
            row_start, col_start = src.index(minx, maxy)
            row_stop, col_stop = src.index(maxx, miny)
        
        return self.byImCoords([row_start, row_stop, col_start, col_stop])
                
                            
                                
    def byQuarters(self, quarter:int):
        """Clip one band or multiband image by selecting a quarter, counting clockwise.

        Args:
            quarter (int): 1 or 2 or 3 or 4

        Returns:
            tuple: The final result as array, corresponding metadata as rasterio-dictionary-metadata
                and the band name as string.
        """

        with rasterio.open(self.im) as src:

            if quarter == 1:
                minx = src.bounds.left
                maxy = src.bounds.top
                maxx = src.bounds.left + (src.bounds.right - src.bounds.left)/2 # middle point
                miny = src.bounds.top - (src.bounds.top - src.bounds.bottom)/2 # middle point

            elif quarter == 2:
                minx = src.bounds.left + (src.bounds.right - src.bounds.left)/2 # upper middle point
                maxy = src.bounds.top # upper middle point
                maxx = src.bounds.right # right middle point
                miny = src.bounds.bottom + (src.bounds.top - src.bounds.bottom)/2 # right middle point

            elif quarter == 3:
                minx = src.bounds.left + (src.bounds.right - src.bounds.left)/2 # middle point
                maxy = src.bounds.top - (src.bounds.top - src.bounds.bottom)/2 # middle point
                maxx = src.bounds.right
                miny = src.bounds.bottom

            elif quarter == 4:
                minx = src.bounds.left # left middle point
                maxy = src.bounds.bottom + (src.bounds.top - src.bounds.bottom)/2 # left middle point
                maxx = src.bounds.left + (src.bounds.right - src.bounds.left)/2 # lower middle point
                miny = src.bounds.bottom # lower middle point

            else:
                logger.error('Quadrant must be integer in range [1,4].')

        return self.byXYcoords([minx, maxx, miny, maxy])

        



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




def vectorize(raster_file, metadata, vector_file, driver, mask_value=None, dropRasterVal=None, **kwargs):
    """ Extract vector from raster. Vector propably will include polygons with holes.
    
    Args:
        raster_file (ndarray): raster image.
        src (DatasetReader type): Keeps path to filesystem.
        vector_file (string): Pathname of output vector file.
        driver (string): Kind of vector file format.
        mask_value (float or integer): No data value.
        dropRasterVal : A raster value we want to omit from vectorization
    
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
            shapes(raster_file, mask=mask, connectivity=4, transform=metadata['transform'])) if v != dropRasterVal)


    logging.debug("Save to disk...")
    with fiona.Env():
        with fiona.open(
                vector_file, 'w', 
                driver = driver,
                crs = metadata['crs'],
                encoding = 'utf-8',
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
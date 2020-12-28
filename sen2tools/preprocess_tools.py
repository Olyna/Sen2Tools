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
    def __init__(self, imPaths:List[str], mode:int, coords:List[float]=None,  srid:int=None, quad:int=None):
        """Clip image (one or more) using bounding box from given coordinates, or one quadrant of
        the original image.

        Args:
            imPaths (List[str]): List containing fullpaths to images which are going to be clipped.
            mode (int): Using Coordinates or Using Quadrants
            coords (List[float], optional): Coordinates of bounding box when mode==1. Defaults to None.
            srid (int, optional): CRS of given coordinates when mode==1. Defaults to None.
            quad (int, optional): One of four quadrants when mode==2. Defaults to None.
        """
        self.imPaths = imPaths

        if mode not in [1, 2]:
            logger.exception(f"Argument 'mode' could be 1 or 2. Given is {mode}")

        if mode == 1:
            # Using Coordinates
            if len(coords) != 4:
                logger.exception(f"Four coordinates should be given in a list. E.g. [minx, maxx, miny, maxy]")
            if srid is None:
                logger.exception(f"Coordinates' CRS must be given")

            minx, maxx, miny, maxy = coords
            self.geometry = self.boundingBox(minx, maxx, miny, maxy, srid)
            
        elif mode == 2:
            # Using Quadrants
            if quad not in [1, 2, 3, 4]:
                logger.exception(f"Acceptable quadrant is 1 or 2 or 3 or 4")

            minx, maxx, miny, maxy = self.quadrants_coords(self.imPaths[0], quad)
            with rasterio.open(self.imPaths[0]) as src:
                srid = src.meta['crs'].to_epsg()
            self.geometry = self.boundingBox(minx, maxx, miny, maxy, srid)

        else:
            logger.error(" .. ")



    def boundingBox(self, minx:float, maxx:float, miny:float, maxy:float, srid:int):
        """Generates polygon geometry, from given coordinates.
        Given coordinates must be in the same CRS, as image's CRS.

        Args:
            minx (float): left = west
            maxx (float): right = east
            miny (float): bottom = south
            maxy (float): top = north
            srid (int): coordinates reference system id.

        Returns:
            geodataframe: polygon geometry
        """
        # Create bounding box
        bbox = box(minx, miny, maxx, maxy)
        # Return geometry as GeoDataFrame
        return gpd.GeoDataFrame({'geometry': bbox}, index=[0], crs=from_epsg(srid))




    @staticmethod
    def quadrants_coords(raster_file:str, quadrant:int):
        """Clip selected quadrant of raster file.

        Args:
            raster_file (str): Raster image file fullpath.
            quadrant (int): One of the four pieces. Start counting from upper left corner, clockwise.
                Integer in range [1,4].
        """

        with rasterio.open(raster_file) as src:

            if quadrant == 1:
                minx = src.bounds.left
                maxy = src.bounds.top
                maxx = src.bounds.left + (src.bounds.right - src.bounds.left)/2 # middle point
                miny = src.bounds.top - (src.bounds.top - src.bounds.bottom)/2 # middle point

            elif quadrant == 2:
                minx = src.bounds.left + (src.bounds.right - src.bounds.left)/2 # upper middle point
                maxy = src.bounds.top # upper middle point
                maxx = src.bounds.right # right middle point
                miny = src.bounds.bottom + (src.bounds.top - src.bounds.bottom)/2 # right middle point

            elif quadrant == 3:
                minx = src.bounds.left + (src.bounds.right - src.bounds.left)/2 # middle point
                maxy = src.bounds.top - (src.bounds.top - src.bounds.bottom)/2 # middle point
                maxx = src.bounds.right
                miny = src.bounds.bottom

            elif quadrant == 4:
                minx = src.bounds.left # left middle point
                maxy = src.bounds.bottom + (src.bounds.top - src.bounds.bottom)/2 # left middle point
                maxx = src.bounds.left + (src.bounds.right - src.bounds.left)/2 # lower middle point
                miny = src.bounds.bottom # lower middle point

            else:
                logger.error('Quadrant must be integer in range [1,4].')

        return(minx, maxx, miny, maxy)


    @staticmethod
    def clip(im:str, geometry, newname_flag:str, resize=False, write=True):
        """Clip image & update metadata of output image. Option to write
        output image to disk.

        Args:
            im (str): Path to image.
            geometry (GeoDataFrame): Bounding box to clip image.
            newname_flag (str): Piece of string added to the end of the new filename.
            resize (bool, optional): Whether to resize output raster. Defaults to False.
            write (bool, optional): Whether to save output raster to disk. Defaults to True.

        Returns:
            out_img (array): clipped array.
            out_meta (dictionary): updated metadata for clipped raster.
        """

        # New name for output image. Split on second occurence of dot.
        out_tif = im.split('.')[0]+ '.'+ im.split('.')[1] + str(newname_flag) + '.tif'

        if os.path.exists(out_tif) == True and os.stat(out_tif).st_size != 0:
            # Pass if file already exists & it's size is not zero.
            return

        with rasterio.open(im) as src:
            # Image metadata.
            metadata = src.meta

            # # It doesn't work well. It changes orinal minx & maxy.
            # # Convert window's CRS to image's CRS.
            # geom = geometry.to_crs(crs=metadata['crs'])

            # Convert x,y to row, col.
            row_start, col_start = src.index(geometry.bounds['minx'][0], geometry.bounds['maxy'][0])
            row_stop, col_stop = src.index(geometry.bounds['maxx'][0], geometry.bounds['miny'][0])

            # Parse pixel size from metadata.
            pixelSize = list(metadata['transform'])[0]

            # Create the new transformation.
            transf = rasterio.transform.from_origin(
                geometry.bounds['minx'][0], geometry.bounds['maxy'][0], pixelSize, pixelSize)

            # Update metadata.
            metadata.update(
                driver='GTiff', transform=transf,
                height=(row_stop-row_start), width=(col_stop-col_start))

            # Construct a window by image coordinates.
            win = Window.from_slices(slice(row_start, row_stop), slice(col_start, col_stop))

            # Clip image.
            out_img = src.read(1, window=win)

        if resize == True:
            # Create the new transformation.
            transf = rasterio.transform.from_origin(
                geometry.bounds['minx'][0], geometry.bounds['maxy'][0], pixelSize//2, pixelSize//2)

            # Update metadata for output image
            metadata.update({"height": out_img.shape[0]*2,
                        "width": out_img.shape[1]*2,
                        "transform": transf})
            # Upsample.
            out_img = cv2.resize(
                out_img, (2*out_img.shape[0], 2*out_img.shape[1]), interpolation=cv2.INTER_LINEAR)

        if write == True:
            # Reshape as rasterio needs the shape.
            temp = out_img.reshape(1, out_img.shape[0], out_img.shape[1])
            # Write output image to disk
            with rasterio.open(out_tif, "w", **metadata) as dest:
                dest.write(temp)

            # # Plot output image
            # clipped = rasterio.open(out_tif)
            # #show((clipped, 1), cmap='terrain')

        return out_img, metadata





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
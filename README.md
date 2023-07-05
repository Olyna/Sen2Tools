# Sen2Tools

Tools to manipulate Sentinel-2 -& not only!- satellite data, as georeferenced raster/vector data.

## Module search_tools

#### findMore(searchPath, startsWith, contains, endsWith, mode, sort=True, **kwargs)

Search for directories or files under the given searchPath.

Args:
* searchPath (string): From where searching starts.
* startsWith (string): Prefix of wanted filename.
* contains (string): Text contained in wanted filename.
* endsWith (string): Ending or file-format of wanted filename.
* mode (int): 1 = search for dirs OR 2 = search for files.
* sort (boolean, optional): True by default, sorts itemsFound by date.

Return:
* itemsFound (list of strings): List with fullpaths of itemsFound, sorted be date.


---------------------------------------------------------------------


#### metaSearch(searchPath, lessThan, **kwargs)

Select fullpaths of Sentinel-2 scenes, by cloud coverage.
Reads MTD.xml metadata file.
Keep images with cloud coverage less than given percentage.

Args:
* searchPath (string): From where searching starts.
* lessThan (float): Cloud coverage value to campare with.

Return:
* itemsFound (list of strings): List with fullpaths of itemsFound, sorted be date.


---------------------------------------------------------------------


#### find(searchPath, pattern, mode, sort=True, **kwargs)

Search for directories or files under the given searchPath, ending by pattern.

Args:
* searchPath (string): From where searching starts.
* pattern (string): End of path or file looking for. For files, must include format.
* mode (int): 1 = search for dirs OR 2 = search for files.
* sort (boolean, optional): True by default, sorts itemsFound by date.

Return:
* itemsFound (list of strings): List with fullpaths of itemsFound, sorted by date.


---------------------------------------------------------------------


#### findRecord(searchPath, satPath, satRow, year, sort=True, **kwargs)

Search for Sentinel-2 scene folders, by satellite's path, row & year.

Args:
* searchPath (string): From where searching starts.
* satPath, satRow (string): Tile path-row, each as 3 digit number.
* year (string or integer: Year searching for.
* sort (boolean, optional): True by default, sorts itemsFound by date.

Return:
* itemsFound (list of strings): List with fullpaths of itemsFound, sorted by date.




## Module preprocess_tools

#### resampleBand(input_im_full_path, before, after, output_name=None, **kwargs)

Upsample one-band image, to half pixelsize (e.g. from 20m to 10m).
Save result to the same folder of input image.

Args:
* input_im_full_path (string): Fullpath to input-image.
* before (int): Pixel resolution before resize.
* after (int): Pixel resolution after resize.
* output_name (string, optional): Filename for output, not a fullpath. Without format ending.

Return:
* None


---------------------------------------------------------------------


#### normalizeCommonLayers(listOfPaths, destDtype, overwrite=False, **kwargs)

Normalize common bands of different dates, to selected dtype range.

Args:
* listOfPaths (list of strings): Fullpaths of common bands.
* destDtype (string): Destination data type name supported by rasterio lib.
* overwrite (boolean, optional): If False, output has new filename &
                    is written to directory where input lives. If True,
                    input is overwritten by output.

Return:
* None


---------------------------------------------------------------------


#### vectorize(raster_file, metadata, vector_file, driver, mask_value=None, **kwargs)

Extract vector from raster. Vector propably will include polygons with holes.

Args:
* raster_file (ndarray): raster image.
* src (DatasetReader type): Keeps path to filesystem.
* vector_file (string): Pathname of output vector file.
* driver (string): Kind of vector file format.
* mask_value (float or integer): No data value.

Return:
* None. Saves folder containing vector shapefile to cwd or to given path.


---------------------------------------------------------------------


#### gml2shp(sourcedataset, outputname=None, **kwargs)

Convert format, from file.gml to file.shp & save to disk.

Args:
* wdir (string): Fullpath of containing folder. Path to find source dataset & to save results.
* outnameshp (string, optional): Not fullpath. New filename for output shapefile.
                            Without format ending. By default uses the source filename.

Return:
* None




## Module cube_tools

#### cubePart(imPath, row_start, row_stop, col_start, col_stop, band_start, band_stop, **kwargs)

Returns part of cube data as 3D array, dataframe and metadata of returned subset.
Dataframe rows correspond to images / bands. Dataframe column corresponds to one pixel's depth.

Args:
* imPath (string): Cube's fullpath.
* row_start, row_stop, col_start, col_stop, band_start, band_stop (int): Image coordinates,
                                                                    starting counting from zero.

Return:
* cube (numpy array): 3D numpy array.
* cube_df (pandas dataframe): Every row is one cube's image, every column is a pixel.
* metadata (dictionary): New image's updated metadata.


---------------------------------------------------------------------


#### readCube(imPath, **kwargs)

Read an image as 3D array, dataframe & corresponding metadata.

Args:
* imPath (string): Fullpath to multiband image.

Return:
* cube (numpy array): 3D numpy array.
* cube_df (pandas dataframe): Every row is one cube's image, every column is a pixel.
* metadata (dictionary): Metadata of original cube.

---------------------------------------------------------------------


#### dataframe2tifCube(df, metadata, newFilename, searchPath, **kwargs)

Writes a dataframe on disk, with georeference.

Args:
* df (pandas dataframe): Every row is one cube's image, every column is a pixel.
* metadata (dictionary): Metadata of original cube.
* newFilename (string): Not a full path. Only the filename, without format ending.
* searchPath (string): Fullpath, where the result will be saved.

Return:
* None

---------------------------------------------------------------------


#### writeCube(listOfPaths, searchPath, newFilename, dtype, sort=False, **kwargs)

Stack images (FROM DIFFERENT FILES) as timeseries cube, without loading them in memory.
If there is a datetime field in filename, could enable sort=True, to sort cube layers by date, ascending.
Also, if sort=True, dates are written at .txt file which will be saved with the same output name, as cube.

Args:
* listOfPaths (list of strings): Paths of images which will participate in cube.
* searchPath (string): Where the result will be saved. Fullpath, ending to dir.
* newFilename (string): Not a full path. Only the filename, without format ending.
* dtype (string): Destination datatype.
* sort (bool (optional)): If True, sorts cube layers by date, extracted from paths.
                            By default is None.

Return:
* datetimes (list of dates): Dates in stacked order.
* metadata (dictionary): Metadata of written cube.
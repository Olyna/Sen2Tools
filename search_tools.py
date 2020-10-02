#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
from os import walk
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
    """ Returns the date extracted from Sentinel-2 fullpath filenames.
    Args:
        oneFullpath (string): Fullpath.
    Returns datetime object.
    """
    return dt.datetime.strptime(oneFullpath.split('.SAFE')[0].split('_')[-1][0:8], '%Y%m%d')



def findMore(searchPath, startsWith, contains, endsWith, mode, sort=True, **kwargs):
    """ Search for directories or files under the given searchPath.

    Args:
    searchPath (string): From where searching starts.
    startsWith (string): Prefix of wanted filename.
    contains (string): Text contained in wanted filename.
    endsWith (string): Ending or file-format of wanted filename.
    mode (int): 1 = search for dirs OR 2 = search for files.
    sort (boolean, optional): True by default, sorts itemsFound by date.

    Return:
    itemsFound (list of strings): List with fullpaths of itemsFound, sorted be date.
    """

    itemsFound = []
    for (dirpath, dirnames, filenames) in walk(searchPath):
        # Search for directories.
        if mode == 1:
            for dirname in dirnames:
                if dirname.startswith(startsWith) and dirname.endswith(endsWith) and contains in dirname:
                    itemsFound.append(os.path.join(dirpath, dirname))

        # Search for files.
        elif mode == 2:
            for filename in filenames:
                if filename.startswith(startsWith) and filename.endswith(endsWith) and contains in filename:
                    itemsFound.append(os.path.join(dirpath, filename))
        else:
            logger.error("Select search-mode, dir or file.")

    # Correctly sorted fullpaths, by date. 
    if sort == False:
        pass
    else:
        itemsFound = sorted(itemsFound, key=_get_pattern)

    logger.info("For given pattern '{}'*'{}'*'{}', found {} results...".format(
        startsWith, contains, endsWith, len(itemsFound)))

    return (itemsFound)




def metaSearch(searchPath, lessThan, **kwargs):
    """ Select fullpaths of Sentinel-2 scenes, by cloud coverage. Reads
    MTD.xml metadata file. Keep images with cloud coverage less than given percentage.

    Args:
    searchPath (string): From where searching starts.
    lessThan (float): Cloud coverage value to campare with.

    Return:
    itemsFound (list of strings): List with fullpaths of itemsFound, sorted by date.
    """
    import xml.etree.ElementTree as ET

    # Find all metadata files in given directory.
    meta_files = find(searchPath, 'MTD_TL.xml', 2)

    itemsFound = []
    for f in meta_files:
        root = ET.parse(f).getroot()
        type_tag = root.find('{https://psd-14.sentinel2.eo.esa.int/PSD/S2_PDI_Level-2A_Tile_Metadata.xsd}Quality_Indicators_Info/Image_Content_QI')
        value = type_tag.find('CLOUDY_PIXEL_PERCENTAGE')

        if float(value) <= float(lessThan):
            path = f.split('.SAFE')[0] + '.SAFE'
            itemsFound.append(path)

            logger.info("{} --> Image accepted...".format(value))
        else:
            logger.info("{} --> Image rejected...".format(value))
    
    return itemsFound



def find(searchPath, pattern, mode, sort=True, **kwargs):
    """ Search for directories or files under the given searchPath, ending by pattern.

    Args:
    searchPath (string): From where searching starts.
    pattern (string): End of path or file looking for. For files, must include format.
    mode (int): 1 = search for dirs OR 2 = search for files.
    sort (boolean, optional): True by default, sorts itemsFound by date.

    Return:
    itemsFound (list of strings): List with fullpaths of itemsFound, sorted by date.
    """

    itemsFound = []
    for (dirpath, dirnames, filenames) in walk(searchPath):
        # Search for directories.
        if mode == 1:
            for dirname in dirnames:
                if dirname.endswith(str(pattern)):
                    itemsFound.append(os.path.join(dirpath, dirname))

        # Search for files.
        elif mode == 2:
            for filename in filenames:
                if filename.endswith(str(pattern)):
                    itemsFound.append(os.path.join(dirpath, filename))
        else:
            logger.error("Select search-mode, dir or file.")

    # Correctly sorted fullpaths, by date. 
    if sort == False:
        pass
    else:
        itemsFound = sorted(itemsFound, key=_get_pattern)

    logger.debug("For pattern '{}', found {} results.".format(pattern, len(itemsFound)))

    return (itemsFound)




def findRecord(searchPath, satPath, satRow, year, sort=True, **kwargs):
    """ Search for Sentinel-2 scene folders, by satellite's path, row & year.

    Args:
    searchPath (string): From where searching starts.
    satPath, satRow (string): Tile path-row, each as 3 digit number.
    year (string or integer: Year searching for.
    sort (boolean, optional): True by default, sorts itemsFound by date.

    Return:
    itemsFound (list of strings): List with fullpaths of itemsFound, sorted by date.
    """

    # Check requested path and row, as given from user.
    if len(str(satPath)) != 3:
        print('Please supply Path with three digits in total.\ne.g. 081')
    if len(str(satRow)) != 3:
        print('Please supply Row with three digits in total.\ne.g. 036')
    
    itemsFound = []
    # Search every directory under the searchPath.
    for (dirpath, dirnames, _) in walk(searchPath):
        # For every folder
        for dirname in dirnames:
            # If folder includes path-row and date.
            if str(satPath)+str(satRow)+'_'+str(year) in dirname:
                # Gather paths to folder.
                itemsFound.append(os.path.join(dirpath, dirname))

    # Correctly sorted fullpaths, by date. 
    if sort == False:
        pass
    else:
        itemsFound = sorted(itemsFound, key=_get_pattern)

    logger.info("For given pattern '{}' * '{}' * '{}', found {} results...".format(
        satPath, satRow, year, len(itemsFound)))

    return (itemsFound)
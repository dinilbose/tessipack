# This module helps to apply flags to a data set.
import pandas as pd
import numpy as np

def search_multiple_strings_in_file(file_name, list_of_strings):
    """Get line from the file along with line numbers, which contains any string from the list"""
    line_number = 0
    list_of_results = []
    # Open the file in read only mode
    with open(file_name, 'r') as read_obj:
        # Read all lines in the file one by one
        for line in read_obj:
            line_number += 1
            # For each line, check if line contains any string from the list of strings
            for string_to_search in list_of_strings:
                if string_to_search in line:
                    # If any string is found in line, then append that line along with line number in list
                    list_of_results.append((string_to_search, line_number, line.rstrip()))
    # Return list of tuples containing matched string, line numbers and lines where string is found
    return list_of_results

def search_string_in_file(file_name, string_to_search):
    """Search for the given string in file and return lines containing that string,
    along with line numbers"""
    line_number = 0
    list_of_results = []
    # Open the file in read only mode
    with open(file_name, 'r') as read_obj:
        # Read all lines in the file one by one
        for line in read_obj:
            # For each line, check if line contains the string
            line_number += 1
            if string_to_search in line:
                # If yes, then add the line number & line as a tuple in the list
                # list_of_results.append((line_number, line.rstrip()))
                list_of_results.append(line.rstrip())

    # Return list of tuples containing line numbers and lines where string is found
    return list_of_results


def read_flagfile(filename='',source=None,cluster=None):
    '''Reads the flag file and returns relevent lines to back'''

    if not source==None:
        search="source="+"'"+source+"'"
        print(search)
        list_of_results=search_string_in_file(filename,search)

    else:
        search="cluster="+"'"+str(cluster)+"'"
        list_of_results=search_string_in_file(filename,search)
    return list_of_results

def flag_to_dictionary(line):
    '''Convert lines into dictionary'''
    b = dict([i.split('=') for i in line.split(",")])
    return b

def apply_flag(filename='',data=None,source=None,cluster=None,apply_source=False,apply_cluster=False):
    # if not data.index.name=='time':
        # data=data.set_index('time',drop=False)
    if apply_source==True:
        if not source==None:
            lines=read_flagfile(filename=filename,source=source)
            print('Flags applied',lines)
            for line in lines:
                values=flag_to_dictionary(line)
                id_mycatalog=eval(values['source'])
                start=eval(values['start'])
                end=eval(values['end'])
                print(start,end)
                # data_removal=data.query('time>=@start & time<=@end')
                #data=data.drop(data_removal.index)
                #data[data_removal.index]=np.nan
                # data.loc[(data.index>=start) & (data.index<=end)]=np.nan
                data.loc[(data.time>=start) & (data.time<=end)]=np.nan

    if apply_cluster==True:
        if not cluster==None:
            lines=read_flagfile(filename=filename,cluster=cluster)
            print('Flags applied',lines)
            for line in lines:
                values=flag_to_dictionary(line)
                id_mycatalog=eval(values['cluster'])
                start=eval(values['start'])
                end=eval(values['end'])
                print(start,end)
                # data_removal=data.query('time>=@start & time<=@end')
                #data=data.drop(data_removal.index)
                #data[data_removal.index]=np.nan
                # data.loc[(data.index>=start) & (data.index<=end)]=np.nan
                data.loc[(data.time>=start) & (data.time<=end)]=np.nan
    return data

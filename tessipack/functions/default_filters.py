import numpy as np
from scipy import stats


def quality_filter(data=None, quality=[]):
    """
    Get back the quality values
    """
    data = data.query('quality == @quality')

    return data


def flux_filter(data='',flux_name='corr_flux',func='median', deviation='mad', sigma=3):
    ''' Filter light curve '''

    data_frame = data

    length=len(data_frame)
    if deviation=='std':
        std=data_frame[flux_name].std()
    if deviation=='mad':
        value_dat=data_frame[flux_name]
        median_value=value_dat.median()
        std = (value_dat - median_value).abs().sum() / len(value_dat)


    if deviation=='mean_abs':
        std=stats.median_absolute_deviation(data_frame[flux_name])

    if func=='zscore':
        z =np.abs(stats.zscore(data_frame[flux_name]))<sigma
        data_frame=data_frame[z]
    if func=='mean':
        mean=data_frame[flux_name].mean()
        data_frame=data_frame[(data_frame[flux_name]<mean+sigma*std)&(data_frame[flux_name]>mean-sigma*std)]
    if func=='median':
        median=data_frame[flux_name].median()
        data_frame=data_frame[(data_frame[flux_name]<median+sigma*std)&(data_frame[flux_name]>median-sigma*std)]


    filter_percent=(len(data_frame)/length)*100
    data_frame['filter_percent']=filter_percent

    return data_frame
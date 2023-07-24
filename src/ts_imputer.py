from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted
import pandas as pd
import numpy as np
import warnings

class TimeSeriesImputer(BaseEstimator, TransformerMixin):
    '''
    Custom Imputer compatible with sklearn pipelines to fill missing values in time series data in a pd.DataFrame

    Parameters
    ----------
    location_index: str
    name of the index with the location information

    time_index (optional): str, [str], default=None
    Information the time component in the index, which is used to sort the data if provided. Make sure to either
    provide this or pass an already sorted dataframe.

    method: str, default='bfill', possible values: ['bfill', 'interpolate']
    Imputation is performed on a location-by-location basis. For correct results, input df needs to be constructed with
    a time and a location index. Df either needs to be sorted by time or the time index needs to be passed to the
    imputer, so the imputation can be performed separately for each location.
    Available methods:
    'bfill': Imputation using bfill where newer data is available combined with ffill where no data for backfilling is
    available.
    'interpolate': Imputation using pandas interpolate. Needs at least 2 non-nan values.

    interp_method: str, default='linear'
    Interpolation method parameter to be passed for pandas.DataFrame.interpolate. Please note that the default option
    does not support extrapolation, for this use e.g. 'slinear'

    interp_tails: str, [str], default='fill', possible values: ['fill', 'extrapolate']
    Fill behaviour for nan tails. Can either be a single string, which applies to both ends, or a list/tuple of length 2
    for end-specific behavior.
    'fill': Fill with last non-nan value in the respective direction.
    'extrapolate': Extrapolate from given observations according to the chosen interpolation method.

    missing_values (optional): default=np.nan
    Value of missing values. If not np.nan, all values in df matching missing_values are replaced
    when calling transform method.

    '''

    def __init__(
            self,
            location_index,
            time_index=None,
            method='bfill', interp_method='linear', interp_tails='fill',
            missing_values=np.nan):
        self.location_index = location_index
        self.time_index = time_index
        self.missing_values = missing_values
        self.method = method
        self.interp_method = interp_method
        self.interp_tails = interp_tails

    def _validate_input(self, X, in_fit):
        # validity check
        try:
            assert type(X) == pd.DataFrame
        except:
            if type(X) == pd.Series:
                X = X.to_frame()
            else:
                raise AssertionError("please pass a pandas DataFrame or Series")
        assert self.location_index in X.index.names
        assert self.method in ['bfill', 'interpolate']
        assert (type(self.interp_tails) is str) or (len(self.interp_tails) == 2)

        if any(X.isna().all()):
            all_nan_cols = X.columns[X.isna().all()].tolist()
            raise ValueError(f'Cannot impute all-nan columns {all_nan_cols}.')

        if (self.method == 'interpolate') and any(X.isna().sum() == len(X)-1):
            single_nan_cols = X.columns[X.isna().sum() == len(X)-1].tolist()
            raise ValueError(f'Cannot interpolate columns with only 1 non-nan value: {single_nan_cols}.')

        if (self.method == 'interpolate') and ('extrapolate' in self.interp_tails) and (self.interp_method == 'linear'):
            raise ValueError(
                'Extrapolation not possible with chosen interpolation method "linear". Only "fill" supported. For '
                'linear extrapolation, please choose interp_method="slinear".'
            )

        # process input
        df = X.copy()
        if not np.isnan(self.missing_values):
            df.replace(self.missing_values, np.nan)

        # no need to process columns without na values when fitting
        if in_fit:
            no_nan_cols = df.columns[(~df.isna()).all()].tolist()
            df = df.drop(columns=no_nan_cols)

            if self.time_index is not None:
                time_index_list = [self.time_index] if type(self.time_index) is str else list(dict.fromkeys(self.time_index))
                sort_levels = [self.location_index] + time_index_list
                df = df.sort_index(level=sort_levels)

        return df

    def fit(self, X, y=None):
        df = self._validate_input(X, in_fit=True)

        update_maps = []
        for loc in df.index.get_level_values(self.location_index).unique():
            df_loc = df.xs(loc, level=self.location_index, drop_level=False)
            if self.method == 'bfill':
                loc_map = df_loc.bfill().ffill()
            elif self.method == 'interpolate':
                loc_map = self._local_fit_interpolate(df_loc, self.interp_method, self.interp_tails)
            update_maps.append(loc_map)

        update_map = pd.concat(update_maps)
        self.update_map_ = update_map
        return self

    def _local_fit_interpolate(self, df_loc, interp_method, interp_tails):
        def get_fill_values():
            if df_loc[col].isna().all():
                message = f'All nan data for location <{df_loc.index.get_level_values(self.location_index)[0]}' \
                          f'>, column <{col}>, no interpolation possible.'
                warnings.warn(message, UserWarning)
                return (np.nan, np.nan)
            else:
                return (df_loc.loc[df_loc[col].first_valid_index(), col], df_loc.loc[df_loc[col].last_valid_index(), col])


        interp_cols = df_loc.columns[df_loc.isna().any()].tolist()
        loc_map = pd.DataFrame(index=df_loc.index)

        for col in interp_cols:

            if type(interp_tails) is str:
                if interp_tails == 'fill':
                    fill_value = get_fill_values()
                    loc_map[col] = df_loc.reset_index()[col].interpolate(
                        method=interp_method, limit_direction='both', fill_value=fill_value
                    ).values
                else:
                    loc_map[col] = df_loc.reset_index()[col].interpolate(
                        method=interp_method, limit_direction='both', fill_value='extrapolate'
                    ).values

            else:
                loc_map[col] = df_loc.reset_index()[col].interpolate(method=interp_method, limit_area='inside').values
                fill_value = get_fill_values()
                limit_direction = ('backward', 'forward')
                for i in range(2):
                    if interp_tails[i] == 'fill':
                        update_data = df_loc.reset_index()[col].interpolate(
                            method=interp_method, limit_direction=limit_direction[i], fill_value=fill_value[i]
                        ).values
                    else:
                        update_data = df_loc.reset_index()[col].interpolate(
                            method=interp_method, limit_direction=limit_direction[i], fill_value='extrapolate'
                        ).values
                    update_series = pd.Series(index=loc_map.index, data=update_data, name=col)
                    loc_map.update(update_series, overwrite=False)

        return loc_map

    def transform(self, X, y=None):
        # make sure that the imputer was fitted
        check_is_fitted(self, 'update_map_')

        df = self._validate_input(X, in_fit=False)

        df.update(self.update_map_, overwrite=False)
        return df

# time_series_imputer
Custom Imputer class for panel data for use in a sklearn pipeline.

Docstring:

    Custom Imputer compatible with sklearn pipelines to fill missing values in panel data in a pd.DataFrame

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

    interp_method: str, default='linear' --currently the only option supported
    Interpolation method parameter to be passed for pandas.DataFrame.interpolate

    interp_tails: str, [str], default='fill', possible values: ['fill', 'extrapolate']
    Fill behaviour for nan tails. Can either be a single string, which applies to both ends, or a list/tuple of length 2
    for end-specific behavior.
    'fill': Fill with last non-nan value in the respective direction.
    'extrapolate': Extrapolate from given observations according to the chosen interpolation method.

    missing_values (optional): default=np.nan
    Value of missing values. If not np.nan, all values in df matching missing_values are replaced
    when calling transform method.

Example use:
    #1
    df = read_some_data()
    
    imp = TimeSeriesImputer(
        location_index='grid_index',
        time_index=['year', context.resources.global_config['sub_year']],
        method='bfill'
    )
    
    df_imputed = imp.fit_transform(df)
    
    #2
    pipe = Pipeline(
        [('impute', imp),
        ('model', RandomForestClassifier())]
    )
    X, y = df[features], df[target]
    
    pipe.fit(X, y)
    
    

# time_series_imputer
Custom Imputer class for panel data for use in a sklearn pipeline, to impute yearly data to monthly data.

For full documentation, see the docstring.

Example use:

    #1: use fit_transform for imputation with prepared dataframe
    df = read_some_panel_data_with_missing_values()
    
    imp = TimeSeriesImputer(
        location_index='location',
        time_index=['year', 'month'],
        method='bfill'
    )
    
    df_imputed = imp.fit_transform(df)
    
    #2: use in a pipeline
    pipe = Pipeline(
        [('impute', imp),
        ('model', RandomForestClassifier())]
    )
    X, y = df[features], df[target]
    
    pipe.fit(X, y)
    
A full example is provided in the jupyter notebook.

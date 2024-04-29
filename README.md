# Description
Custom Imputer class TimeSeriesImputer for panel data for use in a sklearn pipeline, e.g. to impute yearly data to monthly data. Class imputes on a location-by-location basis.

For better (not yet complete) documentation, see the docstring.

# Installation
Currently only directly from github. For the latest version, run
```
pip install git+https://github.com/DaMitti/ts_imputer
```
# Example use:

    #1: use fit_transform for imputation with prepared dataframe
    df = read_some_panel_data_with_missing_values()
    
    imp = TimeSeriesImputer(
        location_index='location',
        time_index=['year', 'month'],
        imputation_method='bfill'
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

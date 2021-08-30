### Probando la hoja de trucos de Pandas y su investigación de uso y campos de instalación.

### Utilizando las hojas de trucos de Pandas, complete cada uno de los ejercicios de pruebas





### - Hoja de trucos de Pandas #1
#### Pandas data structures



```
import pandas as pd
import numpy as np
#Series — One dimensional labeled array
s = pd.Series([3, -5, 7, 4], index=['a', 'b', 'c', 'd'])
print(s)
```

    a    3
    b   -5
    c    7
    d    4
    dtype: int64
    


```
#Data Frame — A two dimensional labeled data structure
data = {'Country': ['Belgium', 'India', 'Brazil'],
 'Capital': ['Brussels', 'New Delhi', 'Brasília'],
 'Population': [11190846, 1303171035, 207847528]}

df = pd.DataFrame(data,
 columns=['Country', 'Capital', 'Population'])
print(df)
```

       Country    Capital  Population
    0  Belgium   Brussels    11190846
    1    India  New Delhi  1303171035
    2   Brazil   Brasília   207847528
    


```
path='datos_rrss.xlsx'
```

#### Leer y escribir en CSV



```
#Read CSV file
#pd.read_csv('file.csv', header=None, nrows=5)
#Write to CSV file
#df.to_csv('myDataFrame.csv')
pd.read_csv('datos_rrss_t1.csv')
```


    ---------------------------------------------------------------------------

    FileNotFoundError                         Traceback (most recent call last)

    <ipython-input-10-0cc8e2b74784> in <module>
          3 #Write to CSV file
          4 #df.to_csv('myDataFrame.csv')
    ----> 5 pd.read_csv('datos_rrss_t1.csv')
    

    ~\anaconda3\lib\site-packages\pandas\io\parsers.py in read_csv(filepath_or_buffer, sep, delimiter, header, names, index_col, usecols, squeeze, prefix, mangle_dupe_cols, dtype, engine, converters, true_values, false_values, skipinitialspace, skiprows, skipfooter, nrows, na_values, keep_default_na, na_filter, verbose, skip_blank_lines, parse_dates, infer_datetime_format, keep_date_col, date_parser, dayfirst, cache_dates, iterator, chunksize, compression, thousands, decimal, lineterminator, quotechar, quoting, doublequote, escapechar, comment, encoding, dialect, error_bad_lines, warn_bad_lines, delim_whitespace, low_memory, memory_map, float_precision, storage_options)
        608     kwds.update(kwds_defaults)
        609 
    --> 610     return _read(filepath_or_buffer, kwds)
        611 
        612 
    

    ~\anaconda3\lib\site-packages\pandas\io\parsers.py in _read(filepath_or_buffer, kwds)
        460 
        461     # Create the parser.
    --> 462     parser = TextFileReader(filepath_or_buffer, **kwds)
        463 
        464     if chunksize or iterator:
    

    ~\anaconda3\lib\site-packages\pandas\io\parsers.py in __init__(self, f, engine, **kwds)
        817             self.options["has_index_names"] = kwds["has_index_names"]
        818 
    --> 819         self._engine = self._make_engine(self.engine)
        820 
        821     def close(self):
    

    ~\anaconda3\lib\site-packages\pandas\io\parsers.py in _make_engine(self, engine)
       1048             )
       1049         # error: Too many arguments for "ParserBase"
    -> 1050         return mapping[engine](self.f, **self.options)  # type: ignore[call-arg]
       1051 
       1052     def _failover_to_python(self):
    

    ~\anaconda3\lib\site-packages\pandas\io\parsers.py in __init__(self, src, **kwds)
       1865 
       1866         # open handles
    -> 1867         self._open_handles(src, kwds)
       1868         assert self.handles is not None
       1869         for key in ("storage_options", "encoding", "memory_map", "compression"):
    

    ~\anaconda3\lib\site-packages\pandas\io\parsers.py in _open_handles(self, src, kwds)
       1360         Let the readers open IOHanldes after they are done with their potential raises.
       1361         """
    -> 1362         self.handles = get_handle(
       1363             src,
       1364             "r",
    

    ~\anaconda3\lib\site-packages\pandas\io\common.py in get_handle(path_or_buf, mode, encoding, compression, memory_map, is_text, errors, storage_options)
        640                 errors = "replace"
        641             # Encoding
    --> 642             handle = open(
        643                 handle,
        644                 ioargs.mode,
    

    FileNotFoundError: [Errno 2] No such file or directory: 'datos_rrss_t1.csv'


#### Leer y escribir en Excel


```
#Read Excel file
#pd.read_excel('file.xlsx')
#Write to Excel file
#pd.to_excel('dir/myDataFrame.xlsx', sheet_name='Sheet1')
### Read multiple sheets from the same file
 #xlsx = pd.ExcelFile('file.xls')
#df = pd.read_excel(xlsx, 'Sheet1')
pd.read_excel('datos_rrss.xlsx')
```


    ---------------------------------------------------------------------------

    FileNotFoundError                         Traceback (most recent call last)

    <ipython-input-11-ec4d28033183> in <module>
          6  #xlsx = pd.ExcelFile('file.xls')
          7 #df = pd.read_excel(xlsx, 'Sheet1')
    ----> 8 pd.read_excel('datos_rrss.xlsx')
    

    ~\anaconda3\lib\site-packages\pandas\util\_decorators.py in wrapper(*args, **kwargs)
        297                 )
        298                 warnings.warn(msg, FutureWarning, stacklevel=stacklevel)
    --> 299             return func(*args, **kwargs)
        300 
        301         return wrapper
    

    ~\anaconda3\lib\site-packages\pandas\io\excel\_base.py in read_excel(io, sheet_name, header, names, index_col, usecols, squeeze, dtype, engine, converters, true_values, false_values, skiprows, nrows, na_values, keep_default_na, na_filter, verbose, parse_dates, date_parser, thousands, comment, skipfooter, convert_float, mangle_dupe_cols, storage_options)
        334     if not isinstance(io, ExcelFile):
        335         should_close = True
    --> 336         io = ExcelFile(io, storage_options=storage_options, engine=engine)
        337     elif engine and engine != io.engine:
        338         raise ValueError(
    

    ~\anaconda3\lib\site-packages\pandas\io\excel\_base.py in __init__(self, path_or_buffer, engine, storage_options)
       1069                 ext = "xls"
       1070             else:
    -> 1071                 ext = inspect_excel_format(
       1072                     content=path_or_buffer, storage_options=storage_options
       1073                 )
    

    ~\anaconda3\lib\site-packages\pandas\io\excel\_base.py in inspect_excel_format(path, content, storage_options)
        947     assert content_or_path is not None
        948 
    --> 949     with get_handle(
        950         content_or_path, "rb", storage_options=storage_options, is_text=False
        951     ) as handle:
    

    ~\anaconda3\lib\site-packages\pandas\io\common.py in get_handle(path_or_buf, mode, encoding, compression, memory_map, is_text, errors, storage_options)
        649         else:
        650             # Binary mode
    --> 651             handle = open(handle, ioargs.mode)
        652         handles.append(handle)
        653 
    

    FileNotFoundError: [Errno 2] No such file or directory: 'datos_rrss.xlsx'


#### Pidiendo ayuda


```
help(pd.Series.loc)
```

    Help on property:
    
        Access a group of rows and columns by label(s) or a boolean array.
        
        ``.loc[]`` is primarily label based, but may also be used with a
        boolean array.
        
        Allowed inputs are:
        
        - A single label, e.g. ``5`` or ``'a'``, (note that ``5`` is
          interpreted as a *label* of the index, and **never** as an
          integer position along the index).
        - A list or array of labels, e.g. ``['a', 'b', 'c']``.
        - A slice object with labels, e.g. ``'a':'f'``.
        
          .. warning:: Note that contrary to usual python slices, **both** the
              start and the stop are included
        
        - A boolean array of the same length as the axis being sliced,
          e.g. ``[True, False, True]``.
        - An alignable boolean Series. The index of the key will be aligned before
          masking.
        - An alignable Index. The Index of the returned selection will be the input.
        - A ``callable`` function with one argument (the calling Series or
          DataFrame) and that returns valid output for indexing (one of the above)
        
        See more at :ref:`Selection by Label <indexing.label>`.
        
        Raises
        ------
        KeyError
            If any items are not found.
        IndexingError
            If an indexed key is passed and its index is unalignable to the frame index.
        
        See Also
        --------
        DataFrame.at : Access a single value for a row/column label pair.
        DataFrame.iloc : Access group of rows and columns by integer position(s).
        DataFrame.xs : Returns a cross-section (row(s) or column(s)) from the
            Series/DataFrame.
        Series.loc : Access group of values using labels.
        
        Examples
        --------
        **Getting values**
        
        >>> df = pd.DataFrame([[1, 2], [4, 5], [7, 8]],
        ...      index=['cobra', 'viper', 'sidewinder'],
        ...      columns=['max_speed', 'shield'])
        >>> df
                    max_speed  shield
        cobra               1       2
        viper               4       5
        sidewinder          7       8
        
        Single label. Note this returns the row as a Series.
        
        >>> df.loc['viper']
        max_speed    4
        shield       5
        Name: viper, dtype: int64
        
        List of labels. Note using ``[[]]`` returns a DataFrame.
        
        >>> df.loc[['viper', 'sidewinder']]
                    max_speed  shield
        viper               4       5
        sidewinder          7       8
        
        Single label for row and column
        
        >>> df.loc['cobra', 'shield']
        2
        
        Slice with labels for row and single label for column. As mentioned
        above, note that both the start and stop of the slice are included.
        
        >>> df.loc['cobra':'viper', 'max_speed']
        cobra    1
        viper    4
        Name: max_speed, dtype: int64
        
        Boolean list with the same length as the row axis
        
        >>> df.loc[[False, False, True]]
                    max_speed  shield
        sidewinder          7       8
        
        Alignable boolean Series:
        
        >>> df.loc[pd.Series([False, True, False],
        ...        index=['viper', 'sidewinder', 'cobra'])]
                    max_speed  shield
        sidewinder          7       8
        
        Index (same behavior as ``df.reindex``)
        
        >>> df.loc[pd.Index(["cobra", "viper"], name="foo")]
               max_speed  shield
        foo
        cobra          1       2
        viper          4       5
        
        Conditional that returns a boolean Series
        
        >>> df.loc[df['shield'] > 6]
                    max_speed  shield
        sidewinder          7       8
        
        Conditional that returns a boolean Series with column labels specified
        
        >>> df.loc[df['shield'] > 6, ['max_speed']]
                    max_speed
        sidewinder          7
        
        Callable that returns a boolean Series
        
        >>> df.loc[lambda df: df['shield'] == 8]
                    max_speed  shield
        sidewinder          7       8
        
        **Setting values**
        
        Set value for all items matching the list of labels
        
        >>> df.loc[['viper', 'sidewinder'], ['shield']] = 50
        >>> df
                    max_speed  shield
        cobra               1       2
        viper               4      50
        sidewinder          7      50
        
        Set value for an entire row
        
        >>> df.loc['cobra'] = 10
        >>> df
                    max_speed  shield
        cobra              10      10
        viper               4      50
        sidewinder          7      50
        
        Set value for an entire column
        
        >>> df.loc[:, 'max_speed'] = 30
        >>> df
                    max_speed  shield
        cobra              30      10
        viper              30      50
        sidewinder         30      50
        
        Set value for rows matching callable condition
        
        >>> df.loc[df['shield'] > 35] = 0
        >>> df
                    max_speed  shield
        cobra              30      10
        viper               0       0
        sidewinder          0       0
        
        **Getting values on a DataFrame with an index that has integer labels**
        
        Another example using integers for the index
        
        >>> df = pd.DataFrame([[1, 2], [4, 5], [7, 8]],
        ...      index=[7, 8, 9], columns=['max_speed', 'shield'])
        >>> df
           max_speed  shield
        7          1       2
        8          4       5
        9          7       8
        
        Slice with integer labels for rows. As mentioned above, note that both
        the start and stop of the slice are included.
        
        >>> df.loc[7:9]
           max_speed  shield
        7          1       2
        8          4       5
        9          7       8
        
        **Getting values with a MultiIndex**
        
        A number of examples using a DataFrame with a MultiIndex
        
        >>> tuples = [
        ...    ('cobra', 'mark i'), ('cobra', 'mark ii'),
        ...    ('sidewinder', 'mark i'), ('sidewinder', 'mark ii'),
        ...    ('viper', 'mark ii'), ('viper', 'mark iii')
        ... ]
        >>> index = pd.MultiIndex.from_tuples(tuples)
        >>> values = [[12, 2], [0, 4], [10, 20],
        ...         [1, 4], [7, 1], [16, 36]]
        >>> df = pd.DataFrame(values, columns=['max_speed', 'shield'], index=index)
        >>> df
                             max_speed  shield
        cobra      mark i           12       2
                   mark ii           0       4
        sidewinder mark i           10      20
                   mark ii           1       4
        viper      mark ii           7       1
                   mark iii         16      36
        
        Single label. Note this returns a DataFrame with a single index.
        
        >>> df.loc['cobra']
                 max_speed  shield
        mark i          12       2
        mark ii          0       4
        
        Single index tuple. Note this returns a Series.
        
        >>> df.loc[('cobra', 'mark ii')]
        max_speed    0
        shield       4
        Name: (cobra, mark ii), dtype: int64
        
        Single label for row and column. Similar to passing in a tuple, this
        returns a Series.
        
        >>> df.loc['cobra', 'mark i']
        max_speed    12
        shield        2
        Name: (cobra, mark i), dtype: int64
        
        Single tuple. Note using ``[[]]`` returns a DataFrame.
        
        >>> df.loc[[('cobra', 'mark ii')]]
                       max_speed  shield
        cobra mark ii          0       4
        
        Single tuple for the index with a single label for the column
        
        >>> df.loc[('cobra', 'mark i'), 'shield']
        2
        
        Slice from index tuple to single label
        
        >>> df.loc[('cobra', 'mark i'):'viper']
                             max_speed  shield
        cobra      mark i           12       2
                   mark ii           0       4
        sidewinder mark i           10      20
                   mark ii           1       4
        viper      mark ii           7       1
                   mark iii         16      36
        
        Slice from index tuple to index tuple
        
        >>> df.loc[('cobra', 'mark i'):('viper', 'mark ii')]
                            max_speed  shield
        cobra      mark i          12       2
                   mark ii          0       4
        sidewinder mark i          10      20
                   mark ii          1       4
        viper      mark ii          7       1
    
    

#### Selección
#### Obtener


```

#Get one element
s['b']

```




    -5




```
#Get subset of a DataFrame
df[1:]
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Country</th>
      <th>Capital</th>
      <th>Population</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>India</td>
      <td>New Delhi</td>
      <td>1303171035</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Brazil</td>
      <td>Brasília</td>
      <td>207847528</td>
    </tr>
  </tbody>
</table>
</div>



#### Selección, indexación booleana y configuración


#### Por posición


```
#Select single value by row & Column
df.iloc[0,0]
df.iat([0],[0])
```


    ---------------------------------------------------------------------------

    TypeError                                 Traceback (most recent call last)

    <ipython-input-15-cbbdae3b5dfe> in <module>
          1 #Select single value by row & Column
          2 df.iloc[0,0]
    ----> 3 df.iat([0],[0])
    

    TypeError: '_iAtIndexer' object is not callable


#### Por etiqueta



```
#Select single value by row and column labels
df.loc[0,'Country']
df.at([0], ['Country']) 
```


    ---------------------------------------------------------------------------

    TypeError                                 Traceback (most recent call last)

    <ipython-input-16-4057bc237b0d> in <module>
          1 #Select single value by row and column labels
          2 df.loc[0,'Country']
    ----> 3 df.at([0], ['Country'])
    

    TypeError: '_AtIndexer' object is not callable


#### Por etiqueta / posición


```
#Select single row of subset rows
df.ix[2]
```


    ---------------------------------------------------------------------------

    AttributeError                            Traceback (most recent call last)

    <ipython-input-17-a52a43abc23f> in <module>
          1 #Select single row of subset rows
    ----> 2 df.ix[2]
    

    ~\anaconda3\lib\site-packages\pandas\core\generic.py in __getattr__(self, name)
       5463             if self._info_axis._can_hold_identifiers_and_holds_name(name):
       5464                 return self[name]
    -> 5465             return object.__getattribute__(self, name)
       5466 
       5467     def __setattr__(self, name: str, value) -> None:
    

    AttributeError: 'DataFrame' object has no attribute 'ix'



```
#Select a single column of subset of columns
df.ix[:,'Capital']
```


```
#Select rows and columns
df.ix[1,'Capital']
```


    ---------------------------------------------------------------------------

    AttributeError                            Traceback (most recent call last)

    <ipython-input-18-639a70ca7082> in <module>
          1 #Select rows and columns
    ----> 2 df.ix[1,'Capital']
    

    ~\anaconda3\lib\site-packages\pandas\core\generic.py in __getattr__(self, name)
       5463             if self._info_axis._can_hold_identifiers_and_holds_name(name):
       5464                 return self[name]
    -> 5465             return object.__getattribute__(self, name)
       5466 
       5467     def __setattr__(self, name: str, value) -> None:
    

    AttributeError: 'DataFrame' object has no attribute 'ix'


#### Indexación booleana


```
#Series s where value is not >1
s[~(s > 1)] 
```




    b   -5
    dtype: int64




```
#s where value is <-1 or >2
s[(s < -1) | (s > 2)] 
```




    a    3
    b   -5
    c    7
    d    4
    dtype: int64




```
#Use filter to adjust DataFrame
 df[df['Population']>1200000000]

```


      File "<ipython-input-21-12e7ff37468c>", line 2
        df[df['Population']>1200000000]
        ^
    IndentationError: unexpected indent
    


#### Configuración


```
# Set index a of Series s to 6
s['a'] = 6 
print(s)
```

    a    6
    b   -5
    c    7
    d    4
    dtype: int64
    

#### Leer y escribir en una tabla de base de datos o consulta SQL


```
# Read SqL Query
from sqlalchemy import create_engine
engine = create_engine('sqlite:///:memory:')
pd.read_sql("SELECT * FROM my_table;", engine)
pd.read_sql_table('my_table', engine)
pd.read_sql_query("SELECT * FROM my_table;", engine)
#Write to Sql Query
 pd.to_sql('myDF', engine)
```


      File "<ipython-input-23-307920bb94ef>", line 8
        pd.to_sql('myDF', engine)
        ^
    IndentationError: unexpected indent
    


#### Goteante


```
#values 
print(s)
```


```
#Drop values from rows (axis=0)
s.drop(['a', 'c'])
```


```
#Drop values from columns(axis=1)
df.drop('Country', axis=1)
```

#### Ordenar y clasificar


```
#Values df
print(df)
```


```
#Sort by labels along an axis
df.sort_index()
```


```
#Sort by the values along an axis
df.sort_values(by='Country')
```


```
#Assign ranks to entries
df.rank() 
```

#### Recuperación de información de series / marcos de datos

#### Información básica



```
#Values df
print(df)
```


```
#(rows,columns) 
df.shape
```


```
#Describe index
df.index
```


```
#Describe DataFrame columns
df.columns
```


```
#Info on DataFrame
df.info()
```


```
#Number of non-NA values
df.count()
```

#### Resumen


```
#Sum of values
df.sum()
```


```
#Cummulative sum of values
df.cumsum()
```


```
#Minimum
df.min()
#max values
df.max()
```


```
#Minimum/Maximum index value
df.idxmin()
df.idxmax()
```


```
#Summary statistics
df.describe()
```


```
#Mean of values
df.mean()
```


```
#Median of values
df.median()
```

#### Aplicar funciones


```
#Apply function
df.apply(lambda x: x*2)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Country</th>
      <th>Capital</th>
      <th>Population</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>BelgiumBelgium</td>
      <td>BrusselsBrussels</td>
      <td>22381692</td>
    </tr>
    <tr>
      <th>1</th>
      <td>IndiaIndia</td>
      <td>New DelhiNew Delhi</td>
      <td>2606342070</td>
    </tr>
    <tr>
      <th>2</th>
      <td>BrazilBrazil</td>
      <td>BrasíliaBrasília</td>
      <td>415695056</td>
    </tr>
  </tbody>
</table>
</div>



#### Alineación de datos


```
s3 = pd.Series([7, -2, 3], index=['a', 'c', 'd'])
s + s3
```


```
s.sub(s3, fill_value=2)
```


```
 s.div(s3, fill_value=4)

```


```
s.mul(s3, fill_value=3)
```

### - Hoja de trucos de Pandas #2


#### Cambiar la forma de los datos

#### Pivote


```
import pandas as pd

###Declaracion de df
df = pd.DataFrame({'species': ['bear', 'bear', 'marsupial'],
                  'population': [1864, 22000, 80000]},
                  index=['panda', 'polar', 'koala'])

#Data Frame — A two dimensional labeled data structure
data = {'Date': ['2016-03-01','2016-03-02','2016-03-01','2016-03-03','2016-03-02','2016-03-03'],
        'Type': ['a','b','c','a','a','c'],
        'Value': [11.432,13.031,20.784,99.906,1.303,20.784]}

df2 = pd.DataFrame(data,
 columns=['Date', 'Type', 'Value'])

print(df2)
#Spread rows into columns 

```


```
#Spread rows into columns
df3= df2.pivot(index='Date',
              columns='Type',
              values='Value')
print(df3)
```

#### Tabla dinámica


```
df4= pd.pivot_table(df2,
                   values='Value',
                   index='Date',
                   columns='Type')
print(df4)
```


```
# Pivot a level of column labels

```


```
#Gather columns into rows
pd.melt(df2,
        id_vars=["Date"],
        value_vars=["Type","Value"],
        value_name="Observations")


```


```
##Column-index,series pairs
df.iteritems()
```


```
#Row-index,series pairs
df.iterrows()
```

#### Indexación avanzada


#### Seleccionar


```
#Select cols with any vals >1
df3.loc[:,(df3>1).any()]
```


```
#Select cols with vals>1
df3.loc[:,(df3>1).all()]
```


```
#Select cols with NaN
df3.loc[:,df3.isnull().any()]
```


```
#Select cols without NAN
df3.loc[:,df3.notnull().all()]
```

#### Indexación con isin


```
#Find same elements
df[(df.Country.isin(df2.Type))]
```


```
#Filter on values
df3.filter(items=“a”,“b”])
```


```
#Select specific elements
df.select(lambda x: not x%5)
```

#### Dónde


```
#Subset the data
s.where(s > 0)
```

#### Consulta


```
#Query DataFrame
df6.query('second > first')
```

#### Configuración / restablecimiento del índice


```
# Set the index
df.set_index('Country')
```


```
#Reset the index
df4 = df.reset_index()
print(df4)
```


```
#Renamme DataFrame
df = df.rename(index=str,
               columns={"Country":"cntry",
                        "Capital":"cptl",
                        "Population":"ppltn"})
print(df)
```

#### Reindexar


```
s2 = s.reindex(['a','c','d','e','b'])
print(s2)
```


```
#Forward Filling
df.reindex(range(4),
           method='Ffill')
```


```
#Backward Filling
s3 = s.reindex(range(5),
               method='bfill')
```

#### Indexación múltiple


```
arrays = [np.array([1,2,3]),
          np.array([5,4,3])]
df5= pd.DataFrame(np.random.rand(3, 2), index=arrays)
tuples = list(zip(*arrays))
index = pd.MultiIndex.from_tuples(tuples,
                                  names=['first', 'second'])
df6= pd.DataFrame(np.random.rand(3, 2), index=index)
df2.set_index(["Date", "Type"])
```

#### Datos duplicados


```
#Return unique values
s3.unique()
print(s3)
```


```
#Check duplicates 
df2.duplicated('Type')
```


```
#Drop duplicates
df2.drop_duplicates('Type', keep='last')
```


```
#Check index duplicates
df.index.duplicated()
```

#### Agrupar datos

#### Agregación 


```
df2.groupby(by=['Date','Type']).mean()
```


```
df4.groupby(level=0).sum()
```


```
df4.groupby(level=0).agg({'a':lambda x:sum(x)/len(x),
                          'b': np.sum})
```

#### Transformación


```
customSum = lambda x: (x+x%2)
df4.groupby(level=0).transform(customSum)
```

#### Datos perdidos


```
#Drop NaN values
df.dropna()
```


```
#Fill NaN values with a predetermined value
df3.fillna(df3.mean())
```


```
#Replace values with others
df2.replace("a","f")
```

#### Combinando datos


```

data1 = pd.DataFrame({'X1': ['a','b','c'], 'X2': [11.432,1.303, 99.906]}); data1
data2 = pd.DataFrame({'X1': ['a','b','d'], 'X3': [20.78,"NaN", 20.784]}); data2
print(data1)
print(data2)
```

#### Unir


```
pd.merge(data1,
         data2,
        how='left',
        on='X1')
```


```
pd.merge(data1,
         data2,
        how='right',
        on='X1')
```


```
pd.merge(data1,
         data2,
        how='inner',
        on='X1')
```


```
pd.merge(data1,
         data2,
        how='outer',
        on='X1')
```

#### Entrar


```
data1.join(data2, how='right')
```

#### Concatenar


```
#Vertical
s.append(s2)

```


```
#Horizontal/vertical
pd.concat([s,s2],axis=1, keys=['One','Two'])

```


```
pd.concat([data1, data2], axis=1, join='inner')
```

#### fechas


```
df2['Date']= pd.to_datetime(df2['Date'])
df2['Date']= pd.date_range('2000-1-1',
                            periods=6,
                            freq='M')
dates = [datetime(2012,5,1), datetime(2012,5,2)]
index = pd.DatetimeIndex(dates)
index = pd.date_range(datetime(2012,2,1), end, freq='BM')
```

#### Visualización


```
import matplotlib.pyplot as plt
s.plot()
plt.show()
print(s)
```

    Matplotlib is building the font cache; this may take a moment.
    


    
![png](output_130_1.png)
    


    a    3
    b   -5
    c    7
    d    4
    dtype: int64
    


```
df2.plot()
plt.show()
```


    ---------------------------------------------------------------------------

    NameError                                 Traceback (most recent call last)

    <ipython-input-6-ba5ab7381d57> in <module>
    ----> 1 df2.plot()
          2 plt.show()
    

    NameError: name 'df2' is not defined



```

```


```

```

#optimize types
def optimize_types(dataframe):
    np_types = [np.int8 ,np.int16 ,np.int32, np.int64,
                np.uint8 ,np.uint16, np.uint32, np.uint64]
    np_types = [np_type.__name__ for np_type in np_types]
    type_df = pd.DataFrame(data=np_types, columns=['class_type'])
    type_df['min_value'] = type_df['class_type'].apply(lambda row: np.iinfo(row).min)
    type_df['max_value'] = type_df['class_type'].apply(lambda row: np.iinfo(row).max)
    type_df['range'] = type_df['max_value'] - type_df['min_value']
    type_df.sort_values(by='range', inplace=True)
    for col in dataframe.loc[:, dataframe.dtypes <= np.integer]:
        col_min = dataframe[col].min()
        col_max = dataframe[col].max()
        temp = type_df[(type_df['min_value'] <= col_min) & (type_df['max_value'] >= col_max)]
        optimized_class = temp.loc[temp['range'].idxmin(), 'class_type']
        print("Col name : {} Col min_value : {} Col max_value : {} Optimized Class : {}".format(col, col_min, col_max, optimized_class))
        dataframe[col] = dataframe[col].astype(optimized_class)
    return dataframe

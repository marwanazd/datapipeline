from etl import *
from sklearn.pipeline import Pipeline


def extract():
    extractor = ExtractFromLocal("precipitation.csv")
    data1 = extractor.extract()
    extractor = ExtractFromLocal("agricultural_production.csv")
    data2 = extractor.extract()
    return data1, data2


def transform(df1, df2):
    # Specify the columns you want to apply the transformation to

    # Data pipeline1 for df1

    target_columns1 = ['Year', 'Morocco']

    selcet_subset1 = SubsetSelectorTransformer(
        selected_columns=target_columns1
    )

    rename_col1 = ColumnRenamingTransformer(
        column_name_mapping={
            'Morocco': 'Précipitation'
        }

    )

    steps1 = [
        ('step1', selcet_subset1),
        ('step2', rename_col1),
    ]

    pl1 = Pipeline(steps=steps1)

    df1_transformed = pl1.fit_transform(df1)

    # Data pipeline1 for df2 and df1

    target_columns2 = ['Item', 'Year', 'Value']
    selcet_subset2 = SubsetSelectorTransformer(
        selected_columns=target_columns2
    )

    reshape = DataPivotingTransformer(
        column_name='Item'

    )

    drop_duplicates = DropDuplicatesTransformer(
        columns=True,
        rows=False
    )

    rename_col2 = ColumnRenamingTransformer(
        column_name_mapping={
            'Year_Apples': 'Year',
            'Item_Cereals n.e.c.': 'Item_Cereals',
            'Value_Cereals n.e.c.': 'Value_Cereals'
        }

    )

    add_precipitation = DataBlendingTransformer(
        dfs=df1_transformed,
        method='Merge',
        on='Year',
        axis=0,
        how='inner',
        ignore_index=True

    )

    steps2 = [
        ('step1', selcet_subset2),
        ('step2', reshape),
        ('step3', drop_duplicates),
        ('step4', rename_col2),
        ('step5', add_precipitation)
    ]

    pl2 = Pipeline(steps=steps2)

    transformed_data = pl2.fit_transform(df2)
    return transformed_data

def load(data):
    loader = LoadToPostgresSql(
        username='postgres',
        password='postgresql#DataPipeline',
        host='localhost',
        port='5432',
        database='mydatabase'
    )
    loader.connect()
    loader.load_data(
        df=data,
        table_name='results'
    )


def run_pipeline():
    # extract
    df1, df2 = extract()

    # transform
    df = transform(df1=df1, df2=df2)

    # load
    load(data=df)

    return print('Pipeline executed successfully')


if __name__ == '__main__':
    # executer le code de pipeline
    run_pipeline()
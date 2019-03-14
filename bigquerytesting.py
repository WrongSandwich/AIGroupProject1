from google.cloud import bigquery
import pandas as pd
import numpy as np

#BigQuery from Google
client = bigquery.Client()
ng_dataset_ref = client.dataset('noaa_gsod', project = 'bigquery-public-data')
ng_dset = client.get_dataset(ng_dataset_ref)

#Lists all of the tables (i.e. from 1929 to current)
#[print(x.table_id) for x in client.list_tables(ng_dset)]

#Getting a table reference to the year 1929
table_ref = ng_dataset_ref.table("gsod1929")
table = client.get_table(table_ref)

df = client.list_rows(table).to_dataframe()

df.info() #Prints all rows in 1 table


#ML attempts
#dataset = bigquery.Dataset(ng_dataset_ref)
#client.create_dataset(dataset)

#%load_ext google.cloud.bigquery

#%%bigquery
train_query = """
    CREATE OR REPLACE MODEL 
    'weather_model.linear'
    OPTIONS(model_type='linear_reg') AS
    SELECT
      temp,
      dewp,
      slp,
      stp,
      wdsp,
      year,
    FROM
      'table'
    """

training_job = client.query(train_query)
print(training_job)

training_info = """
  SELECT
    *
  FROM
    ML.TRAINING_INFO(MODEL 'weather_model.linear')
  """

training_info_job = client.query(training_info)
print(training_info_job)

weights = """
SELECT
  category,
  weight
FROM
  UNNEST((
    SELECT
      category_weights
    FROM
      ML.WEIGHTS(MODEL 'weather.linear')
    WHERE
      processed_input = 'input_col'))
"""

weights_info = client.query(weights)
print(weights_info)



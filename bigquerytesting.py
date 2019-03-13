from google.cloud import bigquery
import pandas as pd

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

df.info()
import os
import time
from functools import wraps
import requests
import json
from time import sleep
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, explode, to_json, struct, lit, current_timestamp, date_format
from pyspark.sql.types import StructType, StructField, StringType, TimestampType
from gcp_utils.gcp_util import *
from connection_helpers.bq_helper import *
from concurrent.futures import ThreadPoolExecutor, as_completed

# Configuration remains unchanged
config = {
    'DEV': {
        'url': 'https://esap-share-nonprod-apim-01-west-az.albertsons.com/abs/acceptancepub/dirm/menuservice/v2/recipe-discovery/by-bpn',
        'headers': {
            'Authorization': '',
            'Accept': 'application/json',
            'ocp-apim-subscription-key': 'cbcdfd64139a4800bd0a1562ce4eb4b0',
            'Content-Type': 'application/json'
        },
        'params': {
            'bpn': 'bpn_no',
        }
    },
    'qa': {
        'url': 'https://esap-share-nonprod-apim-01-west-az.albertsons.com/abs/perfpub/dirm/menuservice/v2/recipe-discovery/by-bpn',
        'headers': {
            'Authorization': '',
            'Accept': 'application/json',
            'ocp-apim-subscription-key': '3d6f2f41907c495aa82c1278a7842020',
            'Content-Type': 'application/json'
        },
        'params': {
            'bpn': 'bpn_no',
        }
    },
    'prod': {
        'url': 'https://esap-apim-prod-01.albertsons.com/abs/pub/dirm/menuservice/v2/recipe-discovery/by-bpn',
        'headers': {
            'Authorization': 'Basic e3tiYXNpY0F1dGhVc2VybmFtZX19Ont7YmFzaWNBdXRoUGFzc3dvcmR9fQ==',
            'Accept': 'application/json',
            'ocp-apim-subscription-key': 'ed0dadf2d6fe47a08c1e88e197f971a7',
            'Content-Type': 'application/json'
        },
        'params': {
            'bpn': 'bpn_no',
        }
    }
}

def rate_limited(max_per_second):
    """
    Decorator function to limit the rate of function calls to a maximum
    number per second.

    Args:
        max_per_second (int): Maximum number of function calls allowed per second.

    Returns:
        function: The decorated function with rate limiting applied.
    """
    def decorator(func):
        last_called = [0.0]
        period = 1.0 / max_per_second

        @wraps(func)
        def rate_limited_func(*args, **kwargs):
            elapsed = time.time() - last_called[0]
            wait = period - elapsed
            if wait > 0:
                time.sleep(wait)
            result = func(*args, **kwargs)
            last_called[0] = time.time()
            return result

        return rate_limited_func

    return decorator

@rate_limited(max_per_second=80)
def read_api_data(bpn_no, env):
    """
    Fetch data from the API for a given BPN number and environment configuration.

    Args:
        bpn_no (str): The BPN number for which to fetch data.
        env (str): The environment configuration ('dev', 'qa', 'prod').

    Returns:
        dict: The API response data in JSON format, or None if an error occurred.
    """
    max_retries = 3
    for attempt in range(max_retries):
        try:
            env_config = config[env]
            url = env_config['url']
            headers = env_config['headers']
            params = env_config['params']
            params['bpn'] = bpn_no

            response = requests.get(url, headers=headers, params=params, timeout=6)
            #print(response.url)
            if response.status_code == 200:
                return response.json()
            else:
                print(f'Error: {response.status_code}, {response.text}')
        except requests.exceptions.Timeout:
            print(f"Attempt {attempt + 1} timed out. Retrying...")
            sleep(2)
        except requests.exceptions.RequestException as e:
            print(f"An error occurred: {e}")
            break
    return None

def read_input_table(table_name, query_tag_bq):
    """
    Read data from a BigQuery table using the provided table name.

    Args:
        table_name (str): The name of the table to read from.

    Returns:
        DataFrame: A PySpark DataFrame containing the data from the BigQuery table.
    """
    return read_from_bq(table_name, materialization_dataset=materialization_dataset, query_tag=query_tag_bq)

def write_to_delta(df, table_name):
    """
    Write a PySpark DataFrame to a Delta table.

    Args:
        df (DataFrame): The PySpark DataFrame to be written.
        table_name (str): The name of the Delta table where the data will be written.
    """
    df.write.format('delta').option("mergeSchema", "true").mode('overwrite').saveAsTable(f'db_work.{table_name}')
    print(f"Data written to Delta table: {table_name}")

def process_hhid_batch(spark, hhid_batch, model_id, env):
    """
    Process a batch of household IDs and their associated boost BPNs, fetching
    recommendations from the API and storing them in a list of dictionaries.

    Args:
        spark: SparkSession to convert API data to DF
        hhid_batch (list): A list of tuples, each containing a household ID and boost BPN.
        model_id (str): The model ID associated with the recommendations.
        env (str): The environment configuration ('dev', 'qa', 'prod').

    Returns:
        list: A list of dictionaries containing household ID, model ID, recommendations,
        and modification timestamp.
    """
    results = []

    def fetch_and_process(hhid, boostbpn):
        api_response = read_api_data(boostbpn, env)
        if api_response:
            df_t = spark.sql("SELECT current_timestamp() AS timestamp")
            df_t1 = df_t.withColumn("timestamp1", date_format("timestamp", "MM/dd/yyyy HH:mm"))
            df_t2 = df_t1.collect()[0]['timestamp1']
            return {
                'HHID': hhid,
                'MODEL_ID': model_id,
                'RECOMMENDATION': json.dumps(api_response),
                'MODIFIED': df_t2
            }
        return None

    # Use ThreadPoolExecutor to handle concurrent API requests
    with ThreadPoolExecutor(max_workers=80) as executor:
        future_to_hhid = {executor.submit(fetch_and_process, hhid, boostbpn): (hhid, boostbpn) for hhid, boostbpn in hhid_batch}
        for future in as_completed(future_to_hhid):
            result = future.result()
            if result:
                results.append(result)

    return results

def recipes_extractor(spark, input_table, env, query_tag_bq):
    """
    Main function to process input data, fetch recommendations from the API,
    and write the results to Delta and BigQuery tables.

    Args:
        spark: SparkSession to convert API data to DF
        input_table (str): The name of the input table to process.
        env (str): The environment configuration ('dev', 'qa', 'prod').
        query_tag_bq (str): The query tag for BigQuery operations.
    """
    gcp_env, domain = ("NONPRD", "wmpz") if env == 'DEV' else ("PRD", "dppz")

    gcon = GCPConnection(env=gcp_env, domain=domain)
    _, credentials, parent_project, feature_project_id, bq_dataset, materialization_dataset, bqvw_pid, gcs_bucket = gcon.gcp_connection_setup()
    delta_table_name = "PRE_CALCULATED_RECIPE_RECOMMENDATIONS_DELTA"
    bq_table_name = f"{feature_project_id}.{bq_dataset}.PRE_CALCULATED_RECIPE_RECOMMENDATIONS"
    input_df = read_from_bq(input_table, materialization_dataset=materialization_dataset, query_tag=query_tag_bq)
    display(input_df)
    model_id = input_df.collect()[0]['model_id']

    batch_size = 1000
    hhid_batches = input_df.select("household_id", "boost_bpns").rdd.map(lambda x: (x[0], x[1])).collect()

    all_results = []
    for i in range(0, len(hhid_batches), batch_size):
        batch = hhid_batches[i:i + batch_size]
        results = process_hhid_batch(spark, batch, model_id, env)
        all_results.extend(results)
        print(f"Processed batch {i // batch_size + 1} with {len(batch)} records.")

    schema = StructType([
        StructField("HHID", StringType(), True),
        StructField("MODEL_ID", StringType(), True),
        StructField("RECOMMENDATION", StringType(), True),
        StructField("MODIFIED", StringType(), True)
    ])

    result_df = spark.createDataFrame(all_results, schema)
    temp_table_name = f"{bq_table_name}_tmp"
    write_to_bq(result_df, temp_table_name, "overwrite", gcs_bucket, query_tag=query_tag_bq)
    
    merge_query = f'''
    MERGE {bq_table_name} T
    USING {temp_table_name} S
    ON T.HHID = S.HHID AND T.MODEL_ID = S.MODEL_ID
    WHEN MATCHED THEN
      UPDATE SET
        RECOMMENDATION = S.RECOMMENDATION,
        MODIFIED = S.MODIFIED
    WHEN NOT MATCHED THEN
      INSERT (HHID, MODEL_ID, RECOMMENDATION, MODIFIED)
      VALUES (S.HHID, S.MODEL_ID, S.RECOMMENDATION, S.MODIFIED)'''
    
    execute_with_client(parent_project, merge_query, credentials=credentials, query_tag=query_tag_bq)

    # write_to_delta(result_df, delta_table_name)

    print("Recipe Runner process completed successfully.")
    return bq_table_name
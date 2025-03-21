# import the necessary packages
import json
import os
from gcp_utils.gcp_util import *
from connection_helpers.bq_helper import *

env = dbutils.widgets.get("ENV")
print(env)


# Define the class Bq_Connection with the following methods:    
class Bq_Connection:
    # Define the query_tag attribute with the following values:
    query_tag = {
        "project_version": "01",
        "app_code": "aamp",
        "project_name":"normalization_model",
        "env":"prod",
        "portfolio_name":"digital_personalization",
        "object_name":"normalization_model"
    }
    gcp_env,domain=("NONPRD","wmpz") if env=='DEV' else ("PRD","dppz")
    gcon = GCPConnection(env=gcp_env, domain=domain)
    scope, credentials, parent_project, feature_project_id, bq_dataset, materialization_dataset, bqvw_pid, gcs_bucket = gcon.gcp_connection_setup()
       
    # Define the execute_gcp_query method with the following parameters:
    @classmethod
    def execute_gcp_query(cls, query):
        execute_with_client(cls.parent_project, query, credentials=cls.credentials, query_tag=cls.query_tag)
    
    # Define the read_gcp_table method with the following parameters:    
    @classmethod
    def read_gcp_table(cls, query):
        return read_from_bq(query=query, materialization_dataset=cls.materialization_dataset, query_tag=cls.query_tag)
    
    # Define the write_gcp_table method with the following parameters:    
    @classmethod
    def write_gcp_table(cls, df, table_name):
        write_to_bq(df, table_name, "overwrite", cls.gcs_bucket, query_tag=cls.query_tag)

    # Define the write_gcp_append method with the following parameters:
    @classmethod
    def write_gcp_append(cls, df, table_name):
        write_to_bq(df, table_name, "append", cls.gcs_bucket, query_tag=cls.query_tag)

    

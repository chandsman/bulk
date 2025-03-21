from datetime import datetime
from pyspark.sql import SparkSession
from pyspark.sql.types import (StructType, StructField, StringType,TimestampType)

from bq_connection import Bq_Connection

class job_run_capture:

    spark = SparkSession.builder.appName("ReadCSV") \
        .config("spark.executor.memory", "8g") \
        .config("spark.driver.memory", "8g") \
        .config("spark.executor.memoryOverhead", "2g") \
        .config("spark.driver.memoryOverhead", "2g") \
        .config("spark.dynamicAllocation.enabled", "true") \
        .config("spark.dynamicAllocation.minExecutors", "2") \
        .config("spark.dynamicAllocation.maxExecutors", "10") \
        .config("spark.executor.extraJavaOptions", "-XX:+UseG1GC -XX:InitiatingHeapOccupancyPercent=35") \
        .config("spark.driver.extraJavaOptions", "-XX:+UseG1GC -XX:InitiatingHeapOccupancyPercent=35") \
        .getOrCreate()

    schema = StructType([
        StructField('job_id', StringType(), True), 
        StructField('run_id', StringType(), True), 
        StructField('job_start_time', TimestampType(), True), 
        StructField('job_end_time', TimestampType(), True), 
        StructField('duration', StringType(), True), 
        StructField('queue_duration', StringType(), True), 
        StructField('status', StringType(), True), 
        StructField('environment', StringType(), True), 
        StructField('InputContainer', StringType(), True), 
        StructField('InputFolder', StringType(), True), 
        StructField('InputFileName', StringType(), True)])

    run_details_bq_table =f"{Bq_Connection.feature_project_id}.{Bq_Connection.bq_dataset}.msg_bulk_job_run_details"

    print("Run details capture BQ table : " + run_details_bq_table)

    job_actual_start_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")
    # print("Job start time : " + str(job_start_time))
    
    start_time = None
    end_time = None
    duration = None
    status = "Not Started"

    @classmethod
    def job_running_status(cls,job_id, run_id, env, InputContainer, InputFolder, InputFileName, job_start_time):
        cls.start_time = datetime.now()
        cls.status = "Running"
        data = [(job_id, run_id, job_start_time, None, None, None, cls.status, env, InputContainer, InputFolder, InputFileName)]
        df_running = cls.spark.createDataFrame(data, schema=cls.schema)
        df_running.display()
        Bq_Connection.write_gcp_append(df_running, cls.run_details_bq_table)
        print(f"Job running status inserted successfully into {cls.run_details_bq_table}")

    @classmethod
    def job_success_status(cls,job_id, run_id, job_start_time):
        cls.end_time = datetime.now()  
        cls.duration = str(cls.end_time - cls.start_time)  
        cls.status = "Success"

        job_actual_start_time = datetime.strptime(cls.job_actual_start_time, "%Y-%m-%d %H:%M:%S.%f")
        queue_duration = str(job_actual_start_time - job_start_time)

        update_query = f"""
            UPDATE {cls.run_details_bq_table} 
            SET job_end_time = '{cls.end_time}', 
                duration = '{cls.duration}', 
                queue_duration = '{queue_duration}', 
                status = '{cls.status}' 
            WHERE job_id = '{job_id}' AND run_id = '{run_id}'
        """

        print(update_query)
        Bq_Connection.execute_gcp_query(update_query)
        print(f"Job success status inserted successfully into {cls.run_details_bq_table}")

    @classmethod
    def job_failed_status(cls,job_id, run_id, job_start_time):
        cls.end_time = datetime.now()  
        cls.duration = str(cls.end_time - cls.start_time)  
        cls.status = "Failed"

        job_actual_start_time = datetime.strptime(cls.job_actual_start_time, "%Y-%m-%d %H:%M:%S.%f")
        queue_duration = str(job_actual_start_time - job_start_time)

        update_query = f"""
            UPDATE {cls.run_details_bq_table} 
            SET job_end_time = '{cls.end_time}', 
                duration = '{cls.duration}', 
                queue_duration = '{queue_duration}', 
                status = '{cls.status}' 
            WHERE job_id = '{job_id}' AND run_id = '{run_id}'
        """
        print(update_query)
        Bq_Connection.execute_gcp_query(update_query)
        print(f"Job failed status inserted successfully into {cls.run_details_bq_table}")
# Databricks notebook source
from email_notifier import EmailNotification
from run_time_func import job_run_capture
from datetime import datetime

# COMMAND ----------

InputContainer = dbutils.widgets.get("InputContainer")
InputFolder = dbutils.widgets.get("InputFolder")
InputFile = dbutils.widgets.get("InputFile")
job_id = dbutils.widgets.get("job_id")
run_id = dbutils.widgets.get("job_run_id")
job_start_time = dbutils.widgets.get("job_start_time")
ENV = dbutils.widgets.get("ENV")

# COMMAND ----------

print("Input_Container:" + InputContainer)
print("InputFolder:" + InputFolder)
print("InputFile:" + InputFile)
print("job_id:" + job_id)
print("job_run_id:" + run_id)
print("ENV:" + ENV)
print("job_start_time:" + job_start_time)

job_start_time_dt = datetime.strptime(job_start_time, "%Y-%m-%dT%H:%M:%S.%f")


# COMMAND ----------

job_run_capture.job_running_status(job_id, run_id, ENV, InputContainer, InputFolder, InputFile, job_start_time_dt)

# COMMAND ----------

try:
    mailer = EmailNotification()
    mailer.send_confirmation_email(InputFile)
    dbutils.notebook.run(
        "./MSG_BULK_SERVICE",
        0,
        {
            "InputContainer": InputContainer,
            "InputFolder": InputFolder,
            "InputFile": InputFile,
            "job_id": job_id,
            "run_id": run_id,
            "ENV": ENV
        }
    )

    job_run_capture.job_success_status(job_id, run_id, job_start_time_dt)
    success_msg = (
        f"Job Details:\n"
        f"  - Job ID: {job_id}\n"
        f"  - Task Run ID: {run_id}\n"
        f"  - Started At: {job_run_capture.start_time}\n"
        f"  - Completed At: {job_run_capture.end_time}\n"
        f"  - Duration: {job_run_capture.duration}\n"
        f"  - Status: {job_run_capture.status}\n\n"

        f"Parameter Details:\n"
        f"  - Environment: {ENV}\n"
        f"  - InputContainer: {InputContainer}\n"
        f"  - InputFile: {InputFile}\n"
        f"  - InputFolder: {InputFolder}\n"
    )
    step = f"MSG_BULK_SERVICE - Job Success Notification for {ENV}:{InputFile}"
    
    mailer.send_success_email(step, message=success_msg)

    print('MSG_BULK_SERVICE notebook successful')

except Exception as e:
    
    job_run_capture.job_failed_status(job_id, run_id, job_start_time_dt)
    err_msg = (
        f"Job Details:\n"
        f"  - Job ID: {job_id}\n"
        f"  - Task Run ID: {run_id}\n"
        f"  - Started At: {job_run_capture.start_time}\n"
        f"  - Completed At: {job_run_capture.end_time}\n"
        f"  - Duration: {job_run_capture.duration}\n"
        f"  - Status: {job_run_capture.status}\n\n"

        f"Parameter Details:\n"
        f"  - Environment: {ENV}\n"
        f"  - InputContainer: {InputContainer}\n"
        f"  - InputFile: {InputFile}\n"
        f"  - InputFolder: {InputFolder}\n\n"

        f"Error Details:\n"
        f"  - Error: {e}\n"
    )
    step = f"MSG_BULK_SERVICE - Job Error Alert! for {ENV}:{InputFile}"
    mailer = EmailNotification()
    mailer.send_error_email(step, err_msg)
    print('MSG_BULK_SERVICE notebook failed')
    raise Exception("MSG_BULK_SERVICE notebook failed")

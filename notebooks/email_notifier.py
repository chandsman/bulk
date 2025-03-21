
"""
Module used to send email messages.
Utilised for sending email notifications to the users using SAFEWAY.
"""

from databricks.sdk.runtime import *
import logging
import smtplib
import warnings
from platform_helpers.utility_helpers import EmailNotifier
from typing import Union, List, Optional
from textwrap import dedent
from datetime import datetime

class EmailNotification:
    def build_mail_message(self, subject: str, recipients: list, body: str):
        return {
            "subject": subject,
            "recipients": recipients,
            "body": body,
            content_type: "text/html"
        }

    def send_email(self, mail_message: dict):
        print(f"Sending email to {mail_message['recipients']} with subject: {mail_message['subject']}")
        print(f"Body: {mail_message['body']}")

    def send_confirmation_email(self, filename: str):
        recipients = ['P13N.BackendEngineers@albertsons.com']
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        mail_body = f'''
        <html>
        <body>
            <h3 style="font-size: 20px;">Hi Team,</h3>
            <p style="font-size: 20px;"><b>Confirmation:</b> Input File Received</p>
            <p style="font-size: 20px;"><b>File Name:</b> {filename}</p>
            <p style="font-size: 20px;"><b>Status:</b> Job has been triggered successfully</p>
            <p style="font-size: 20px;">Timestamp: {current_time}</p>
            <p style="font-size: 20px;">Thank You!<br><b>MSG Team</b></p>
        </body>
        </html>
        '''

        subject = f'Confirmation: {filename} Received & Job Triggered'
        
        mailer = EmailNotifier()
        mail_message = mailer.build_mail_message(subject=subject, recipients=recipients, body=mail_body)
        mailer.send_email(mail_message)

    def send_success_notification(self, step: str, message: str):
        recipients = ['P13N.BackendEngineers@albertsons.com']
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        mail_body = f'''
        <html>
        <body>
            <h3 style="font-size: 20px;">Hi Team,</h3>
            <p style="font-size: 20px;"><b>Step:</b> {step}</p>
            <p style="font-size: 20px;"><b>Notification Message:</b></p>
            <pre style="background-color: #f4f4f4; padding: 10px; border-left: 4px solid blue;">{message}</pre>
            <p>Thank You!<br><b>MSG Team</b></p>
        </body>
        </html>
        '''
        
        subject = step
        mailer = EmailNotifier()
        mail_message = mailer.build_mail_message(subject=subject, recipients=recipients, body=mail_body)
        mailer.send_email(mail_message)

    def send_error_notification(self, step: str, message: str):
        recipients = ['P13N.BackendEngineers@albertsons.com']
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        mail_body = f'''
        <html>
        <body>
            <h3 style="font-size: 20px;">Hi Team,</h3>
            <p><b>Step:</b> {step}</p>
            <pre style="background-color: #f4f4f4; padding: 10px; border-left: 4px solid red;">{message}</pre>
            <p>Thank You!<br><b>MSG Team</b></p>
        </body>
        </html>
        '''
        subject = step
        mailer = EmailNotifier()
        mail_message = mailer.build_mail_message(subject=subject, recipients=recipients, body=mail_body)
        mailer.send_email(mail_message)

    def send_error_email(self, step: str, error_msg: str):
        recipients = ['P13N.BackendEngineers@albertsons.com']
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        error_lines = error_msg.splitlines()  
        formatted_error_msg = "".join(f"<li>{line.strip()}</li>" for line in error_lines if line.strip())
        mail_body = f'''
        <html>
        <body>
            <h3 style="font-size: 20px;">Hi Team,</h3>
            <p><b>Step:</b> {step}</p>
            <p><b>Error Details:</b></p>
            <ul style="background-color: #f4f4f4; padding: 10px; border-left: 4px solid red;">
                {error_lines}
            </ul>
            <p>Thank You!<br><b>MSG Team</b></p>
        </body>
        </html>
        '''
        
        subject = step
        mailer = EmailNotifier()
        mail_message = mailer.build_mail_message(subject=subject, recipients=recipients, body=mail_body)
        mailer.send_email(mail_message)


    def send_success_email(self, step: str, message: str):
        recipients = ['P13N.BackendEngineers@albertsons.com']
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        mail_body = f'''
        <html>
        <body>
            <h3 style="font-size: 20px;">Hi Team,</h3>
            <p style="font-size: 20px;"><b>Step:</b> {step}</p>
            <p style="font-size: 20px;"><b>Details:</b></p>
            <pre style="background-color: #f4f4f4; padding: 10px; border-left: 4px solid green; font-size: 14px;">{message}</pre>
            <p style="font-size: 20px;">Thank You!<br><b>MSG Team</b></p>
        </body>
        </html>
'''
        subject = step
        mailer = EmailNotifier()
        mail_message = mailer.build_mail_message(subject=subject, recipients=recipients, body=mail_body)
        mailer.send_email(mail_message)

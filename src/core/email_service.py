import os
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

SMTP_SERVER = os.getenv("BREVO_SMTP_SERVER")
SMTP_PORT = int(os.getenv("BREVO_SMTP_PORT", 587))
SMTP_LOGIN = os.getenv("BREVO_SMTP_LOGIN")
SMTP_PASSWORD = os.getenv("BREVO_SMTP_PASSWORD")
FROM_EMAIL = os.getenv("FROM_EMAIL")


def send_verification_email(to_email: str, token: str):
    verify_link = f"http://localhost:3003/verify-email?token={token}"

    subject = "RefineBot Email Verification"
    body = f"""
    Hi there,

    Please verify your email by clicking this link:

    {verify_link}

    Link expires in 24 hours.

    If you did not register, ignore this email.
    """

    msg = MIMEMultipart()
    msg["From"] = FROM_EMAIL
    msg["To"] = to_email
    msg["Subject"] = subject
    msg.attach(MIMEText(body, "plain"))

    with smtplib.SMTP(SMTP_SERVER, SMTP_PORT) as server:
        server.starttls()
        server.login(SMTP_LOGIN, SMTP_PASSWORD)
        server.send_message(msg)

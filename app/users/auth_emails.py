# app/users/auth_emails.py


from config.appconfig import settings
from fastapi import HTTPException
import resend
import os


resend.api_key = settings.RESEND_API_KEY


# =================================================
# ✅ send registration email with verification code
# =================================================
def send_registration_email_with_verification_code(email, verification_code, first_name="User"):
    if not first_name:
        first_name = "User"
        
    r = resend.Emails.send(
        {
            "from": "support@medivarse.com",
            "to": email,
            "subject": "Verify your email!",
            "html": f"<p>Hello {first_name}. \
        Welcome to Dentobase. \
        To verify your account, use the code below when prompted to enter your verification code. \
        This code is meant to not be shared to anyone</p>"
            f"<p><strong> {verification_code} </strong></p>"
            f"<p><strong> Dentobase </strong></p>",
        }
    )


# =================================================
# ✅ send reset password link with token in email
# =================================================
def send_reset_password_link_with_token_in_email(email, reset_link, first_name="User"):
    if not first_name:
        first_name = "User"
        
    r = resend.Emails.send(
        {
            "from": "support@medivarse.com",
            "to": email,
            "subject": "Reset your password!",
            "html": f"<p> Hello {first_name}. We are sorry to hear that you have been having trouble logging in on your Dentobase account. \
          To reset your password, click the link below</p>"
            f"<p><a href='{reset_link}'>{reset_link}</a></p>"
            "<p>You can only use this link once, not to be shared to anyone</p>"
            f"<p><strong> Dentobase </strong></p>",
        }
    )

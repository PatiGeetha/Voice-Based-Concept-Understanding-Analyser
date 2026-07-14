"""
modules/email_utils.py

SMTP email dispatcher for VBCUA.
Sends 6-digit OTP verification codes to user email inboxes.
"""

import logging
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

logger = logging.getLogger(__name__)


def send_otp_email(
    to_email: str,
    otp: str,
    smtp_host: str,
    smtp_port: int,
    smtp_user: str,
    smtp_password: str,
    smtp_from: str = None
) -> bool:
    """
    Connects to the specified SMTP server and sends a verification OTP email.
    
    Args:
        to_email: The target recipient's email address.
        otp: The 6-digit validation code.
        smtp_host: SMTP server address.
        smtp_port: SMTP server port.
        smtp_user: SMTP login username.
        smtp_password: SMTP login password or App Password.
        smtp_from: Optional sender email. Defaults to smtp_user.
        
    Returns:
        True if the email was sent successfully.
        
    Raises:
        Exception: SMTP transmission errors or authorization failures.
    """
    if not smtp_host or not smtp_user or not smtp_password:
        raise ValueError("SMTP Host, Username, and Password must be fully configured.")
        
    sender = smtp_from or smtp_user
    
    # Create Message
    msg = MIMEMultipart("alternative")
    msg["Subject"] = f"🎙️ VBCUA — Your Verification OTP Code: {otp}"
    msg["From"] = sender
    msg["To"] = to_email
    
    # Plain text version
    text_content = f"""
Hello,

Welcome to the Voice-Based Concept Understanding Analyser (VBCUA)!

To complete your sign-up and unlock the dashboard, please enter the following 6-digit verification code:

👉 {otp}

This code is temporary and will expire soon. If you did not request this code, you can safely ignore this email.

Best regards,
The VBCUA Team
"""

    # HTML version (styled to match the premium dark theme)
    html_content = f"""
<!DOCTYPE html>
<html>
<head>
  <style>
    body {{
      font-family: 'Inter', Helvetica, Arial, sans-serif;
      background-color: #0a0e1a;
      color: #c9d1d9;
      margin: 0;
      padding: 0;
    }}
    .email-container {{
      max-width: 500px;
      margin: 30px auto;
      background: #0d1117;
      border: 1px solid #30363d;
      border-radius: 12px;
      padding: 32px;
      box-shadow: 0 4px 12px rgba(0,0,0,0.5);
    }}
    .header {{
      text-align: center;
      margin-bottom: 24px;
    }}
    .title {{
      font-size: 22px;
      font-weight: 800;
      color: #ffffff;
      margin: 0;
    }}
    .code-box {{
      text-align: center;
      background: rgba(46,204,113,0.1);
      border: 2px dashed #2ecc71;
      border-radius: 8px;
      padding: 16px;
      font-size: 32px;
      font-weight: 800;
      letter-spacing: 4px;
      color: #2ecc71;
      margin: 28px 0;
    }}
    .footer {{
      text-align: center;
      font-size: 11px;
      color: #8b949e;
      margin-top: 32px;
      border-top: 1px solid #21262d;
      padding-top: 16px;
    }}
  </style>
</head>
<body>
  <div class="email-container">
    <div class="header">
      <h2 class="title">🎙️ VBCUA</h2>
      <p style="color:#8b949e; margin: 4px 0 0 0; font-size:14px;">Voice-Based Concept Understanding Analyser</p>
    </div>
    <p style="font-size:15px; line-height:1.5;">Hello,</p>
    <p style="font-size:15px; line-height:1.5;">To unlock the VBCUA web dashboard, please copy and verify the following 6-digit OTP code:</p>
    <div class="code-box">{otp}</div>
    <p style="font-size:13px; line-height:1.4; color:#8b949e;">If you did not request this verification, you can safely ignore this email.</p>
    <div class="footer">
      Powered by Google Gemini · Whisper STT · Sentence-BERT<br>
      © 2026 VBCUA. All rights reserved.
    </div>
  </div>
</body>
</html>
"""

    msg.attach(MIMEText(text_content, "plain"))
    msg.attach(MIMEText(html_content, "html"))
    
    # Establish connection and transmit
    logger.info("Connecting to SMTP server %s:%d", smtp_host, smtp_port)
    with smtplib.SMTP(smtp_host, smtp_port, timeout=10) as server:
        server.starttls()
        logger.info("Logging into SMTP account %s", smtp_user)
        server.login(smtp_user, smtp_password)
        server.send_message(msg)
        
    logger.info("OTP verification email sent successfully to %s", to_email)
    return True

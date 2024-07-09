import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.base import MIMEBase
from email import encoders
import os


def send_email(subject, body, to_email, from_email, from_password, attachment_path):
    # Create message container
    msg = MIMEMultipart()
    msg['From'] = from_email
    msg['To'] = to_email
    msg['Subject'] = subject

    # Attach the body with the msg instance
    msg.attach(MIMEText(body, 'plain'))

    # Open the file to be sent
    filename = os.path.basename(attachment_path)
    attachment = open(attachment_path, "rb")

    # Instance of MIMEBase and named as p
    part = MIMEBase('application', 'octet-stream')

    # To change the payload into encoded form
    part.set_payload(attachment.read())

    # Encode into base64
    encoders.encode_base64(part)

    part.add_header('Content-Disposition', f"attachment; filename= {filename}")

    # Attach the instance 'part' to instance 'msg'
    msg.attach(part)

    try:
        # Create a secure SSL context
        server = smtplib.SMTP('smtp.gmail.com', 587)
        server.starttls()

        # Login to the server
        server.login(from_email, from_password)

        # Send email
        text = msg.as_string()
        server.sendmail(from_email, to_email, text)

        # Quit the server
        server.quit()

        print("Email sent successfully!")
    except Exception as e:
        print(f"Failed to send email. Error: {str(e)}")



attachment_path = "cv/ResumeeZeynepTozge.pdf"  # e.g., "/path/to/your/file.pdf"



# Example usage
from_email = "tozgezeynep@gmail.com"
from_password = "nhtp jled vmdy gomw"
to_email = "tobiasschmidbauer1312@gmail.com"
subject = "Test Subject"
body = "This is a test email."

send_email(subject, body, to_email, from_email, from_password, attachment_path)

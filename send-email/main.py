import smtplib
import ssl
from  email.message import EmailMessage 

subject = "Email from Python"
sender_email = "subrahmanyampampana1997@gmail.com"
reciever_email = "subrahmanyampampana1997@gmail.com"
body = "hey!, this message from Python"
password = input("Enter password: ")

message = EmailMessage()
message["From"] = sender_email
message["To"] = reciever_email
message["Subject"] = subject
message.set_content(body)

context = ssl.create_default_context()
print("sending Email")
with smtplib.SMTP_SSL("smtp.gmail.com",465,context = context) as server:
    server.login(sender_email,password)
    server.sendmail(sender_email,reciever_email,message.as_string())
print("email sent")
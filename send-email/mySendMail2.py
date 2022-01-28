# Python code to illustrate Sending mail from
# your Gmail account
import smtplib

subject = "Email from Python"
sender_email = "subrahmanyampampana1997@gmail.com"
reciever_email = "subrahmanyampampana1997@gmail.com"
body = "hey!, this message from Python"
password = input("Enter password: ")

# creates SMTP session
s = smtplib.SMTP('smtp.gmail.com', 587)

# start TLS for security
s.starttls()

# Authentication
s.login(sender_email, password)

# sending the mail
s.sendmail(sender_email, reciever_email, body)

print("Mail sent")
# terminating the session
s.quit()

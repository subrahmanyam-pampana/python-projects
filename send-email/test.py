import myAttachment


fromaddr = "subrahmanyampampana1997@gmail.com"
toaddr = "nagarevathi27@gmail.com"
subject = "Email from Subrahmanyam using python"
body = "hey!, this message from Python"
filename = "test.txt"
print(myAttachment.sendMail(fromaddr=fromaddr,toaddr=toaddr,subject=subject,body=body,filename=filename))
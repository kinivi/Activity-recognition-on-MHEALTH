import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText


def sendgrid():
    mail_from = 'Neural Network <kinivi210999@gmail.com>'
    mail_to = 'Nikita Kiselov <kinivi210999@gmail.com>'

    msg = MIMEMultipart()
    msg['From'] = mail_from
    msg['To'] = mail_to
    msg['Subject'] = 'Training end'
    mail_body = """
    Hey,
    
    Training is end
    
    Regards,\nyour NL ❤️
    
    """
    msg.attach(MIMEText(mail_body))

    try:
        server = smtplib.SMTP_SSL('smtp.sendgrid.net', 465)
        server.ehlo()
        server.login('apikey', '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')
        server.sendmail(mail_from, mail_to, msg.as_string())
        server.close()
        print("mail sent")
    except:
        print("issue")
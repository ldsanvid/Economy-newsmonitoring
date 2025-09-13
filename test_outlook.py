import smtplib

remitente = "monitoreo.plus@outlook.com"
password = "ldzotwfazbfshvzf"
destinatario = "ldsantiagovidargas.93@gmail.com"

try:
    server = smtplib.SMTP("smtp.office365.com", 587)
    server.starttls()
    server.login(remitente, password)
    print("✅ Login correcto, la contraseña de aplicación funciona")
except Exception as e:
    print("❌ Error:", e)

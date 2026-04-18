#!/usr/bin/env python3
import sqlite3, hashlib, base64, secrets, sys

db='/label-studio/data/label_studio.sqlite3'
email='taavi.pipar+lablestudio@gmail.com'
new_password='TempPass123!'
iterations=870000
salt=secrets.token_urlsafe(12)
dk=hashlib.pbkdf2_hmac('sha256', new_password.encode(), salt.encode(), iterations)
hash_b64=base64.b64encode(dk).decode().strip()
pw = f"pbkdf2_sha256${iterations}${salt}${hash_b64}"

con=sqlite3.connect(db)
cur=con.cursor()
cur.execute("UPDATE htx_user SET password=? WHERE email=?;", (pw, email))
con.commit()
cur.execute("SELECT id, email FROM htx_user WHERE email=?;", (email,))
row=cur.fetchone()
con.close()
if row:
    print('SUCCESS', row)
    print('Temporary password:', new_password)
else:
    print('USER NOT FOUND', email)
    sys.exit(2)

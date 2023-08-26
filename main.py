from flask import Flask, send_file, request, jsonify, render_template, redirect, url_for, session
from flask_restx import Resource, Api, reqparse
import json
import pickle
from flask_mysqldb import MySQL 
from datetime import datetime
import os
os.environ["OMP_NUM_THREADS"] = "1"
import numpy as np

app = Flask(__name__)

app.config['MYSQL_HOST'] = 'localhost'
app.config['MYSQL_USER'] = 'root'
app.config['MYSQL_PASSWORD'] = ''
app.config['MYSQL_DB'] = 'repository_skripsi'

mysql = MySQL(app)

# Load Model
model = pickle.load(
    open('C:/xampp/htdocs/web_klasifikasi/model/model.pkl', 'rb'))
load_vec = pickle.load(
    open('C:/xampp/htdocs/web_klasifikasi/model/count_vect.pkl', 'rb'))


@app.route('/', methods=['GET', 'POST'])
def send():
    # Post Data
    if request.method == 'POST':
        id_identitas = request.form['id_identitas']
        nama_lengkap = request.form['nama_lengkap']
        prodi = request.form['prodi']
        sebagai = request.form['sebagai']
        gender = request.form['gender']
        id_paket_jawaban = request.form['id_paket_jawaban']
        jawaban = request.form['jawaban']
        
        klas = model.predict(load_vec.transform([jawaban]))
        label = np.array(klas[0])

        cursor = mysql.connection.cursor()
        cursor.execute(''' INSERT INTO klasifikasi VALUES(%s,%s,%s,%s,%s,%s,%s,%s,%s) ''', (id, id_identitas, nama_lengkap, prodi, sebagai, gender, id_paket_jawaban, jawaban, label.tolist()))
        mysql.connection.commit()
        
        return {
                'Identitas': id_identitas,
                'Nama Lengkap': nama_lengkap,
                'Prodi': prodi,
                'Sebagai': sebagai,
                'Gender': gender,
                'id_paket_jawaban': id_paket_jawaban,
                'Jawaban': jawaban,
                'label': label.tolist(),
                'message': f"Data jawaban berhasil masuk!"
        }

    #  Get Data
    if request.method == 'GET':
        cur = mysql.connection.cursor()
        cur.execute(''' SELECT * FROM klasifikasi ''')
        data = cur.fetchall()
        if (data is None):
                return f"Tidak Ada Data!"
        else:
            wadah = []
            for row in data:
                wadah.append({
                        "ID": row[0],
                        "ID Identitas": row[1],
                        "Nama Lengkap": row[2],
                        "Prodi": row[3],
                        "Sebagai": row[4],
                        "Gender": row[5],
                        "Id Paket": row[6],
                        "Jawaban": row[7],
                        "Label": row[8],
                    })
            return wadah
            
if __name__ == '__main__':
    app.run()
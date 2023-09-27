# -*- coding: utf-8 -*-
import numpy as np
import sqlite3
import sqlite_vss
import json
from bitarray import bitarray

vector = [0.1, 0.2]

# SQLite側でバイナリに変換
con = sqlite3.connect(':memory:')
con.enable_load_extension(True)
sqlite_vss.load(con)
con.execute('CREATE TABLE demo(json_vector TEXT,vector BLOB);')
con.execute('INSERT INTO demo(json_vector) VALUES (?)',
            (json.dumps(vector), ))
con.execute('UPDATE demo SET vector = vector_to_blob(vector_from_json(json_vector));')
sqlite_vector_bytes = con.execute('SELECT * FROM demo').fetchone()[1]
bits = bitarray(endian='big')
bits.frombytes(sqlite_vector_bytes)
print('SQLite: ', bits.to01())

# numpyでバイナリに変換
np_bytes = np.array(vector).astype(np.float32).tobytes()
bits = bitarray(endian='big')
bits.frombytes(np_bytes)
print('Numpy:  ', bits.to01())

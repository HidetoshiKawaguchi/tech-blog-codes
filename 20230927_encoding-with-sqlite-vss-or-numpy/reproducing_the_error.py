import sqlite3
import sqlite_vss
import numpy as np


##############################################
# 1. SQLite3データベースのセットアップ
##############################################
con = sqlite3.connect(':memory:')
con.enable_load_extension(True)
sqlite_vss.load(con)

##############################################
# 2. ベクトルデータの生成
##############################################
np.random.seed(3939)   # 乱数シードを固定
n_samples, x_dim = 50000, 3000
X = np.random.uniform(-2, 2, (n_samples, x_dim))
X = X.astype(np.float32)

##############################################
# 3. テーブルの生成
##############################################
table_name = 'vectors'
vss_table_name = 'vss_pq_word'
con.execute(f'CREATE TABLE {table_name} (vector BLOB);')
con.execute('''
CREATE VIRTUAL TABLE {} using vss0 (
    vector({}) factory="PQ15,IDMap2"
);'''.format(vss_table_name, x_dim))

##############################################
# 4. ベクトルデータベースの構築および訓練
##############################################
for vector in X:
    con.execute(f'INSERT INTO {table_name} (vector) VALUES (?)',
                (vector.tobytes(), ))
con.commit()
con.execute(
    """
    INSERT INTO {} (operation, vector)
    SELECT 'training', vector FROM {}
    """.format(vss_table_name, table_name)
)
con.commit()

print('プログラムが正常に実行されました。')

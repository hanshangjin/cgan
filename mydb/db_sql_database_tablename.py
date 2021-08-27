# encoding: utf-8
import zlib
import sqlite3 as sqlite
import numpy as np
# import json
import ujson


class MySqliteDatabaseWithTablename():
	def __init__(self, file):
		self.connect = sqlite.connect(file)
		self.cursor = self.connect.cursor()

		# for epoch shuffle
		self.count = 0
		self.idx=0
		self.idx_list = None

		# for rename table
		self.tablename = 'molecule'


	# +++++++shuffle starts++++++++++++++++++++++
	def get_idx(self, idx):
		if self.idx_list == None:
			self.init_idx_list()
		self.count += 1
		_idx = idx
		if self.count == self.db_size:
			self.shuffle_idx_list()
			self.count = 0
		# return self.idx_list[_idx-1]
		return self.idx_list[_idx]

	def shuffle_idx_list(self):
		np.random.shuffle(self.idx_list)

	def init_idx_list(self):
		self.idx_list = []
		self.db_size = self.get_size()
		for i in range(self.db_size):
			# self.idx_list.append(i+1)
			self.idx_list.append(i)
		self.shuffle_idx_list()
	# +++++++shuffle finished++++++++++++++++++++++
	def set_tablename(self, tablename):
		self.tablename = tablename

	"""
	tup_dict:	Pack the tuples stored in the database into a dictionary object, so that they can be converted into json data;

	"""
	# write data items into database
	def add_tuple(self, tup_dict): # id is not needed, because the id value of datatable is automatically incremented
		tup_string = ujson.dumps(tup_dict)

		# string --> bytes
		tup_bytes = tup_string.encode()
		tup_bytes_compressed = zlib.compress(tup_bytes)

		# bytes --> BLOB(Binary Large OBject)
		tup_blob = sqlite.Binary(tup_bytes_compressed)

		# insert BLOB to DB table: frames
		idx = self.idx
		self.cursor.execute('insert into %s values (%s, ?)'%(self.tablename, idx), (tup_blob, ))
		self.idx += 1

	# Used for writing auxiliary tables
	def add_tuple_strID(self, strID, tup_dict): #tup_dict: python dictionary object

		# Converts arbitrary object recursively into JSON. 
		# Use ensure_ascii=false to output UTF-8
		tup_string = ujson.dumps(tup_dict)

		# string --> bytes
		tup_bytes = tup_string.encode()
		tup_bytes_compressed = zlib.compress(tup_bytes)

		# bytes --> BLOB(Binary Large OBject)
		tup_blob = sqlite.Binary(tup_bytes_compressed)

		# insert BLOB to DB table
		# self.cursor.execute('insert into datatable values (null, ?)', (tup_blob, ))
		str_id = '"' + strID + '"'
		# self.cursor.execute('insert into %s values (null, %s, ?)'%(self.tablename, str_id), (tup_blob, ))
		self.cursor.execute('insert into %s values (%s, ?)'%(self.tablename, str_id), (tup_blob, ))

	# read data items from database
	def get_tuple(self, idx):
		_idx = self.get_idx(idx)

		sql = 'select tuple from %s where id=?'%(self.tablename)
		rows = self.cursor.execute(sql, (_idx, ))
		tup_blob_rows = rows.fetchone()
		tup_bytes = tup_blob_rows[0]

		# tup_bytes = BytesIO(np.array(tup_blob)).getbuffer()

		# bytes --> decompressed
		# decompressed = zlib.decompress(tup_bytes.getbuffer())
		decompressed = zlib.decompress(tup_bytes)

		# decompressed --> string
		tup_string = decompressed.decode()

		# string --> python dictionary object
		# tup_dict = json.loads(tup_string)
		tup_dict = ujson.loads(tup_string)

		# dictionary object --> tuple
		# return (tup_dict["X_drug"], tup_dict["X_target"], tup_dict["y"])
		return tup_dict

	# Used for reading of auxiliary tables
	def get_tuple_strID(self, strID):
		sql = "select tuple from %s where id=?"%(self.tablename)
		rows = self.cursor.execute(sql, (strID, ))
		tup_blob_rows = rows.fetchone()
		tup_bytes = tup_blob_rows[0]

		decompressed = zlib.decompress(tup_bytes)

		# decompressed --> string
		tup_string = decompressed.decode()

		# string --> python dictionary object
		tup_dict = ujson.loads(tup_string)
		return tup_dict

	# database size
	def get_size(self):
		sql = 'select count(id) from %s'%(self.tablename)
		rows = self.cursor.execute(sql)
		row_tuple = rows.fetchone()
		count = row_tuple[0]
		return count

	def create_table(self):
		sql = 'create table %s (id integer primary key, tuple blob)'%(self.tablename)
		self.cursor.execute(sql)

	# Used for auxiliary table creation
	def create_table_strID(self):
		sql = 'create table %s (id text primary key, tuple blob)'%(self.tablename)
		self.cursor.execute(sql)

	def close(self):
		self.cursor.close()
		self.connect.close()

	def commit(self):
		self.connect.commit()

	def close_with_commit(self):
		self.cursor.close()
		self.connect.commit()
		self.connect.close()


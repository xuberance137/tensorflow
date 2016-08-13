
import tempfile
import urllib
import pandas as pd
import tensorflow as tf

flags = tf.app.flags
FLAGS = flags.FLAGS

flags.DEFINE_string("model_dir", "", "Base directory for output models.")
flags.DEFINE_string("model_type", "wide_n_deep", "Valid model types: {'wide', 'deep', 'wide_n_deep'}.")
flags.DEFINE_integer("train_steps", 200, "Number of training steps.")
flags.DEFINE_string("train_data", "", "Path to the training data.")
flags.DEFINE_string("test_data", "", "Path to the test data.")

#reading data into a pandas dataframe
COLUMNS = ["age", "workclass", "fnlwgt", "education", "education_num",
           "marital_status", "occupation", "relationship", "race", "gender",
           "capital_gain", "capital_loss", "hours_per_week", "native_country",
           "income_bracket"]
#splitting to categorical and continuous variables
CATEGORICAL_COLUMNS = ["workclass", "education", "marital_status", "occupation", "relationship", "race", "gender", "native_country"]
CONTINUOUS_COLUMNS = ["age", "education_num", "capital_gain", "capital_loss", "hours_per_week"]

def input_fn(df):

	continuous_cols = {k: tf.constant(df[k].values) for k in CONTINUOUS_COLUMNS}
	
	categorical_cols = {k: tf.SparseTensor(
		indices = [[i, 0] for i in range(df[k].size)],
		values = df[k].values, 
		shape = [df[k].size, 1]) for k in CATEGORICAL_COLUMNS}

	features_cols = dict(continuous_cols.items() + categorical_cols.items())
	label = tf.constant(df[LABEL_COLUMN].values)

	return features_cols, label

def train_input_fn():
	return input_fn(df_train)

def test_input_fn():
	return input_fn(df_test)


def build_estimator(model_dir):
	"""Build an estimator."""
	# Categorical variables to base columns
	# Creating sparse base columns, creating keys starting with zero
	gender = tf.contrib.layers.sparse_column_with_keys(column_name="gender", keys=["female", "male"])
	race = tf.contrib.layers.sparse_column_with_keys(column_name="race", keys=["Amer-Indian-Eskimo", "Asian-Pac-Islander", "Black", "Other", "White"])
	# Creating sparse column without knowing set of possible values in advance
	education = tf.contrib.layers.sparse_column_with_hash_bucket("education", hash_bucket_size=1000)
	marital_status = tf.contrib.layers.sparse_column_with_hash_bucket("marital_status", hash_bucket_size=100)
	relationship = tf.contrib.layers.sparse_column_with_hash_bucket("relationship", hash_bucket_size=100)
	workclass = tf.contrib.layers.sparse_column_with_hash_bucket("workclass", hash_bucket_size=100)
	occupation = tf.contrib.layers.sparse_column_with_hash_bucket("occupation", hash_bucket_size=1000)
	native_country = tf.contrib.layers.sparse_column_with_hash_bucket("native_country", hash_bucket_size=1000)

  	# Continuous base columns.
	age = tf.contrib.layers.real_valued_column("age")
	education_num = tf.contrib.layers.real_valued_column("education_num")
	capital_gain = tf.contrib.layers.real_valued_column("capital_gain")
	capital_loss = tf.contrib.layers.real_valued_column("capital_loss")
	hours_per_week = tf.contrib.layers.real_valued_column("hours_per_week")

	# Transformations of continuous to categorical through bucketization
	age_buckets = tf.contrib.layers.bucketized_column(age, boundaries=[18, 25, 30, 35, 40, 45, 50, 55, 60, 65])

	# Defining wide columns
	# including interaction terms
 	wide_columns = [gender, native_country, education, occupation, workclass,
					marital_status, relationship, age_buckets,
					tf.contrib.layers.crossed_column([education, occupation], hash_bucket_size=int(1e4)),
					tf.contrib.layers.crossed_column([age_buckets, race, occupation], hash_bucket_size=int(1e6)),
					tf.contrib.layers.crossed_column([native_country, occupation], hash_bucket_size=int(1e4))]
	
	# Defining deep columns
	deep_columns = [
		tf.contrib.layers.embedding_column(workclass, dimension=8),
		tf.contrib.layers.embedding_column(education, dimension=8),
		tf.contrib.layers.embedding_column(marital_status, dimension=8),
		tf.contrib.layers.embedding_column(gender, dimension=8),
		tf.contrib.layers.embedding_column(relationship, dimension=8),
		tf.contrib.layers.embedding_column(race, dimension=8), 
		tf.contrib.layers.embedding_column(native_country, dimension=8),
		tf.contrib.layers.embedding_column(occupation, dimension=8),
		age,
		education_num,
		capital_gain,
		capital_loss,
		hours_per_week,
		]

	if FLAGS.model_type == "wide":
		print("Creating Wide Model")
		# wide model comprising of a standard linear classifier
		m = tf.contrib.learn.LinearClassifier(model_dir=model_dir, feature_columns=wide_columns)
	elif FLAGS.model_type == "deep":
		print("Creating Deep Model")
		# deep feed forward NN comprising of two hidden layers
		m = tf.contrib.learn.DNNClassifier(model_dir=model_dir, feature_columns=deep_columns, hidden_units=[100, 50])
	else:
		print("Creating Wide+Deep Model")
		# Combined model with wide and deep components using the DNN TF API
		m = tf.contrib.learn.DNNLinearCombinedClassifier(
			model_dir=model_dir, 
			linear_feature_columns=wide_columns,
			dnn_feature_columns=deep_columns,
			dnn_hidden_units=[100, 50])

	return m

### MAIN FUNCTION ###
if __name__ == '__main__':
	#accessing census data
	train_file = tempfile.NamedTemporaryFile()
	test_file = tempfile.NamedTemporaryFile()
	urllib.urlretrieve("https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data", train_file.name)
	urllib.urlretrieve("https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.test", test_file.name)

	df_train = pd.read_csv(train_file, names=COLUMNS, skipinitialspace=True)
	df_test = pd.read_csv(test_file, names=COLUMNS, skipinitialspace=True, skiprows=1)

	#set label to 1 if income > 50K and 0 if income < 50K
	LABEL_COLUMN = "label"
	df_train[LABEL_COLUMN] = (df_train["income_bracket"].apply(lambda x: ">50K" in x)).astype(int)
	df_test[LABEL_COLUMN] = (df_test["income_bracket"].apply(lambda x: ">50K" in x)).astype(int)

	model_dir = tempfile.mkdtemp() if not FLAGS.model_dir else FLAGS.model_dir
	print("Model directory = %s" % model_dir)
	print("Building Estimator")

	model_str = ["wide", "deep", "wide_n_deep"] 

	for item in model_str:
	
		FLAGS.model_type = item

		m = build_estimator(model_dir)

		step_vals = [1, 10, 200]
		for val in step_vals:
			print("Fitting Training Data")
			print("Results with Step Count = %d" % val)
			m.fit(input_fn=lambda: input_fn(df_train), steps=val)

			print("Evaluating on Test Data")
			results = m.evaluate(input_fn=lambda: input_fn(df_test), steps=1)
			for key in sorted(results):
				print("%s: %s" % (key, results[key]))






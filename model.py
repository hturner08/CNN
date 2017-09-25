import tempfile
import urllib.request
import tensorflow as tf
import pandas as pd

train_file = tempfile.NamedTemporaryFile()
test_file = tempfile.NamedTemporaryFile()
urllib.request.urlretrieve("https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data", train_file.name)
urllib.request.urlretrieve("https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.test", test_file.name)

CSV_COLUMNS = [
"age", "workclass", "fnlwgt", "education", "education_num",
"marital_status", "occupation", "relationship", "race", "gender",
"capital_gain", "capital_loss", "hours_per_week", "native_country",
"income_bracket"]
df_train = pd.read_csv(train_file.name, names=CSV_COLUMNS, skipinitialspace=True)
df_test = pd.read_csv(test_file.name, names=CSV_COLUMNS, skipinitialspace=True, skiprows=1)

def input_fn(data_file, num_epochs, shuffle):
    """Input builder function."""
    df_data = pd.read_csv(
    tf.gfile.Open(data_file),
    names=CSV_COLUMNS,
    skipinitialspace=True,
    engine="python",
    skiprows=1)
    # remove NaN elements
    df_data = df_data.dropna(how="any", axis=0)
    labels = df_data["income_bracket"].apply(lambda x: ">50K" in x).astype(int)
    return tf.estimator.inputs.pandas_input_fn(
    x=df_data,
    y=labels,
    batch_size=100,
    num_epochs=num_epochs,
    shuffle=shuffle,
    num_threads=5)

    # Base categorical Features
    gender = tf.feature_column.categorical_column_with_vocabulary("gender",["Female","Male"])
    occupation = tf.feature_column.categorical_column_with_hash_bucket("occupation", hash_bucket_size=1000)
    education = tf.feature_column.categorical_column_with_vocabulary("education",["Bachelors", "HS-grad", "11th", "Masters", "9th",
    "Some-college", "Assoc-acdm", "Assoc-voc", "7th-8th",
    "Doctorate", "Prof-school", "5th-6th", "10th", "1st-4th",
    "Preschool", "12th"])
    marital_status = tf.feature_column.categorical_column_with_vocabulary("marital_status",["Married-civ-spouse", "Divorced", "Married-spouse-absent",
    "Never-married", "Separated", "Married-AF-spouse", "Widowed"
    ])
    relationship = tf.feature_column.categorical_column_with_vocabulary_list(
    "relationship", [
    "Husband", "Not-in-family", "Wife", "Own-child", "Unmarried",
    "Other-relative"
    ])
    workclass = tf.feature_column.categorical_column_with_vocabulary_list(
    "workclass", [
    "Self-emp-not-inc", "Private", "State-gov", "Federal-gov",
    "Local-gov", "?", "Self-emp-inc", "Without-pay", "Never-worked"
    ])
    native_country = tf.feature_column.categorical_column_with_hash_bucket(
    "native_country", hash_bucket_size=1000)

    # Base Continuous Features
    age = tf.feature_column.numeric_column("age")
    education_num = tf.feature_column.numeric_column("education_num")
    capital_gain = tf.feature_column.numeric_column("capital_gain")
    capital_loss = tf.feature_column.numeric_column("capital_loss")
    hours_per_week = tf.feature_column.numeric_column("hours_per_week")

    #Features where relationship to label isn't always linear
    age_buckets = tf.feature_column.bucketized_column(age, boundaries=[18, 25, 30, 35, 40, 45, 50, 55, 60, 65])

    #Cross-column features
    age_buckets_x_education_x_occupation = tf.feature_column.crossed_column(a)
    education_x_occupation = tf.feature_column.crossed_column(["education", "occupation"], hash_bucket_size=1000)
    native_country_x_occupation = tf.feature_column.crossed_column(["native_country", "occupation"], hash_bucket_size=1000)
    #Building model
    base_columns = [gender, native_country, education, occupation, workclass, relationship, age_buckets]
    crossed_columns= [age_buckets_x_education_x_occupation, education_x_occupation, native_country_x_occupation]
    model_dir = tempfile.mkdtemp()
    model = tf.estimator.LinearClassifier(mdel_dir=model_dir, feature_columns = base_columns + crossed_columns)
    #Training Model
    model.train(
    input_fn=input_fn(train_file_name, num_epochs=None, shuffle=True),
    steps=train_steps)
    results = model.evaluate(input_fn=input_fn(test_file_name, num_epochs=1, shuffle=False),
    steps=None)
    print("model directory = %s" % model_dir)
    for key in sorted(results):
        print("%s: %s" % (key, results[key]))

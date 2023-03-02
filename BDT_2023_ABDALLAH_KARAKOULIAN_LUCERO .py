# Databricks notebook source
# MAGIC %md
# MAGIC # Project Participants: Beirut & Bogota StatsBomb
# MAGIC 
# MAGIC ## ABDALLAH Haitham
# MAGIC ## KARAKOULIAN Maria
# MAGIC ## LUCERO Fabrizio
# MAGIC 
# MAGIC 
# MAGIC libraries and functions for the project...

# COMMAND ----------

#Libraries
import sys
import numpy as np
import pandas as pd
import seaborn as sb
from pyspark.ml import Pipeline
import matplotlib.pyplot as plt
from pyspark import SparkContext
from pyspark.sql.functions import *
from pyspark.sql.window import Window
from sklearn.metrics import roc_curve
from pyspark.ml.stat import Correlation
from pyspark.mllib.stat import Statistics
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.classification import OneVsRest
from pyspark.ml.functions import vector_to_array
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.feature import OneHotEncoder, StringIndexer, MaxAbsScaler, RFormula, Imputer, PCA
from pyspark.sql.types import TimestampType, DateType, StringType, StructType, StructField, DoubleType
from pyspark.ml.classification import RandomForestClassifier, GBTClassifier, LogisticRegression, LinearSVC, FMClassifier, MultilayerPerceptronClassifier


#SOURCE: Functions library in Spark https://spark.apache.org/docs/latest/api/python/reference/pyspark.sql/functions.html

#functions
def show_data(train_data, test_data):
    print(f"{train_data} TABLE")
    print(" TRAIN DATASET ")
    print(" ")
    train_data.show(3)
    print("TEST DATASET ")
    print(" ")
    test_data.show(3)

# Function to create the new variables
def create_orders_variables(df):
    'The function calculates the "delivery_duration_days" and "delivery_delay_days" by subtracting the "order_purchase_timestamp" from the "order_delivered_customer_date" and "order_estimated_delivery_date", respectively. It then adds a "status" column to the dataframe by transforming the 8 possible values of the "order_status" column into 4 values based on a conditional statement. The data is grouped by "customer_id" and a "repeat_customer" column is added indicating if a customer has made multiple orders. The original dataframe is then joined with the grouped data using a left join. Lastly, the function adds two columns to the dataframe indicating the presence of delivery duration and delay information.'
    df = df.withColumn("delivery_duration_days", datediff(col("order_delivered_customer_date"),col("order_purchase_timestamp")))
    df = df.withColumn("delivery_delay_days", datediff(col("order_delivered_customer_date"),col("order_estimated_delivery_date")))
    df = df.withColumn("status", when(col("order_status")=="delivered", "delivered")\
                             .when(col("order_status")=="unavailable", "unavailable")\
                             .when(col("order_status")=="canceled", "canceled")\
                             .otherwise("processing"))
    df_group = df.groupBy("customer_id").agg(countDistinct("order_id").alias("number_orders"))
    df_group = df_group.withColumn("repeat_customer", when(col("number_orders")>1, "yes").otherwise("no"))
    df_group = df_group.drop("number_orders")
    df = df.join(df_group, ["customer_id"], "left")
    df = df.withColumn("duration_flag", when(~isnull(df["delivery_duration_days"]), 1).otherwise(0))
    df = df.withColumn("delays_flag", when(~isnull(df["delivery_delay_days"]), 1).otherwise(0))
    return df

def create_order_payments_variables(df):
    'The function creates new columns for order payment information, including the number of payment types per order in the "payment_methods" column and the total number of payment installments and amounts in the "installments" and "amount" columns, respectively. The function first groups the data by "order_id" and aggregates the number of distinct payment types. Then it pivots the data by "payment_type" and calculates the sum of payment installments and values. The "payment_methods" column is added to indicate if an order was made using one or multiple payment types. The result of both operations is combined into a single dataframe.'
    order_payments_group = df.groupBy("order_id").agg(count_distinct("payment_type").alias("number_payment_types"))\
    .withColumn("payment_methods", when(col("number_payment_types")>1, "multiple").otherwise("single"))\
    .drop("number_payment_types")

    order_payments_pivot=df\
    .groupby('order_id')\
    .pivot('payment_type',)\
    .agg(sum("payment_installments").alias('installments'),sum("payment_value").alias('amount'))\
    .na.fill(0)

    order_payments = order_payments_group.join(order_payments_pivot,["order_id"],"left")
    
    return order_payments


#SOURCE FOR FINDING THE CUMSUM OF A COLUMN: https://stackoverflow.com/questions/45946349/python-spark-cumulative-sum-by-group-using-dataframe
#SOURCE FOR UNDERSTANDING RDD FLATMAP|: https://sparkbyexamples.com/pyspark/pyspark-flatmap-transformation/
def create_items_products_variables(order_items, products):
    'The function takes in two dataframes "order_items" and "products", and performs a left join between the two dataframes on the "product_id" column. The function then calculates the volume of a product by multiplying its length, height, and width and adds a new column "prod_volume_cm3". It also calculates the shipping cost per gram of a product by dividing the shipping cost by its weight and adds a new column "shipping_cost_per_g". The function groups the data by the "product_category_name" column and identifies categories that are present in less than 80% of the orders. The product categories that fall under this criteria are transformed into the value "not_in_top_20". The function then pivots the data and groups it by the "order_id", creating columns for each product category indicating the number of products in each category. The function also groups the data by the "order_id" and calculates the mean values for several columns. The resulting dataframe is joined with the pivoted data and the joined dataframe is returned.'
    items_products = order_items.join(products, ["product_id"], "left")
    items_products = items_products\
        .withColumn('prod_volume_cm3', col("product_length_cm") * col("product_height_cm") * col("product_width_cm"))\
        .drop("product_length_cm","product_height_cm","product_width_cm")\
        .withColumn('shipping_cost_per_g', col("shipping_cost") / col("product_weight_g"))

    categories = items_products\
        .groupby('product_category_name')\
        .agg(count('product_id').alias('present in'))\
        .sort(col('present in').desc())\
        .withColumn('Accumulate', sum(col('present in')).over(Window.partitionBy().orderBy().rowsBetween(-sys.maxsize, 0)))

    categories = categories\
        .where(col('Accumulate') < (categories.agg(max(col("Accumulate"))).collect()[0][0] * 0.8))\
        .select('product_category_name')\
        .rdd.flatMap(lambda x: x).collect()

    items_products = items_products\
        .withColumn('product_category_name', when(col('product_category_name').isin(categories),
                                                 'in_top_20').otherwise('not_in_top_20'))

    items_products_ = items_products\
        .groupby('order_id')\
        .pivot('product_category_name')\
        .agg(count('product_id'))\
        .na.fill(0)

    items_products__ = items_products\
        .groupby('order_id')\
        .agg(mean('price').alias('avg_price'),
             mean('shipping_cost').alias('avg_ship_cost'),
             mean('product_name_lenght').alias('avg_prod_name_length'),
             mean('product_description_lenght').alias('avg_prod_desc_length'),
             mean('product_photos_qty').alias('avg_photos_qty'),
             mean('product_weight_g').alias('avg_prod_weight_g'),
             mean('prod_volume_cm3').alias('avg_prod_volume_cm3'),
             mean('shipping_cost_per_g').alias('avg_ship_cost_per_g'))\
        .na.fill(0)

    items_products = items_products_.join(items_products__, ["order_id"], "inner")
    return items_products

def clean_orders_na_values(df, col_to_analyze, columns_na):
    'This function performs several operations to clean up the orders data. It starts by printing the number of missing values in each column in the dataframe. Then it removes any rows with missing values in the "col_to_analyze" column and fills in missing values in the "columns_na" with the value 999. The function then casts the "col_to_analyze" columns to StringType and fills in any remaining missing values with "2080-01-01 01:00:00".'
    print('at the beginning:')
    df.select([count(when(col(c).isNull(), c)).alias(c) for c in df.columns]).show()
    
    df = df\
    .dropna(subset=col_to_analyze, how='all')\
    .fillna(999,subset=columns_na)\
    .select([column(c).cast(StringType()).alias(c) if c in col_to_analyze else c for c in df.columns])\
    .fillna("2080-01-01 01:00:00", subset=col_to_analyze)
    
    print('after cleaning:')
    df.select([count(when(col(c).isNull(), c)).alias(c) for c in df.columns]).show()
    print('remaining rows:', df.count())
    
    return df


def plot_roc_curve(model, data, data1):
    'The function plot_roc_curve takes in three inputs: model, data, and data1. The function uses the model to make predictions on both data and data1, and it plots the ROC curve for each set of predictions.The first step is to use the model to make predictions on data, and extract the prediction scores and labels from the predictions. These scores and labels are then converted to a NumPy array. The function then calculates the false positive rate (fpr), true positive rate (tpr), and thresholds for each prediction using the roc_curve function. The ROC curve for the data is then plotted using these values.The process is repeated for data1 with the same steps applied, and the resulting ROC curve is plotted on the same figure. The plot is labeled with the "False Positive Rate" and "True Positive Rate" on the x-axis and y-axis respectively, and it has a title "ROC Curve". The two curves are labeled with "training" and "validate". Finally, the legend is added to the plot and it is displayed using the plt.show() function.'
    pred=model.transform(data)
    scoreAndLabels_=pred.select("probability", "label").rdd.map(lambda row: (row[0][1], row[1]))
    scoreAndLabelsArray= np.array(scoreAndLabels_.collect())
    fpr, tpr, thresholds= roc_curve(scoreAndLabelsArray[:, 1], scoreAndLabelsArray[:, 0])
    plt.figure(figsize=(10, 8))
    plt.plot(fpr, tpr, label="training")

    pred1= model.transform(data1)
    scoreAndLabels1= pred1.select("probability", "label").rdd.map(lambda row: (row[0][1], row[1]))
    scoreAndLabelsArray1= np.array(scoreAndLabels1.collect())
    fpr1, tpr1, thresholds1= roc_curve(scoreAndLabelsArray1[:, 1], scoreAndLabelsArray1[:, 0])
    plt.plot(fpr1, tpr1, label="validate")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend()
    plt.show()
    
def plot_lift_curve_from_df(df, prediction_col, label_col, title=None):
    'The function is creating a Lift Curve, which is a graphical representation of the performance of a binary classification model. It starts by transforming the data into a RDD (Resilient Distributed Dataset) and mapping the columns of interest, "prediction_col" and "label_col", into pairs of (float, float). Then it sorts the RDD by the prediction value in descending order.Next, it counts the total number of positive and negative samples in the data. Then it splits the sorted data into 10 deciles, with each decile representing 10% of the positive samples. For each decile, it calculates the lift value as the ratio of the number of positive samples in the decile divided by the total number of positive samples over the total number of samples in the decile over the total number of samples.Finally, it returns two lists, "deciles" and "lift", which represent the x and y values respectively of the Lift Curve.'
    predictionAndLabels = df.select(prediction_col, label_col).rdd.map(lambda row: (float(row[prediction_col]), float(row[label_col])))
    sortedPredictionLabels = predictionAndLabels.sortBy(lambda x: x[0], ascending=False)
    totalPositives = sortedPredictionLabels.filter(lambda x: x[1] == 1.0).count()
    totalNegatives = sortedPredictionLabels.filter(lambda x: x[1] == 0.0).count()
    decile = totalPositives / 10.0
    deciles = []
    lift = []
    current_positive = 0.0
    current_negative = 0.0
    current_decile = 0.0
    current_index = 0
    for row in sortedPredictionLabels.collect():
        if row[1] == 1.0:
            current_positive += 1.0
        else:
            current_negative += 1.0
            current_index += 1.0
        if current_positive >= current_decile + decile:
            deciles.append(current_index / (totalPositives + totalNegatives))
            lift.append((current_positive / current_index) / (current_positive / (totalPositives)))
            current_decile += decile
    deciles.append(1.0)
    lift.append(1.0)
    
    return deciles, lift


def plot_multiple_lift_curves(df1, df2, prediction_col, label_col, title=None):
    'This function plots multiple lift curves in the same plot by calling the plot_lift_curve_from_df function twice, once for each dataframe passed as arguments df1 and df2. It takes the decile and lift values from both the dataframes and plots two separate curves on the same plot. It also plots a reference line with lift equal to 1 and adds labels for the x and y axis. The plot is then displayed using plt.show(). If the optional title argument is passed, it will be used as the title for the plot.'
    deciles1, lift1 = plot_lift_curve_from_df(df1, prediction_col, label_col)
    deciles2, lift2 = plot_lift_curve_from_df(df2, prediction_col, label_col)
    plt.plot(deciles1, lift1, label='Lift Curve 1')
    plt.plot(deciles2, lift2, label='Lift Curve 2')
    plt.plot([0,1],[1,1], color="red") # Plot a line with lift equal to 1
    plt.xlabel("Decile")
    plt.ylabel("Lift")
    if title:
        plt.title(title)
    plt.legend()
    plt.show()
    

# COMMAND ----------

# MAGIC %md
# MAGIC #1. Reading the Data

# COMMAND ----------

#Read datasets from the project 

#TRAIN & VALIDATION DATASETS
orders=spark\
.read\
.format("csv")\
.option("header","true")\
.option("inferSchema","true")\
.load("/FileStore/tables/orders.csv")

order_payments=spark\
.read\
.format("csv")\
.option("header","true")\
.option("inferSchema","true")\
.load("/FileStore/tables/order_payments.csv")

order_reviews=spark\
.read\
.format("csv")\
.option("header","true")\
.option("inferSchema","true")\
.load("/FileStore/tables/order_reviews.csv")

order_items=spark\
.read\
.format("csv")\
.option("header","true")\
.option("inferSchema","true")\
.load("/FileStore/tables/order_items.csv")

products=spark\
.read\
.format("csv")\
.option("header","true")\
.option("inferSchema","true")\
.load("/FileStore/tables/products.csv")


#TEST DATASETS

test_orders=spark\
.read\
.format("csv")\
.option("header","true")\
.option("inferSchema","true")\
.load("/FileStore/tables/test_orders.csv")

test_order_payments=spark\
.read\
.format("csv")\
.option("header","true")\
.option("inferSchema","true")\
.load("/FileStore/tables/test_order_payments.csv")

test_order_items=spark\
.read\
.format("csv")\
.option("header","true")\
.option("inferSchema","true")\
.load("/FileStore/tables/test_order_items.csv")

test_products=spark\
.read\
.format("csv")\
.option("header","true")\
.option("inferSchema","true")\
.load("/FileStore/tables/test_products.csv")

# COMMAND ----------

# MAGIC %md
# MAGIC #2. Cleaning the data

# COMMAND ----------

# MAGIC %md
# MAGIC ##Show tables

# COMMAND ----------

#Orders table
show_data(orders, test_orders)

#Order Payments table 
show_data(order_payments, test_order_payments)

#Order Reviews table
print(f"{order_reviews} TABLE")
print(" TRAIN DATASET ")
print(" ")
order_reviews.show(3)

#Order Items table
show_data(order_items, test_order_items)

#Products table
show_data(products, test_products)


# COMMAND ----------

# MAGIC %md
# MAGIC ## Changing to suitable data types (cast)

# COMMAND ----------

#ORDERS TABLE
print("-------------------------------------------------------------ORDERS-----------------------------------------------------------------------------------------")
print("---------------TRAIN--------------")
#Check the table schema (data types to be fix and let's fix them)
orders.printSchema()
#String to timestamp
cols_to_convert = ["order_purchase_timestamp", "order_approved_at", "order_delivered_carrier_date", "order_delivered_customer_date", "order_estimated_delivery_date"]
orders = orders.select([to_timestamp(col(c), "yyyy-MM-dd HH:mm:ss").alias(c) if c in cols_to_convert else c for c in orders.columns])
#was it fix? 
print(orders.printSchema())
print("---------------TEST--------------")
#Check the table schema (data types to be fix and let's fix them)
test_orders.printSchema()
#String to timestamp
cols_to_convert = ["order_purchase_timestamp", "order_approved_at", "order_delivered_carrier_date", "order_delivered_customer_date", "order_estimated_delivery_date"]
test_orders = test_orders.select([to_timestamp(col(c), "yyyy-MM-dd HH:mm:ss").alias(c) if c in cols_to_convert else c for c in test_orders.columns])
#was it fix? 
test_orders.printSchema()

#ORDER PAYMENTS
print("-------------------------------------------------------------ORDER PAYMENTS-----------------------------------------------------------------------------------------")
print("---------------TRAIN--------------")
#Inspecting data types (seems alright!)
order_payments.printSchema()
print("---------------TEST--------------")
#Inspecting data types (seems alright!)
test_order_payments.printSchema()


#ORDER ITEMS
print("-------------------------------------------------------------ORDER ITEMS -----------------------------------------------------------------------------------------")
print("---------------TRAIN--------------")
# Before changing (seems alright!)
order_items.printSchema()
print("---------------TEST--------------")
# Before changing (seems alright!)
test_order_items.printSchema()

#PRODUCTS
print("------------------------------------------------------------- PRODUCTS -----------------------------------------------------------------------------------------")
print("---------------TRAIN--------------")
# Before changing (seems alright!)
products.printSchema()
print("---------------TEST--------------")
# Before changing (seems alright!)
test_products.printSchema()


#ORDER REVIEWS
print("-------------------------------------------------------------ORDER REVIEWS-----------------------------------------------------------------------------------------")
print("---------------TRAIN--------------")
#before changing 
order_reviews.printSchema()
#String to timestamp
order_reviews=order_reviews\
.withColumn("review_answer_timestamp", to_timestamp(col("review_answer_timestamp"),"yyyy-MM-dd HH:mm:ss"))
#after changing
order_reviews.printSchema()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Creating new variables for visualization and modelling

# COMMAND ----------

#ORDERS TABLE - WE NEED ORDER ID GRANULARITY

#Create the new variables in both datasets
orders = create_orders_variables(orders)#Please see the function at the beggining of the code
test_orders = create_orders_variables(test_orders)#Please see the function at the beggining of the code
print("Orders table had new variables created in the TRAIN and TEST datasets! ")

#ORDER PAYMENTS TABLE

#Create the new variables in both datasets
order_payments = create_order_payments_variables(order_payments) #Please see the function at the beggining of the code
# Add a row to replace not_defined with credit_card as it has the highest occurence & value 
#- there are only 3 entries with not defined payment type having 0 as payment values)
test_order_payments = test_order_payments\
.withColumn("payment_type", regexp_replace("payment_type", "not_defined", "credit_card"))
test_order_payments = create_order_payments_variables(test_order_payments)#Please see the function at the beggining of the code

#ORDER ITEMS & PRODUCTS ( Joined together in order to prevent data loss)

items_products=create_items_products_variables(order_items,products)
test_items_products=create_items_products_variables(test_order_items,test_products)

#ORDER REVIEWS

#for this, reviews score will be average for all interactions inside an order_id. 
order_reviews=order_reviews\
.groupby('order_id')\
.agg(mean("review_score").alias('review_score'),
     min('review_creation_date').alias('first_review_date'),
     max('review_creation_date').alias('last_review_date'))\
.withColumn('review_score', floor(col('review_score')))


# COMMAND ----------

# MAGIC %md
# MAGIC ## Checking null values across orders

# COMMAND ----------

#ORDER TABLES
cols_to_analyze=["order_purchase_timestamp", "order_approved_at","order_delivered_carrier_date","order_delivered_customer_date","order_estimated_delivery_date"]
columns_nas=["delivery_duration_days","delivery_delay_days"]
orders = clean_orders_na_values(orders, 
                        col_to_analyze=cols_to_analyze,
                        columns_na=columns_nas)

test_orders = clean_orders_na_values(test_orders, 
                        col_to_analyze=cols_to_analyze,
                        columns_na=columns_nas)

#ORDER PAYMENTS

#Counting nulls across columns (seems alright!)
order_payments.select([count(when(col(c).isNull(), c)).alias(c) for c in order_payments.columns]).show()
test_order_payments.select([count(when(col(c).isNull(), c)).alias(c) for c in test_order_payments.columns]).show()

#ORDER ITEMS AND PRODUCTS
items_products.select([count(when(col(c).isNull(), c)).alias(c) for c in items_products.columns]).show()
test_items_products.select([count(when(col(c).isNull(), c)).alias(c) for c in test_items_products.columns]).show()

#ORDER REVIEWS
#counting nulls across columns (seems alright!)
order_reviews.select([count(when(col(c).isNull(), c)).alias(c) for c in order_reviews.columns]).show() #We won't use review answer timestamp


# COMMAND ----------

# MAGIC %md
# MAGIC ## Cleaned tables (display)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Orders

# COMMAND ----------

print("TRAIN")
orders.display()
print("TEST")
test_orders.display()

# COMMAND ----------

# MAGIC %md
# MAGIC ### Order payments

# COMMAND ----------

print("TRAIN")
order_payments.display()
print("TEST")
test_order_payments.display()

# COMMAND ----------

# MAGIC %md
# MAGIC ### Items & Products

# COMMAND ----------

print("TRAIN")
items_products.display()
print("TEST")
test_items_products.display()

# COMMAND ----------

# MAGIC %md 
# MAGIC ### Reviews

# COMMAND ----------

print("TRAIN")
order_reviews.display()

# COMMAND ----------

# MAGIC %md
# MAGIC ##Merging the tables and creating a basetable

# COMMAND ----------

#TRAIN
train_basetable = orders\
.join(order_payments,["order_id"],"inner")\
.join(order_reviews,["order_id"],"left")\
.withColumn("review_score",when(col('review_score').isNull(),0).otherwise(col("review_score")))\
.join(items_products,["order_id"],"left")

# Creating a new variable for reviews (for visualization purposes and to highlight orders that have no reviews)
train_basetable = train_basetable\
.orderBy('order_purchase_timestamp')\
.withColumn("review_sentiment",when(col("review_score")>3, "Positive").when(col("review_score")==0, "No Review").otherwise("Negative"))\
.withColumn('review_score', col('review_score').cast('double'))

#TEST
test_basetable = test_orders\
.join(test_order_payments,["order_id"],"inner")\
.join(test_items_products,["order_id"],"left")\
.orderBy('order_purchase_timestamp')

print('TRAIN BASETABLE')
train_basetable.display()
#check nulls after joining
train_basetable.select([count(when(col(c).isNull(), c)).alias(c) for c in train_basetable.columns]).display()

print('TEST BASETABLE')
test_basetable.display()
#check nulls after joining
test_basetable.select([count(when(col(c).isNull(), c)).alias(c) for c in test_basetable.columns]).display()

# COMMAND ----------

# MAGIC %md
# MAGIC #3. Model building and predicting

# COMMAND ----------

# MAGIC %md
# MAGIC ### Initial filtering

# COMMAND ----------

#FILTERING COLUMNS - We keep the strings we need to make dummies of and the order id as identifier, plus "review score as our label"

#TRAIN
print('TRAIN')
selected_columns_train = [field for field in train_basetable.schema.fields 
                    if (field.dataType != DateType() 
                        and field.dataType != StringType() 
                        or field.name in ['order_id','review_score', 'status', 'repeat_customer','payment_methods'])]
#save the list.
selected_column_names_train = [field.name for field in selected_columns_train]
print(selected_column_names_train)
#applying the filter
train_basetable_filtered = train_basetable.select(*selected_column_names_train)

#TEST
print('TEST')
selected_columns_test = [field for field in test_basetable.schema.fields 
                    if (field.dataType != DateType() 
                        and field.dataType != StringType() 
                        or field.name in ['order_id', 'status', 'repeat_customer','payment_methods'])]
#save the list.
selected_column_names_test = [field.name for field in selected_columns_test]
print('we only remove review score as this indicator does not exist yet')
#applying the filter
test_basetable_filtered = test_basetable.select(*selected_column_names_test)

# COMMAND ----------

# MAGIC %md 
# MAGIC ### Addresing missing data from joins

# COMMAND ----------

#390 Orders do not have information regarding the order items and of the products. 
#From the nature of columns, store the ones you will need to modify based on the median. (values not following a normal distribution) EXAMPLE: avg_price
cols_to_convert_median = ["in_top_20",
                            "not_in_top_20",
                            "avg_price", 
                            "avg_ship_cost",
                            "avg_prod_name_length", 
                            "avg_prod_desc_length",
                            "avg_photos_qty",
                            "avg_prod_weight_g",
                            "avg_prod_volume_cm3",
                            "avg_ship_cost_per_g"
                           ]
display(train_basetable_filtered.select('avg_price','in_top_20','avg_prod_volume_cm3'))
#We will use the pipeline to replace this values.

# COMMAND ----------

# MAGIC %md 
# MAGIC ### Pre-processing pipeline

# COMMAND ----------

#SOURCE FOR THE PCA AND IMPUTER: https://stackoverflow.com/questions/31774311/pca-analysis-in-pyspark / https://spark.apache.org/docs/latest/api/python/reference/api/pyspark.ml.feature.Imputer.html

#If we need it.
#PCA(k=3, inputCol="std_features", outputCol="pca_features")
# Fit the PCA model on the transformed data
#pca = PCA(k=3, inputCol="std_features", outputCol="pca_features_2")
#pcaModel = pca.fit(transformed_df)
# Get the loadings of each feature on each component
#loadings = pcaModel.pc.toArray().transpose()
# Fit the data through the pre-processing pipeline

pipeline_preprocessing = Pipeline(stages=[
    
    Imputer(inputCols=cols_to_convert_median, outputCols=["{}".format(c) for c in cols_to_convert_median]).setStrategy("median"),
    RFormula(formula="review_score ~ . - order_id -avg_ship_cost_per_g"),
    MaxAbsScaler(inputCol='features',outputCol='std_features')
    
]).fit(train_basetable_filtered)

#transform those datasets to be ready to model! 
basetable_train_ap = pipeline_preprocessing\
.transform(train_basetable_filtered)\
.select(col('order_id'),col('std_features').alias('features'),col('label'))

# COMMAND ----------

# MAGIC %md 
# MAGIC ###Train, Validation and Test

# COMMAND ----------

# calculate the number of rows in the dataframe
num_rows = basetable_train_ap.count()

# calculate the index where the split should occur
split_index = int(num_rows * 0.7)

# create the two dataframes, one with the first 70% of the data and one with the rest (We are trying to search for seasonality of reviews( date importance))
stratified_train = basetable_train_ap.limit(split_index)
stratified_validate = basetable_train_ap.exceptAll(stratified_train)

print(stratified_train, stratified_validate)

# COMMAND ----------

# MAGIC %md
# MAGIC ####Binary Model ( Positive review: 5, 4 / Negative Review: 3, 2, 1)

# COMMAND ----------

#if they don't have a review, let's drop them. (Also keep the binary indicators)
stratified_train_Binary = stratified_train\
.filter(stratified_train["label"] != 0)\
.withColumn("label", when((stratified_train["label"] == 5) |(stratified_train["label"] == 4), 1).otherwise(0))

stratified_validate_Binary = stratified_validate\
.filter(stratified_validate["label"] != 0)\
.withColumn("label", when((stratified_validate["label"] == 5) |(stratified_validate["label"] == 4), 1).otherwise(0))

# COMMAND ----------

# MAGIC %md 
# MAGIC ##### Evaluating in different models! 

# COMMAND ----------

#We create a list for every classifier we want to test. 6 in total
classifiers = [MultilayerPerceptronClassifier(layers=[len(selected_column_names_train)-1,2]), 
               GBTClassifier(seed=134),
               FMClassifier(), 
               RandomForestClassifier(), 
               LogisticRegression(), 
               LinearSVC()]
classifier_names = ["MultilayerPerceptronClassifier",
                    "GBTClassifier",
                    "FMClassifier",
                    "Random Forest", 
                    "Logistic Regression",
                    "LinearSVC"]

#We save a list with the scores to compare and save the best one! 
#scores will be saved base on their Area under ROC curve
model_scores = np.zeros((1, len(classifiers)))
for i, (clf, clf_name) in enumerate(zip(classifiers, classifier_names)):
    cmodel = clf.fit(stratified_train_Binary)
    pred = cmodel.transform(stratified_validate_Binary)
    evaluator1 = BinaryClassificationEvaluator(metricName="areaUnderPR")
    evaluator2 = BinaryClassificationEvaluator()
    model_scores[:, i] = evaluator1.evaluate(pred)
    print(f"{clf_name}: Area under Precision-Recall curve", evaluator1.evaluate(pred))
    print(f"{clf_name}: Area under ROC curve", evaluator2.evaluate(pred))

# COMMAND ----------

# MAGIC %md 
# MAGIC ##### Feature selection through feature importance! 

# COMMAND ----------

## Feature selection through feature importance
max_index = np.argmax(model_scores)
model_name = classifiers[max_index]
#fitting best model. 
best_model=model_name.fit(stratified_train_Binary)

#Saving feature importance
clf_importance_scores = best_model.featureImportances.toArray()

#select the correct names in order for the variables. take into account the dummies created in status
model_columns = [col for col in selected_column_names_train[1:] if col not in ['review_score','avg_ship_cost_per_g']]
status_position = 3
#store the list
lst = model_columns[:status_position] + ["status_1"] + ["status_2"] + model_columns[status_position:]
# Create a dictionary to store the feature names and the mean importance scores
feature_importances = {lst[i]: clf_importance_scores[i] for i in range(len(lst))}

#We will create a dataframe in order to display the feature importances.
sc = SparkContext.getOrCreate()
rdd = sc.parallelize(feature_importances.items())
sorted_rdd = rdd.sortBy(lambda x: x[1], ascending=False)
sorted_dict = dict(sorted_rdd.collect())

#we use pandas for easiness.
df = pd.DataFrame(sorted_dict, index=[0])
df = df.transpose()
df.columns = ['value']
df.reset_index(inplace=True)
df.rename(columns={'index':'key'}, inplace=True)
spark_df = spark.createDataFrame(df)
spark_df.display()

#save the features names (values that have a relative importance greater than 0.05)
top_N_keys = spark_df.where(col('value')>0.05).take(10)
features_selected_train = ['order_id','review_score']+[row.key for row in top_N_keys]
features_selected_test = ['order_id']+[row.key for row in top_N_keys]

# COMMAND ----------

# MAGIC %md
# MAGIC ##### Prepare our final model (After ft)

# COMMAND ----------

#let's see the features saved! including order id and the review score.
features_selected_train

# COMMAND ----------

#filter after feature selection...
train_basetable_ft_filtered= train_basetable_filtered.select(*features_selected_train)
test_basetable_ft_filtered= test_basetable_filtered.select(*features_selected_test)

#cols to put inside the pipeline to replace by it's training median (just the remaining ones)
cols_to_convert_median = ["in_top_20",
                          "avg_prod_desc_length",
                          "avg_prod_weight_g",
                          "avg_ship_cost",
                          "avg_prod_name_length",
                           ]
#Pipeline for the features selected
pipeline_preprocessing_ft = Pipeline(stages=[
    
    Imputer(inputCols=cols_to_convert_median, outputCols=["{}".format(c) for c in cols_to_convert_median]).setStrategy("median"),
    RFormula(formula="review_score ~ . - order_id "),
    MaxAbsScaler(inputCol='features',outputCol='std_features')
    
]).fit(train_basetable_ft_filtered)

#transform train with the reduced features
basetable_train_aft = pipeline_preprocessing_ft\
.transform(train_basetable_ft_filtered)\
.select(col('order_id'),col('std_features').alias('features'),col('label'))

#STRATIFIED SAMPLING: Equally distributed labels for both samples. (Now we do it randomly but keeping the proportions in the labels)
#Trying to correctly assign values rather than random miss-sampling
fractions = {0: 0.7, 1: 0.7,2: 0.7, 3: 0.7,4: 0.7, 5: 0.7}
stratified_train_aft = basetable_train_aft.stat.sampleBy("label", fractions, seed=123)
stratified_validate_aft = basetable_train_aft.subtract(stratified_train_aft)

#if they don't have a review, let's drop them. Again we do the binary indicator
stratified_train_Binary_aft= stratified_train_aft\
.filter(stratified_train_aft["label"] != 0)\
.withColumn("label", when((stratified_train_aft["label"] == 5) |(stratified_train_aft["label"] == 4), 1).otherwise(0))

stratified_validate_Binary_aft = stratified_validate_aft\
.filter(stratified_validate_aft["label"] != 0)\
.withColumn("label", when((stratified_validate_aft["label"] == 5) |(stratified_validate_aft["label"] == 4), 1).otherwise(0))




##HOLDOUT DATA
#Now, we do the process with the features selected ALSO in the HOLDOUT DATA
basetable_test_aft = pipeline_preprocessing_ft\
.transform(test_basetable_ft_filtered)\
.select(col('order_id'),col('std_features').alias('features'))

# COMMAND ----------

# MAGIC %md
# MAGIC ##### Cross validation with best model selected

# COMMAND ----------

print(model_name.explainParams())
#Explanation: 

# COMMAND ----------

fractions_CV = { 1: 0.05,2: 0.05, 3: 0.05, 4: 0.05, 5: 0.05}
stratified_train_Binary_CV = stratified_train_Binary_aft.stat.sampleBy("label", fractions_CV, seed=123)


from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
Binary_CV = CrossValidator(
  estimator = model_name,
  evaluator = BinaryClassificationEvaluator(),
  estimatorParamMaps = ParamGridBuilder().addGrid(gbt.maxDepth, [4,6,8,10,12]).build(),
  numFolds=3
)

#Fitting the model
CVmodel = Binary_CV.fit(stratified_train_Binary_CV)
print('Chosen Model: ',CVmodel.bestModel)

#Please note that model takes too long to run... we tried subsampling the data, even at 5% of original, but since GBT is a black box model, it could run for more than 1h:35 min.
#Best model was the following according to cross-validation GBTClassifier(seed=132, maxDepth=10)

# COMMAND ----------

#DONT RUN THIS LINE
#Fitting the model. For running it faster, we manually inputed the best-model. GBTClassifier(seed=132, maxDepth=10)
CVmodel = GBTClassifier(seed=132, maxDepth=10).fit(stratified_train_Binary_aft)
pred=CVmodel.transform(stratified_validate_Binary_aft)
#Let's evaluate in both Binary evaluators available
print(f"{model_name}: Area under Precision-Recall curve", evaluator1.evaluate(pred))
print(f"{model_name}: Area under ROC curve", evaluator2.evaluate(pred))

# COMMAND ----------

# MAGIC %md 
# MAGIC ##### Checking training v validation

# COMMAND ----------

CVmodel = GBTClassifier(seed=132, maxDepth=10).fit(stratified_train_Binary)# Plot the ROC curve for the validation data and the training data
plot_roc_curve(CVmodel,stratified_train_Binary, stratified_validate_Binary)
#There's overfitting with the train dataset, but results where more accurate with GBT on the validate data according to ROC score!

# COMMAND ----------

#Plotting know the lift curve for our final conclusion! 
a=GBTClassifier(seed=134,maxDepth=10).fit(stratified_train_Binary)
b=a.transform(stratified_train_Binary)
c=a.transform(stratified_validate_Binary)

#explanation about the functions in the first cell of the notebook...

#train
lift_train = b\
.withColumn("probability_column", vector_to_array("probability"))

lift_train=lift_train\
.select(["order_id","label","probability"] + [col("probability_column")[i] for i in range(2)])

lift_train=lift_train\
.withColumn("max",greatest(lift_train["probability_column[0]"],lift_train["probability_column[1]"]))

#test
lift_validate= c\
.withColumn("probability_column", vector_to_array("probability"))

lift_validate=lift_validate\
.select(["order_id","label","probability"] + [col("probability_column")[i] for i in range(2)])

lift_validate=lift_validate\
.withColumn("max",greatest(lift_validate["probability_column[0]"],lift_validate["probability_column[1]"]))

# COMMAND ----------

#functions details at the start of the notebook...
plot_multiple_lift_curves(lift_train, lift_validate, "max", "label", "Lift Curve")

# COMMAND ----------

# MAGIC %md
# MAGIC #####BINARY REVIEW SCORES

# COMMAND ----------

#Final Fit, Train v Test. The final moment.
final_train= basetable_train_aft\
.withColumn("label", when((basetable_train_aft["label"] == 5) |(basetable_train_aft["label"] == 4), 1).otherwise(0))
final_test= basetable_test_aft

fitted=GBTClassifier(seed=132, maxDepth=10).fit(final_train)
pred=fitted.transform(final_test)
pred.select('order_id','prediction').display()

# COMMAND ----------

#check proportions
pred.groupBy('prediction').count().show()
pred.select(col('order_id'),col('prediction').alias('review_score')).write.csv("/FileStore/tables/final_scores.csv")

# COMMAND ----------

# MAGIC %md
# MAGIC #### Multiclass Model (Bonus )

# COMMAND ----------

#if they don't have a review, let's drop them.
stratified_train_Multi = stratified_train\
.filter(stratified_train["label"] != 0)

stratified_validate_Multi = stratified_validate\
.filter(stratified_validate["label"] != 0)

# COMMAND ----------

# MAGIC %md
# MAGIC ##### Trying out different One vs Rest Models

# COMMAND ----------

#We create a list for every classifier we want to test. 5 in total
classifiers = [
               OneVsRest(classifier=MultilayerPerceptronClassifier(layers=[len(selected_column_names_train)-1,2])), 
               OneVsRest(classifier=GBTClassifier(seed=134)),
               OneVsRest(classifier=FMClassifier()),  
               OneVsRest(classifier=LogisticRegression()), 
               ]

#Store the names to be able to visualize them in the print
classifier_names = ["MultilayerPerceptronClassifier",
                    "GBTClassifier",
                    "FMClassifier", 
                    "Logistic Regression",
                    ]

#We save a list with the scores to compare and save the best one! 
#scores will be saved base on their Area under ROC curve and PR (Precision-Recall)

for (clf_1, clf_name_1) in (zip(classifiers, classifier_names)):
    cmodel = clf_1.fit(stratified_train_Multi)
    pred = cmodel.transform(stratified_validate_Multi)
    evaluator1 = MulticlassClassificationEvaluator()
    evaluator2 = MulticlassClassificationEvaluator(metricName='accuracy')
    print(f"{clf_name_1}: f1 score", evaluator1.evaluate(pred))
    print(f"{clf_name_1}: Accuracy", evaluator2.evaluate(pred))

# COMMAND ----------

# MAGIC %md 
# MAGIC ##### After feature importance

# COMMAND ----------

#filter after feature selection...
train_basetable_ft_filtered= train_basetable_filtered.select(*features_selected_train)
test_basetable_ft_filtered= test_basetable_filtered.select(*features_selected_test)

#cols to put inside the pipeline to replace by it's training median (just the remaining ones)
cols_to_convert_median = ["in_top_20",
                          "avg_prod_desc_length",
                          "avg_prod_weight_g",
                          "avg_ship_cost",
                          "avg_prod_name_length",
                           ]
#Pipeline for the features selected
pipeline_preprocessing_ft = Pipeline(stages=[
    
    Imputer(inputCols=cols_to_convert_median, outputCols=["{}".format(c) for c in cols_to_convert_median]).setStrategy("median"),
    RFormula(formula="review_score ~ . - order_id "),
    MaxAbsScaler(inputCol='features',outputCol='std_features')
    
]).fit(train_basetable_ft_filtered)

#transform train with the reduced features
basetable_train_aft = pipeline_preprocessing_ft\
.transform(train_basetable_ft_filtered)\
.select(col('order_id'),col('std_features').alias('features'),col('label'))

#STRATIFIED SAMPLING: Equally distributed labels for both samples. (Now we do it randomly but keeping the proportions in the labels)
#Trying to correctly assign values rather than random miss-sampling
fractions = {0: 0.7, 1: 0.7,2: 0.7, 3: 0.7,4: 0.7, 5: 0.7}
stratified_train_aft = basetable_train_aft.stat.sampleBy("label", fractions, seed=123)
stratified_validate_aft = basetable_train_aft.subtract(stratified_train_aft)

#if they don't have a review, let's drop them. Again we do the binary indicator
stratified_train_Multi_aft= stratified_train_aft\
.filter(stratified_train_aft["label"] != 0)

stratified_validate_Multi_aft = stratified_validate_aft\
.filter(stratified_validate_aft["label"] != 0)

# COMMAND ----------

# MAGIC 
# MAGIC %md 
# MAGIC ##### Printing results of the previous cross validation ( for the first model (Binary))

# COMMAND ----------

#Fitting the model. For running it faster, we manually inputed the best-model. GBTClassifier(seed=132, maxDepth=10)
CVmodel = OneVsRest(classifier=GBTClassifier(seed=132, maxDepth=10)).fit(stratified_train_Multi_aft)
pred=CVmodel.transform(stratified_validate_Multi_aft)
#Let's evaluate in both Binary evaluators available
print(f"{model_name}: F1-score", evaluator1.evaluate(pred))
print(f"{model_name}: Accuracy", evaluator2.evaluate(pred))

# COMMAND ----------

#Confusion Matrix
# Initialize the confusion matrix with zeros
confusion_matrix = [[0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]]

# Calculate the values for each class
for i in range(1, 6):
    for j in range(1, 6):
        confusion_matrix[i-1][j-1] = pred.filter((col("label") == i) & (col("prediction") == j)).count()

# Convert the confusion matrix to a Pandas dataframe
confusion_matrix_df = pd.DataFrame(confusion_matrix, index=[1, 2, 3, 4, 5], columns=[1, 2, 3, 4, 5])

# Plot the confusion matrix using seaborn's heatmap function
sns.heatmap(confusion_matrix_df, annot=True, fmt='d')
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix")
plt.show()

# COMMAND ----------

final_multi_train= basetable_train_aft
final_multi_test= basetable_test_aft

fitted=OneVsRest(classifier=GBTClassifier(seed=132, maxDepth=10)).fit(final_multi_train)
pred_multi=fitted.transform(final_multi_test)
pred_multi.select('order_id','prediction').display()

# COMMAND ----------

# MAGIC %md
# MAGIC ##OUR FINAL RESULTS :)

# COMMAND ----------

pred_binary=spark\
.read\
.format("csv")\
.option("header","false")\
.option("inferSchema","true")\
.load("/FileStore/tables/final_scores.csv")\
.select(col('_c0').alias('order_id'), col('_c1').alias('prediction_Binary'))

Scores_BDT= pred_binary\
.join(pred_multi,['order_id'],'inner')\
.select(col('order_id'),col('prediction_Binary').alias('prediction_Binary(1-Positive/0-Negative)'),col('prediction').alias('prediction_Multiclass/label'))

Scores_BDT.show(5, truncate=False)
Scores_BDT.write.csv("/FileStore/tables/SCORES_BDT_ABDALLAH_KARAKOULIAN_LUCERO.csv")

# COMMAND ----------

# MAGIC %md
# MAGIC #4. Visualizing and deriving insights

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ##intro: Correlation

# COMMAND ----------

#SOURCE: https://stackoverflow.com/questions/39409866/correlation-heatmap/48278285#48278285

#Capture only the selected variables
cm = train_basetable.select(
 'review_score',
 'delivery_delay_days',
 'delivery_duration_days',
 'avg_prod_desc_length',
 'avg_prod_weight_g',
 'avg_prod_name_length',
 'in_top_20',
 'avg_ship_cost')

#Create a vector assembler with all correlation features (handle missing values)

vector_col = "corr_features"
corr_pipeline = Pipeline(stages=[
Imputer(inputCols=['in_top_20','avg_prod_desc_length','avg_prod_weight_g','avg_prod_name_length','avg_ship_cost'],
        outputCols=["{}".format(c) for c in ['in_top_20','avg_prod_desc_length','avg_prod_weight_g','avg_prod_name_length','avg_ship_cost'] ]).setStrategy("median"),
VectorAssembler(inputCols=['review_score','delivery_delay_days','delivery_duration_days','avg_prod_desc_length','avg_prod_weight_g','avg_prod_name_length','in_top_20','avg_ship_cost'], 
                            outputCol=vector_col)
    
                                ]).fit(cm)
corr_vector=corr_pipeline.transform(cm)
matrix = Correlation.corr(corr_vector, vector_col)


corr_values = matrix.collect()[0]["pearson({})".format(vector_col)].values
#re-shape into 2-d array of (8,8)
corr_matrix = np.reshape(corr_values, (8, 8))
# Convert the values to a Pandas dataframe
corr_df = pd.DataFrame(corr_matrix, 
                       columns=['review_score','delivery_delay_days','delivery_duration_days','avg_prod_desc_length','avg_prod_weight_g','avg_prod_name_length','in_top_20','avg_ship_cost'], 
                       index=['review_score','delivery_delay_days','delivery_duration_days','avg_prod_desc_length','avg_prod_weight_g','avg_prod_name_length','in_top_20','avg_ship_cost'])

# Plot the correlation matrix using seaborn's heatmap function
sns.heatmap(corr_df, annot=True)
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC As we can see both variables taking into account the days of the order and the delays are highly correlated. Due to time we didn't group them into a **PCA single factor**. But, a great opportunity to move forward would be related to this type of dimensionality reduction analysis, as well that with weight and shipping cost! 

# COMMAND ----------

# Join predicted review sentiment to the test set

prediction=spark\
.read\
.format("csv")\
.option("header", "false")\
.option("inferSchema", "true")\
.load("/FileStore/tables/final_scores.csv")

test_basetable2 = test_basetable.join(prediction, test_basetable.order_id == prediction._c0, "left")\
                                .drop("_c0")\
                                .withColumn("_c1", regexp_replace("_c1", "0.0", "Negative"))\
                                .withColumn("_c1", regexp_replace("_c1", "1.0", "Positive"))
test_basetable2.display()

# COMMAND ----------

# MAGIC %md
# MAGIC ##4.1. Reviews by Sentiment

# COMMAND ----------

# Check the distribution of the historical reviews 
sentiment_train = train_basetable.groupBy("review_sentiment").count()
sentiment_train.display()

# COMMAND ----------

# Check the distribution of the predicted reviews 
sentiment_test = test_basetable2.groupBy(col("_c1").alias("review_sentiment")).count()
sentiment_test.display()

# COMMAND ----------

# MAGIC %md
# MAGIC Positive review rate is is currently 76% and based on our prediction it is expected to increase to 93% in the upcoming 4 months.

# COMMAND ----------

# MAGIC %md
# MAGIC ##4.2. Reviews by Customer Category

# COMMAND ----------

# Check the share of repeat customers
customer=train_basetable.groupBy("review_sentiment","repeat_customer").count().orderBy(col("review_sentiment").asc(),col("count").desc())
customer_pivot=customer\
.groupby('review_sentiment')\
.pivot('repeat_customer')\
.agg(sum("count").alias('nbr_reviews'))\
.na.fill(0)

customer_pivot.display()

# COMMAND ----------

# MAGIC %md
# MAGIC Despite the high positive review rate, repeat customers account for only ~10%. This share is more or less the same across all review sentiment, which indicates that there is no correlation between positive reviews and customer return.

# COMMAND ----------

# MAGIC %md
# MAGIC ##4.3. Reviews by Order Status

# COMMAND ----------

# Check the distribution of orders by status across the 3 review sentiments
status = train_basetable.groupBy("review_sentiment","status").count().orderBy(col("review_sentiment").asc(),col("count").desc())
status_pivot=status\
.groupby('review_sentiment')\
.pivot('status')\
.agg(sum("count").alias('nbr_reviews'))\
.na.fill(0)

status_pivot.display()

# COMMAND ----------

# MAGIC %md
# MAGIC The majority of orders have been delivered or in the process, which means the delivery status does not have an affect on review sentiments. 

# COMMAND ----------

# MAGIC %md
# MAGIC ##4.4. Reviews by Delivery Duration & Delay

# COMMAND ----------

# Check the correlation between delivery duration and review scores
delivery = train_basetable.where(col('review_score')!='0.0').groupBy('review_score').agg(mean('delivery_duration_days'))
delivery.display()

# COMMAND ----------

# Check the correlation between delivery delay and review scores
delay = train_basetable.where((col('review_score')!='0.0') & (col("delivery_delay_days")>0)).groupBy('review_score').agg(mean('delivery_delay_days'))
delay.display()

# COMMAND ----------

# MAGIC %md
# MAGIC However, the delivery duration and long delays are highly and negatively correlated with how consumers rate an order.

# COMMAND ----------

# MAGIC %md
# MAGIC ##4.5. Reviews by Shipping Cost

# COMMAND ----------

# Check the correlation between shipment cost and review scores
shipping = train_basetable.where(col('review_score')!='0.0').groupBy('review_score').agg(mean('avg_ship_cost'))
shipping.display()

# COMMAND ----------

# Check the correlation between product weight and review scores
weight = train_basetable.where((col('review_score')!='0.0') & (col("avg_prod_weight_g")>0)).groupBy('review_score').agg(mean('avg_prod_weight_g'))
weight.display()

# COMMAND ----------

# MAGIC %md
# MAGIC Another important factor that negatively affected customer reviews is the shipping cost. We can also notice that product weitght is correlated as well, which means shipping costs are based on product weight.

# COMMAND ----------

# MAGIC %md
# MAGIC ##4.6. Reviews by Product Category

# COMMAND ----------

# Check the correlation between product categories and review scores
top20 = train_basetable.where(col('review_score')!='0.0').groupBy('review_score').agg(sum("in_top_20"), sum("not_in_top_20"))
top20.display()

# COMMAND ----------

# MAGIC %md
# MAGIC Finally we can say that orders which include products that are among the Top 20 categories have higher reviews. This is more flargrant in positive reviews as the gap there is larger 

# COMMAND ----------

# MAGIC %md
# MAGIC ##4.7. Reviews Payment Method

# COMMAND ----------

# Check the distribution of orders by number of payment methods used
method = train_basetable\
.groupBy('payment_methods')\
.agg(count('payment_methods'))

method.display()

# COMMAND ----------

# MAGIC %md
# MAGIC Adding some general insights about customers, we can see that the majority uses a single payment method to make a purchase.

# COMMAND ----------

# Check the distribution of orders by payment method
payment = train_basetable\
.agg(sum('credit_card_amount'), 
     sum('debit_card_amount'), 
     sum('mobile_amount'),
     sum('voucher_amount'))

payment.display()

# COMMAND ----------

# MAGIC %md
# MAGIC As for the majorly used payment methods, credit cards account for the lion's share followed by mobile payments. It is worth noting that vouchers account for a minimal share.

# COMMAND ----------

# MAGIC %md
# MAGIC ##4.8. Reviews Product Display

# COMMAND ----------

# Check the distribution of orders by payment method
photo = train_basetable.where((col('review_sentiment')!='no review') & (col("avg_photos_qty")>0)).groupBy('review_score').agg(mean('avg_photos_qty'))
photo.display()

# COMMAND ----------

name = train_basetable.where((col('review_score')!='0') & (col("avg_prod_name_length")>0)).groupBy('review_score').agg(mean('avg_prod_name_length'))
name.display()

# COMMAND ----------

desc = train_basetable.where((col('review_score')!='0') & (col("avg_prod_desc_length")>0)).groupBy('review_score').agg(mean('avg_prod_desc_length'))
desc.display()

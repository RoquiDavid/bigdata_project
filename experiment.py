from pyspark.ml import Pipeline
from pyspark.ml.feature import VectorIndexer
from pyspark.ml.linalg import Vectors
from pyspark.ml.feature import VectorAssembler
from pyspark.sql.types import IntegerType
from pyspark.sql import SparkSession
from pyspark.conf import SparkConf
from pyspark.ml.feature import StringIndexer
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.feature import PCA
from pyspark.sql.functions import monotonically_increasing_id, row_number
from pyspark.sql.window import Window
from pyspark.sql import functions as F
from pyspark.ml.feature import VectorSlicer
from pyspark.ml.feature import UnivariateFeatureSelector
import findspark
import utils
findspark.init()

###############################################################################
#                     SPARK SESSION INITIALIZATION                            #
###############################################################################

#Charge la session spark
spark = (SparkSession.
         builder.
         master('local[*]').
         appName('test').
         config(conf = SparkConf()).
         getOrCreate())

spark.conf.set("spark.sql.repl.eagerEval.enabled",True)


###############################################################################
#                              Data loading                                   #
###############################################################################

#DECOMMENTER SI VOUS VOULEZ POUR GENERER LES FICHIERS TRAIN ET TEST CSV SI 
#JAMAIS VOUS NE LES AVEZ PAS DANS VOTRE REPERTOIR. PENSEZ A LES RENOMER
#RESPECTIVEMENT train.csv ET test.csv. POUR CE RENDU LES FICHIERS SONT 
#NORMALEMENT DEJA GENERER ET PRESENT DANS LE REPERTOIRE

#df = spark.read.csv("data/application_record.csv", header=True, inferSchema= \
#                                                                True,sep=",")
#df_2 = spark.read.csv("data/credit_record.csv", header=True, inferSchema= True\
#                                                       ,sep=",")
       
#df = df.join(df_2,df.ID ==  df_2.ID,"inner").drop(df.ID)
#utils.file_train_test_creator(df,spark)


df_train = spark.read.csv("data/train/train.csv", header=True, inferSchema= \
                                                                True,sep=",")
df_test = spark.read.csv("data/test/test.csv", header=True, inferSchema= True\
                                                       ,sep=",")
    

###############################################################################
#                             Data preparation                                #
###############################################################################
#Resumer statistique du dataset                                                        
#df.summary().show(truncate=False, vertical=True)

#Resumer statistique du dataset                                                                
#df2.summary().show(truncate=False, vertical=True)
#
#for col in df.columns:
#    if("NAME" in col):
#        df.select(col).distinct().show()
        


df_vector_train,df_vector_test,label_train,label_test = utils.\
                                            train_test_split(df_train, df_test)
###############################################################################
#                               Feature selection                             #
###############################################################################

#Vector slicer
df_vec_slice_train = utils.vec_slicer(df_vector_train)
df_vec_slice_test = utils.vec_slicer(df_vector_test)
#PCA
df_pca_train = utils.pca_feature_selection(df_vector_train, label_train,3)
df_pca_test = utils.pca_feature_selection(df_vector_test, label_test,3)
#Univariate
df_uni_train = utils.univariate(df_vector_train)
df_uni_test = utils.univariate(df_vector_test)
###############################################################################
#                               Classification                                #
###############################################################################

#Gradient boosting classification
print("\n GRADIENT SLICE")
utils.gradient_boosting_analysis(df_vec_slice_train,df_vec_slice_test, 2, 10)
print("\n GRADIENT PCA")
utils.gradient_boosting_analysis(df_pca_train,df_pca_test, 2, 10)
print("\n GRADIENT UNI")
utils.gradient_boosting_analysis(df_uni_train,df_uni_test, 2, 10)


#Random forest
print("\n RF SLICE")
utils.random_forest_analysis(df_vec_slice_train,df_vec_slice_test)
print("\n RF PCA")
utils.random_forest_analysis(df_pca_train,df_pca_test)
print("\n RF UNI")
utils.random_forest_analysis(df_uni_train,df_pca_test)

#MLP 

print("\n MLP SCLICE")
utils.mlp_analysis(df_vec_slice_train,df_vec_slice_test, 1)
print("\n MLP PCA")
utils.mlp_analysis(df_pca_train,df_pca_test, 3)
print("\n MLP UNI")
utils.mlp_analysis(df_uni_train,df_uni_test, 1)

spark.stop()

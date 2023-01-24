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
from pyspark.ml.feature import Normalizer
from pyspark.ml.feature import VectorSlicer
from pyspark.ml.feature import UnivariateFeatureSelector
from pyspark.ml.classification import GBTClassifier
from pyspark.ml.feature import StringIndexer, VectorIndexer
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.mllib.evaluation import MulticlassMetrics
from pyspark.sql.types import FloatType
from pyspark.ml.classification import MultilayerPerceptronClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
import utils
def factorisation(df):
    string_col = [c for c, t in df.dtypes if t.startswith('string')]
    for colunm in string_col:
        print(colunm," ",(df.count(), len(df.columns)))
        indexer = StringIndexer(inputCol=colunm, 
                                outputCol=colunm+'_facto').fit(df)
    
        df = indexer.setHandleInvalid("skip").transform(df)
        df = df.withColumn(colunm,df[colunm].cast(IntegerType()))
    #Supprime les anciennes colonnes ayant le type string
    df = df.drop(*string_col)
    df.printSchema()
    return df

#Fonction pour normaliser un dataframe pyspark
def normalize(df_vector):
    
    #Initilisation du normaliser
    normalizer = Normalizer(inputCol="features", outputCol="normFeatures",\
                            p=1.0)
    #Application de la normalisation
    l1NormData = normalizer.transform(df_vector)
    print("Normalized using L^1 norm")
    df_vector = l1NormData
    
    #Renomage des colonne
    df_vector = df_vector.withColumnRenamed("features","oldfeatures")
    df_vector = df_vector.withColumnRenamed("normFeatures","features")
    df_vector = df_vector.drop("oldfeatures")
    df_vector.show()
    return df_vector


#Fonction pour appliquer un vector slicer (feature selection)
def vec_slicer(df_vector):
    #Initialisation du slicer
    slicer = VectorSlicer(inputCol="features", outputCol="vec_features",\
                          indices=[1])

    #Application du slicer
    df_vec_slice = slicer.transform(df_vector)

    #Renommage des colonnes
    df_vec_slice.select("features", "vec_features").show()
    df_vec_slice = df_vec_slice.drop("features")
    df_vec_slice = df_vec_slice.withColumnRenamed("vec_features","features")
    df_vec_slice.show()
    return df_vec_slice


#Fonction pour appliquer un PRINCIPAL COMPONENTS ANALYSIS (PCA)
def pca_feature_selection(df_vector, label, nb_compo):
    
    #Initialisation du PCA
    pca = PCA(k=nb_compo, inputCol="features", outputCol="pcaFeatures")
    #Application du PCA
    model = pca.fit(df_vector)

    #Renommage des colonnes
    result = model.transform(df_vector).select("pcaFeatures")
    result.show(truncate=False)
    df_pca = result
    df_pca = df_pca.withColumnRenamed("pcaFeatures","features")

    #Creation de la colonne row_index afin de pouvoir faire la jointure
    #du resultat de PCA avec les label du dataframe originel
    df_pca=df_pca.withColumn('row_index', row_number().over(Window.orderBy(\
                                               monotonically_increasing_id())))
    label=label.withColumn('row_index', row_number().over(Window.orderBy(\
                                               monotonically_increasing_id())))
        
    #Jointure entre pca et les label du dataframe originel
    df_pca = df_pca.join(label, on=["row_index"]).drop("row_index")

    df_pca.show()
    return df_pca

#Fonction pour appliquer une feature selection univarier
def univariate(df_vector):
    #Initialisation du selecteur
    selector = UnivariateFeatureSelector(featuresCol="features", \
                                         outputCol="selectedFeatures",\
                                         labelCol="label", \
                                         selectionMode="numTopFeatures")
    selector.setFeatureType("continuous").setLabelType("categorical").\
                                              setSelectionThreshold(1)

    #Application du selector
    result = selector.fit(df_vector).transform(df_vector)
    df_uni = result
    df_uni = df_uni.drop("features")
    #Renommage des colonnes
    df_uni = df_uni.withColumnRenamed("selectedFeatures","features")

    print("UnivariateFeatureSelector output with top %d features selected \
          using f_classif"
          % selector.getSelectionThreshold())
    df_uni.show()
    return df_uni

#Fonction pour appliquer une classification par gradient boostring 
def gradient_boosting_analysis(df_vector_train,df_vector_test,nb_cate, ite):
    
    print("\n\n Gradient boostring classification")
    print("Training Dataset Count: " + str(df_vector_train.count()))
    print("Test Dataset Count: " + str(df_vector_test.count()))  
    #Encode la colonne des labels
    labelIndexer = StringIndexer(inputCol="label", outputCol="indexedLabel").\
                                                        fit(df_vector_train)
    #Factorisation des feature categoriel si <4 sinon elle sont traitees 
    #comme features continues et non discrete
    featureIndexer =\
        VectorIndexer(inputCol="features", outputCol="indexedFeatures",\
                      maxCategories=nb_cate).fit(df_vector_train)

    #Entrainement du model
    gbt = GBTClassifier(labelCol="indexedLabel", featuresCol="indexedFeatures"\
                        , maxIter=ite)
    
    #Application d'une pipeline pour appliquer le gradient boosting
    #au label et feature index
    pipeline = Pipeline(stages=[labelIndexer, featureIndexer, gbt])
    
    #Entrainement de la pipepline
    model = pipeline.fit(df_vector_train)
    
    #Rendu des predictions
    predictions = model.transform(df_vector_test)
    
    # Affichage du resultats
    predictions.select("prediction", "indexedLabel", "features").show(5)
    
    #Precision du model
    evaluator = MulticlassClassificationEvaluator(
        labelCol="indexedLabel", predictionCol="prediction", metricName=\
            "accuracy")
    accuracy = evaluator.evaluate(predictions)
    print("Test Error = %g" % (1.0 - accuracy))
    
    gbtModel = model.stages[2]
    print(gbtModel)
    
#Fonction pour appliquer une forêt aleatoire 
def random_forest_analysis(df_vector_train,df_vector_test):
        
    #Random forest
    print("\n\n Random forest")
    print("Training Dataset Count: " + str(df_vector_train.count()))
    print("Test Dataset Count: " + str(df_vector_test.count()))   
    
    #Initialisation de la foret
    rf = RandomForestClassifier(featuresCol = 'features', labelCol = 'label')
    #Entrainement de la fret
    rfModel = rf.fit(df_vector_train)
    #Recuperation des peredictions de la foret
    predictions = rfModel.transform(df_vector_test)
    
    #Evaluation du model
    evaluator = MulticlassClassificationEvaluator(labelCol="label",\
                                                  predictionCol="prediction")
    accuracy = evaluator.evaluate(predictions)
    print("Accuracy = %s" % (accuracy))
    print("Test Error = %s" % (1.0 - accuracy))
    
    #Matrice de fonfusion
    preds_and_labels = predictions.select(['prediction','label']).\
                       withColumn('label', F.col('label').cast(FloatType())).\
                       orderBy('prediction')
    preds_and_labels = preds_and_labels.select(['prediction','label'])
    metrics = MulticlassMetrics(preds_and_labels.rdd.map(tuple))
    print(metrics.confusionMatrix().toArray())
    
#Fonction pour appliquer un Multi Layer Perceptron(MLP)
def mlp_analysis(df_vector_train,df_vector_test,nb_unit_first_layer):
    
    print("\n\n MLP")
    print("Training Dataset Count: " + str(df_vector_train.count()))
    print("Test Dataset Count: " + str(df_vector_test.count()))  
    #Initialisation du nombre neuronnes et de couches
    layers = [nb_unit_first_layer,5,5,3]
    
    #Initialisation du reseau
    mlp = MultilayerPerceptronClassifier(layers = layers, seed = 4)
    
    #Entrainement du model
    mlp_model = mlp.fit(df_vector_train)
    
    #Recuperation de la prediction du model
    pred_df = mlp_model.transform(df_vector_test)
    
    #Evaluation du model
    evaluator = MulticlassClassificationEvaluator(labelCol = 'label',\
                                                  predictionCol = 'prediction'\
                                                 , metricName = 'accuracy')
    mlpacc = evaluator.evaluate(pred_df)
    print(mlpacc)
    
#Fonction permettant la creation des fichier train.csv et test.csv
def file_train_test_creator(df, spark):
    #Initialisation du pourcentage de train et de test
    low = round(df.count()*0.8)
    hight = df.count()
    #Creation du datafram train
    train = spark.createDataFrame(df.collect()[0:low],schema = df.schema)
    train.printSchema()
    print("train count: ",train.count())
    #Sauvegarde le dataset train dans un fichier csv
    train.coalesce(1).write.format("com.databricks.spark.csv").option("header"\
                                                   , "true").save("data/train")

    #Creatoin du dataset test
    test = spark.createDataFrame(df.collect()[low+1:hight],schema = df.schema)
    test.printSchema()
    print(test.count())
    #Sauvegarde du dataset test dans un fichier csv
    test.coalesce(1).write.format("com.databricks.spark.csv").option("header"\
                                                    , "true").save("data/test")
    
    
#Fonction permettant d effectuer la preparation des données et le split
#train test
def train_test_split(df_train,df_test):
    
    print("df_train=",(df_train.count(), len(df_train.columns)))
    print("df_test=",(df_test.count(), len(df_test.columns)))
    #Jointure via l'id des deux dataset
    nan_thesh = round(len(df_train.columns)*0.8)
    print(nan_thesh)
    #Drop les colonnes avec plus de 40% de nan/null
    df_train = df_train.na.drop(thresh=nan_thesh)
    df_test = df_test.na.drop(thresh=nan_thesh)

    #Factorise les colonnes string
    df_train = utils.factorisation(df_train)
    df_test = utils.factorisation(df_test)
    #Creation de la colonne label
    df_train = df_train.withColumn(
        'label',
        F.when((F.col("STATUS_facto") > 1) , 0).otherwise(1))
    df_train.printSchema()

    df_train.groupBy("label").count().show()  

    df_test = df_test.withColumn(
        'label',
        F.when((F.col("STATUS_facto") > 1) , 0).otherwise(1))
    df_test.printSchema()

    df_test.groupBy("label").count().show()  

    # Création du vecteur
    vector_col = "features"
    assembler = VectorAssembler(inputCols=df_train.drop("ID").columns, \
                                outputCol=vector_col)
    df_vector_train = assembler.setHandleInvalid("skip").\
        transform(df_train.drop("ID")).select([vector_col,"label"])

    # Création du vecteur
    assembler = VectorAssembler(inputCols=df_test.drop("ID").columns, \
                                outputCol=vector_col)
    df_vector_test = assembler.setHandleInvalid("skip").transform(df_test.\
                                       drop("ID")).select([vector_col,"label"])

    df_vector_test.show()

    #Normalisation L1
    df_vector_train = utils.normalize(df_vector_train)
    label_train = df_vector_train.select("label")

    #Normalisation L1
    df_vector_test = utils.normalize(df_vector_test)
    label_test = df_vector_test.select("label")
    
    return df_vector_train,df_vector_test,label_train,label_test
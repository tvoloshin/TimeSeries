package org.example.timeseries

import org.apache.spark.sql.SparkSession
import org.apache.spark.ml.linalg.Vector
import org.deeplearning4j.nn.conf.NeuralNetConfiguration
import org.deeplearning4j.nn.conf.layers.{DenseLayer, OutputLayer}
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork
import org.deeplearning4j.nn.weights.WeightInit
import org.example.timeseries.models.{Dl4jModel, WrappedModel}
import org.example.timeseries.utils.FeatureUtils.splitTrainTest
import org.example.timeseries.utils.ResultsDisplay
import org.example.timeseries.utils.ResultsDisplay.plotTS
import org.nd4j.linalg.activations.Activation
import org.nd4j.linalg.dataset.DataSet
import org.nd4j.linalg.factory.Nd4j
import org.nd4j.linalg.learning.config.Sgd
import org.nd4j.linalg.lossfunctions.LossFunctions

object Dl4jFNNExample extends App {
  val spark = SparkSession
    .builder()
    .appName("TimeSeries")
    .config("spark.master", "local[*]")
    .getOrCreate()
  spark.sparkContext.setLogLevel("ERROR")

  val numFeatures = 100

  val data = spark.read.option("inferSchema", value = true).option("header", value = true).csv("./data/jena_climate/pressure_cut.csv")
  val Array(train, test) = splitTrainTest(data, 120000, numFeatures)

  plotTS("train", train.select("value").collect().map(_.getDouble(0)))
  plotTS("test", test.select("value").collect().map(_.getDouble(0)))

  val model = new NeuralNetConfiguration.Builder()
    .seed(42)
    .weightInit(WeightInit.XAVIER)
    .updater(new Sgd(0.00000001))
    .list()
    .layer(new DenseLayer.Builder()
      .activation(Activation.RELU).nIn(numFeatures).nOut(200).build())
    .layer(new DenseLayer.Builder()
      .activation(Activation.RELU).nIn(200).nOut(200).build())
    .layer(new DenseLayer.Builder()
      .activation(Activation.RELU).nIn(200).nOut(200).build())
    .layer(new OutputLayer.Builder(LossFunctions.LossFunction.MSE)
      .activation(Activation.IDENTITY).nIn(200).nOut(1).build())
    .build()

  val net = new MultiLayerNetwork(model)
  net.init()

  val ds = new DataSet(
    Nd4j.create(train.select("features").collect().map(_.getAs[Vector](0).toArray)),
    Nd4j.create(train.select("value").collect().map(r => Array(r.getDouble(0))))
  )

  for (i <- 1 to 50) {
    net.fit(ds)
    println(s"Epoch #$i: ${net.score(ds)}")
  }

  val resultsDisplay = new ResultsDisplay(test)

  val dl4jModel = new WrappedModel()
    .setInitVector(train, numFeatures)
    .setModels(new Dl4jModel(net))
    .setNames("prediction")

  var dl4jPredictions = dl4jModel.predict(test)
  resultsDisplay.showResults("Long predictions", dl4jPredictions)

  dl4jPredictions = dl4jModel.predict(test, "value")
  resultsDisplay.showResults("Using true values", dl4jPredictions)

}

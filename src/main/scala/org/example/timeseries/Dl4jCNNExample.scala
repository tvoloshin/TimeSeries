package org.example.timeseries

import org.apache.spark.sql.SparkSession
import org.apache.spark.ml.linalg.Vector
import org.deeplearning4j.nn.conf.NeuralNetConfiguration
import org.deeplearning4j.nn.conf.inputs.InputType
import org.deeplearning4j.nn.conf.layers.{ConvolutionLayer, DenseLayer, OutputLayer, SubsamplingLayer}
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork
import org.deeplearning4j.nn.weights.WeightInit
import org.example.timeseries.models.{Dl4jModel, WrappedModel}
import org.example.timeseries.utils.FeatureUtils.splitTrainTest
import org.example.timeseries.utils.ResultsDisplay
import org.nd4j.linalg.activations.Activation
import org.nd4j.linalg.dataset.DataSet
import org.nd4j.linalg.factory.Nd4j
import org.nd4j.linalg.learning.config.Adam
import org.nd4j.linalg.lossfunctions.LossFunctions

object Dl4jCNNExample extends App {

  val spark = SparkSession
    .builder()
    .appName("TimeSeries")
    .config("spark.master", "local[*]")
    .getOrCreate()
  spark.sparkContext.setLogLevel("ERROR")

  val numFeatures = 30

  val data = spark.read.option("inferSchema", value = true).option("header", value = true).csv("./data/sales/normalized.csv")
  val Array(train, test) = splitTrainTest(data, 850, numFeatures)

  val channels = 1
  val HEIGHT = 1
  val WIDTH = numFeatures
  val N_OUTCOMES = 1
  val conf = new NeuralNetConfiguration.Builder()
    .updater(new Adam(0.001, 0.9, 0.999, 1.0E-7))
    .weightInit(WeightInit.XAVIER_UNIFORM)
    .list()
    .layer(new ConvolutionLayer.Builder(1, 2)
      .stride(1, 1).nOut(128).activation(Activation.RELU).build())
    .layer(new ConvolutionLayer.Builder(1, 2)
      .stride(1, 1).nOut(128).activation(Activation.RELU).build())
    .layer(new ConvolutionLayer.Builder(1, 2)
      .stride(1, 1).nOut(128).activation(Activation.RELU).build())
    .layer(new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX)
      .kernelSize(1, 2).stride(1, 2).build())
    .layer(new DenseLayer.Builder().activation(Activation.RELU)
      .nOut(100).build())
    .layer(new OutputLayer.Builder(LossFunctions.LossFunction.MSE)
      .nOut(N_OUTCOMES).activation(Activation.IDENTITY).build())
    .setInputType(InputType.convolutionalFlat(HEIGHT, WIDTH, channels))
    .build()

  val net = new MultiLayerNetwork(conf)
  net.init()
  println(net.summary())

  val ds = new DataSet(
    Nd4j.create(train.select("features").collect().map(r => r.getAs[Vector](0).toArray)),
    Nd4j.create(train.select("value").collect().map(r => Array(r.getDouble(0))))
  )

  for (i <- 1 to 200) {
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

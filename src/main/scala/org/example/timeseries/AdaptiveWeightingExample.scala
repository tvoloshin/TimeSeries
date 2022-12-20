package org.example.timeseries

import org.apache.spark.ml.regression.{GBTRegressor, RandomForestRegressor}
import org.apache.spark.sql.SparkSession
import org.example.timeseries.models.{AdaptiveWeighting, VotingModel}
import org.example.timeseries.models.WrappedModel
import org.example.timeseries.utils.FeatureUtils.splitTrainTest
import org.example.timeseries.utils.ResultsDisplay

object AdaptiveWeightingExample extends App {

  val winSize = 15

  val spark = SparkSession
    .builder()
    .appName("TimeSeries")
    .config("spark.master", "local[*]")
    .getOrCreate()
  spark.sparkContext.setLogLevel("ERROR")

  val data = spark.read.option("inferSchema", value = true).option("header", value = true).csv("./data/kafka_metrics.csv")

  val trainSize = 100
  val Array(train, test) = splitTrainTest(data, trainSize, winSize)

  val resultsDisplay = new ResultsDisplay(test)

  val wrappedModel = new WrappedModel()
    .setInitVector(train, winSize)
    .setNames("prediction")

  val gbtRegressor = new GBTRegressor()
    .setLabelCol("value")
    .setFeaturesCol("features")
  val gbtModel = gbtRegressor.fit(train)
  resultsDisplay.showResults("GBT", wrappedModel.setModels(gbtModel))

  val rfRegressor = new RandomForestRegressor()
    .setLabelCol("value")
    .setFeaturesCol("features")
  val rfModel = rfRegressor.fit(train)
  resultsDisplay.showResults("RandomForest", wrappedModel.setModels(rfModel))

  val votingModel = new VotingModel(Array(gbtModel, rfModel)).setWeights(0.5, 0.5)
  resultsDisplay.showResults("Voting", wrappedModel.setModels(votingModel))

  var adaptiveWeighting = new AdaptiveWeighting(gbtModel, rfModel)
    .setInitVector(train, winSize)
    .setPredictionInterval(10)
  resultsDisplay.showResults("Adaptive model - default init", adaptiveWeighting)

  adaptiveWeighting = new AdaptiveWeighting(gbtModel, rfModel)
    .setInitVector(train, winSize)
    .setPredictionInterval(10)
    .setInitWeights(Seq(0.3, 0.7))
  resultsDisplay.showResults("Adaptive model - manual init", adaptiveWeighting)

  adaptiveWeighting = AdaptiveWeighting.fromRegressors(train, 10, gbtRegressor, rfRegressor)
  resultsDisplay.showResults("Adaptive model - init from regressors", adaptiveWeighting)

}

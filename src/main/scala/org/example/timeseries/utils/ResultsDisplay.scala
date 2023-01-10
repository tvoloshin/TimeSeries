package org.example.timeseries.utils

import breeze.plot.{Figure, plot}
import org.apache.spark.ml.evaluation.RegressionEvaluator
import org.apache.spark.sql
import org.apache.spark.sql.SparkSession
import org.example.timeseries.models.MyModel
import org.example.timeseries.utils.ResultsDisplay.plotTS

class ResultsDisplay(test: sql.DataFrame) {

  private val spark = SparkSession.builder.getOrCreate()
  import spark.implicits._

  private val valueArr = test.orderBy("t").select("value").map(_.getDouble(0)).collect()

  private val evaluator = new RegressionEvaluator()
    .setLabelCol("value")
    .setPredictionCol("prediction")

  def showMetrics(name: String, predictions: sql.DataFrame): Unit = {
    val rmse = evaluator.setMetricName("rmse").evaluate(predictions)
    val mae = evaluator.setMetricName("mae").evaluate(predictions)
    println(s"$name\n\tRMSE = $rmse; MAE = $mae")
  }

  def showPlot(name:String, predictions: sql.DataFrame): Unit = {
    val predictArr = predictions.orderBy("t").select("prediction").map(_.getDouble(0)).collect()
//    plotTS(name, valueArr, predictArr)
//    plotTS(name, ("true", valueArr), ("predicted", predictArr))
    new PlotUtils(name).addPlots(valueArr, predictArr).addLegend("true", "predicted").create()
  }

  def showResults(name:String, model: MyModel): Unit = {
    val predictions = model.predict(test)
    showMetrics(name, predictions)
    showPlot(name, predictions)
  }

  def showResults(name: String, predictions: sql.DataFrame): Unit = {
    showMetrics(name, predictions)
    showPlot(name, predictions)
  }

}

object ResultsDisplay {

  def plotTS(name: String, valArrays: Array[Double]*): Unit = {
//    val f = Figure(name)
//    val p = f.subplot(0)
//    for (arr <- valArrays) p += plot(arr.indices.map(_.toDouble), arr)
    new PlotUtils(name).addPlots(valArrays: _*).create()
  }

//  def plotTS(name: String, namedArrays: (String, Array[Double])*): Unit = {
//    val f = Figure(name)
//    val p = f.subplot(0)
//    p.legend = true
//    for ((name, arr) <- namedArrays) p += plot(arr.indices.map(_.toDouble), arr, name=name)
//  }

}

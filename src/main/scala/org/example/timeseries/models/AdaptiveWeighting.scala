package org.example.timeseries.models

import org.apache.spark.ml.evaluation.RegressionEvaluator
import org.apache.spark.ml.linalg.Vector
import org.apache.spark.ml.regression.{RegressionModel, Regressor}
import org.apache.spark.sql
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.functions._
import org.example.timeseries.models.AdaptiveWeighting.countWeights

class AdaptiveWeighting(models: RegressionModel[Vector, _ <: RegressionModel[Vector, _]]*) extends MyModel {

  private val spark = SparkSession.builder.getOrCreate()
  import spark.implicits._

  private var predictionInterval: Int = _
  private var initVector: Vector = _
  private var initWeights: Option[Seq[Double]] = None

  private val votingModel = new VotingModel(models.toArray)//.setWeights(0.5, 0.5)
  private val wrappedModel = new WrappedModel().setModels(votingModel +: models: _*)
  private val evaluator = new RegressionEvaluator().setLabelCol("value").setMetricName("rmse")

  def setPredictionInterval(interval: Int): this.type = {
    predictionInterval = interval
    this
  }

  def setInitVector(data: sql.DataFrame, size: Int): this.type = {
    wrappedModel.setInitVector(data, size)
    initVector = wrappedModel.getInitVector
    this
  }

  def setInitWeights(weights: Seq[Double]): this.type = {
    initWeights = Some(weights)
    this
  }

  override def predict(test: sql.DataFrame): sql.DataFrame = {
    votingModel.setWeights(initWeights.getOrElse(Seq.fill(models.size) {1.0 / models.size}): _*)
    wrappedModel.setInitVector(initVector).setModels(votingModel +: models: _*)

    val modelNames = for (i <- 1 until wrappedModel.getNames.length) yield wrappedModel.getNames(i)

    val num_intervals = test.count() / predictionInterval
    var withPredictions = test.select(Array($"t", $"value") ++ ("prediction" +: modelNames).map(lit(null).cast("double").as(_)): _*)

    for (i <- 0 to num_intervals.toInt) {
      val newTest = test.except(test.orderBy("t").limit(i * predictionInterval))
      withPredictions = withPredictions.limit(i * predictionInterval).union(wrappedModel.predictInterval(newTest, predictionInterval))

      wrappedModel.updateVector(newTest.orderBy("t").limit(predictionInterval))

      val errors = modelNames.map(evaluator.setPredictionCol(_).evaluate(withPredictions.limit((i + 1) * predictionInterval)))
      val weights = countWeights(errors)
      wrappedModel.setModels(votingModel.setWeights(weights: _*) +: models: _*)
    }
    withPredictions
  }
}


object AdaptiveWeighting {

  private def countWeights(errors: Seq[Double]): Seq[Double] =
    errors.map(e => (1 / e) / errors.map(1 / _).sum)

  def fromRegressors(data: sql.DataFrame, predictionInterval: Int,
                     regressors: Regressor[Vector, _, _ <: RegressionModel[Vector, _ <: RegressionModel[Vector, _]]]*): AdaptiveWeighting = {

    val vectorSize = data.head().getAs[Vector]("features").size

    val limTrain = data.orderBy("t").limit(data.count().toInt - predictionInterval)
    val limTest = data.except(limTrain)

    val wrappedModel = new WrappedModel()
      .setInitVector(limTrain, vectorSize)
      .setNames("prediction")

    val evaluator = new RegressionEvaluator()
      .setLabelCol("value")
      .setPredictionCol("prediction")
      .setMetricName("rmse")

    val errors = for (regressor <- regressors) yield {
      evaluator.evaluate(
        wrappedModel
          .setModels(regressor.fit(limTrain))
          .predict(limTest)
      )
    }
    val weights = countWeights(errors)

    new AdaptiveWeighting(regressors.map(_.fit(data)): _*)
      .setPredictionInterval(predictionInterval)
      .setInitVector(data, vectorSize)
      .setInitWeights(weights)
  }

}

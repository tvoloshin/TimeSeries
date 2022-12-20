package org.example.timeseries.models

import org.apache.spark.ml.linalg.Vector
import org.apache.spark.ml.param.ParamMap
import org.apache.spark.ml.regression.RegressionModel

class VotingModel(models: Array[_ <: RegressionModel[Vector, _]]) extends RegressionModel[Vector, VotingModel] {

  private var weights: Array[Double] = _

  def setWeights(newWeights: Double*): this.type = {
    weights = newWeights.toArray
    this
  }

  def getWeights: Array[Double] = weights

  override val uid: String = "VotingModel"

  override def copy(extra: ParamMap): Nothing = ???

  override def predict(features: Vector): Double = {
    var res = 0D
    for ((model, weight) <- models zip weights) {
      res += model.predict(features) * weight
    }
    res
  }

}

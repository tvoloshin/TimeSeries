package org.example.timeseries.models

import org.apache.spark.ml.linalg.Vector
import org.apache.spark.ml.param.ParamMap
import org.apache.spark.ml.regression.RegressionModel
import org.apache.spark.mllib.linalg.Vectors
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork
import org.deeplearning4j.spark.util.MLLibUtil

class Dl4jModel(network: MultiLayerNetwork) extends RegressionModel[Vector, Dl4jModel] {

  override val uid: String = "Dl4jModel"

  override def copy(extra: ParamMap): Nothing = ???

  override def predict(features: Vector): Double =
    network.output(MLLibUtil.toVector(Vectors.fromML(features))).getDouble(0L)

}

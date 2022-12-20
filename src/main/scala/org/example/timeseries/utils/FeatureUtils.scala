package org.example.timeseries.utils

import org.apache.spark.ml.functions.array_to_vector
import org.apache.spark.sql
import org.apache.spark.sql.expressions.Window
import org.apache.spark.sql.functions._

object FeatureUtils {

  def withFeatures(data: sql.DataFrame, winSize: Int): sql.DataFrame = {
    val w = Window.orderBy("t").rowsBetween(-winSize, -1)
    data.withColumn("features", array_to_vector(collect_list("value") over w))
  }

  def splitTrainTest(data: sql.DataFrame, trainSize: Int, winSize: Int): Array[sql.DataFrame] = {
    val train = withFeatures(data, winSize).where(s"t >= $winSize").where(s"t < ${trainSize + winSize}")
    val test = data.where(s"t >= ${trainSize + winSize}")
    Array(train, test)
  }

}

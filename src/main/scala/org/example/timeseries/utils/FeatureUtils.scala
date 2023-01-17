package org.example.timeseries.utils

import org.apache.spark.ml.functions.array_to_vector
import org.apache.spark.sql
import org.apache.spark.sql.expressions.Window
import org.apache.spark.sql.functions
import org.apache.spark.sql.functions._

object FeatureUtils {

  def withFeatures(data: sql.DataFrame,
                   winSize: Int,
                   orderCol: String = "t",
                   valueCol: String = "value",
                   featuresCol: String = "features",
                  ): sql.DataFrame = {
    val w = Window.orderBy(orderCol).rowsBetween(-winSize, -1)
    data.withColumn(featuresCol, array_to_vector(collect_list(valueCol) over w))
  }

  def splitData(data: sql.DataFrame, numRows: Int, orderCol: String = "t"): Array[sql.DataFrame] = {
    val withID = data.withColumn("temp_id", functions.row_number().over(Window.orderBy(orderCol)))
    val firstDF = withID.where(s"temp_id <= $numRows").drop("temp_id")
    val secondDF = withID.where(s"temp_id > $numRows").drop("temp_id")
    Array(firstDF, secondDF)
  }

  def splitTrainTest(data: sql.DataFrame, trainSize: Int, winSize: Int, orderCol: String = "t"): Array[sql.DataFrame] = {
    val splitted = splitData(data, trainSize + winSize, orderCol)
    Array(splitData(withFeatures(splitted(0), winSize, orderCol), winSize, orderCol)(1), splitted(1))
  }

}

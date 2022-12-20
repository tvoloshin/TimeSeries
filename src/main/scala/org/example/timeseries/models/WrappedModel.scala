package org.example.timeseries.models

import org.apache.spark.ml.linalg.{Vector, Vectors}
import org.apache.spark.ml.regression.RegressionModel
import org.apache.spark.sql
import org.apache.spark.sql.functions.lit
import org.apache.spark.sql.types.{DoubleType, IntegerType, StructField, StructType}
import org.apache.spark.sql.{Row, SparkSession}

class WrappedModel extends MyModel with Serializable {

  private val spark = SparkSession.builder.getOrCreate()
  import spark.implicits._

  private var testVectorInit: Vector = _
  private var testVector: Vector = _

  private var models: Array[_ <: RegressionModel[Vector, _]] = _
  private var colNames: Array[String] = Array()

  def setInitVector(vector: Vector): this.type = {
    testVectorInit = vector
    this
  }

  def setInitVector(data:sql.DataFrame, size: Int): this.type = {
    testVectorInit = Vectors.dense(data.orderBy($"t".desc).select("value").limit(size).map(_.getDouble(0)).collect().reverse)
    this
  }

  def updateVector(data: sql.DataFrame): this.type = {
    val newValuesCount = testVectorInit.size min data.count().toInt
    val oldValuesCount = testVectorInit.size - newValuesCount

    val newValues = data.orderBy($"t".desc).select("value").limit(newValuesCount).map(_.getDouble(0)).collect().reverse
    testVectorInit = Vectors.dense(((for (i <- testVectorInit.size - oldValuesCount until testVectorInit.size) yield testVectorInit(i)) ++ newValues).toArray)

    this
  }

  def getInitVector: Vector = testVectorInit

  def setModels(regressionModels: RegressionModel[Vector, _ <: RegressionModel[Vector, _]]*): this.type = {
    models = regressionModels.toArray
    this
  }

  def getModels: Array[_ <: RegressionModel[Vector, _]] = models

  def setNames(names: String*): this.type = {
    colNames = names.toArray
    this
  }

  def getNames: Array[String] = {
    if (colNames.length > 0)
      colNames
    else
      models.map(_.uid)
  }

  private def predictAndChangeVector(value: Option[Double] = None): Array[Double] = {
    val predictions = for (model <- models) yield model.predict(testVector)
    testVector = Vectors.dense(((for (i <- 1 until testVector.size) yield testVector(i)) :+ value.getOrElse(predictions(0))).toArray)
    predictions
  }

  override def predict(data: sql.DataFrame): sql.DataFrame = predict(data, "")

  def predict(data: sql.DataFrame, trueCol: String): sql.DataFrame = {
    testVector = testVectorInit
    val t = data.select("t").orderBy("t").map(_.getInt(0)).collect()

    val predictionsDF = {
        val schema = StructType(
          StructField("t", IntegerType, true) +:
            getNames.map(StructField(_, DoubleType, true))
        )

        spark.createDataFrame(
          spark.sparkContext.parallelize(
            if (trueCol.isEmpty)
              for (ti <- t) yield Row.fromSeq(ti +: predictAndChangeVector())
            else {
              val values = data.select(trueCol).orderBy("t").map(_.getDouble(0)).collect()
              for ((ti, value) <- t zip values) yield Row.fromSeq(ti +: predictAndChangeVector(Some(value)))
            }
          ),
          schema
        )
      }

    data.join(predictionsDF, "t")
  }

  def predictInterval(data: sql.DataFrame, interval: Int, trueCol: String = ""): sql.DataFrame = {
    val target = data.orderBy("t").limit(interval)
    data.except(target)
        .select(data.col("*") +: getNames.map(lit(null).cast("double").as(_)): _*)
        .union(predict(target, trueCol))
        .orderBy("t")
  }

}
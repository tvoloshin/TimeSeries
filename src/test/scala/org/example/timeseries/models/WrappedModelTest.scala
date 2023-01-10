package org.example.timeseries.models

import org.apache.spark.ml.regression.RegressionModel
import org.apache.spark.sql
import org.apache.spark.ml.linalg.{Vector, Vectors}
import org.apache.spark.sql.SparkSession
import org.junit.jupiter.api.Assertions.assertArrayEquals
import org.junit.jupiter.api.Test
import org.mockito.ArgumentMatchers.any
import org.mockito.Mockito.{mock, when}

class WrappedModelTest {

  val spark: SparkSession = sql.SparkSession
    .builder()
    .appName("TimeSeries")
    .config("spark.master", "local[*]")
    .getOrCreate()
  import spark.implicits._

  val simpleData: Seq[Double] = Seq.range(1, 10).map(_.toDouble)

  @Test
  def testSetInitVector(): Unit = {
    val vectorSize = 4
    val expectedRes = simpleData.takeRight(vectorSize)

    val wrappedModel = new WrappedModel().setOrderCol("value")
    wrappedModel.setInitVector(simpleData.toDF("value"), vectorSize)

    assertArrayEquals(expectedRes.toArray, wrappedModel.getInitVector.toArray)
  }

  @Test
  def testPredict(): Unit = {
    val df = simpleData.toDF("t")

    val vectorSize = 4
    val initVector = Vectors.dense(Array.range(1, vectorSize + 1).map(_.toDouble))
    val expectedRes = Array.range(vectorSize + 1, simpleData.size + vectorSize + 1).map(_.toDouble)

    val mockedModel = mock(classOf[RegressionModel[Vector, _ <: RegressionModel[Vector, _]]])
    when(mockedModel.predict(any(classOf[Vector])))
      .thenAnswer(i => i.getArgument(0).asInstanceOf[Vector].toArray.last + 1)
    val wrappedModel = new WrappedModel()
      .setModels(mockedModel)
      .setNames("predictions")
      .setInitVector(initVector)
    val actualDF = wrappedModel.predict(df)

    assertArrayEquals(expectedRes, actualDF.collect().map(_.getAs[Double]("predictions")))
  }

}

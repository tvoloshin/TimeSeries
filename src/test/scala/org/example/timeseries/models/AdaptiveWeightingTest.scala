package scala.org.example.timeseries.models

import org.apache.spark.ml.regression.RegressionModel
import org.apache.spark.ml.linalg.{Vector, Vectors}
import org.apache.spark.sql.SparkSession
import org.example.timeseries.models.AdaptiveWeighting
import org.junit.jupiter.api.Assertions.assertArrayEquals
import org.junit.jupiter.api.Test
import org.mockito.ArgumentMatchers.any
import org.mockito.Mockito.{mock, when}

class AdaptiveWeightingTest {

  val spark: SparkSession = SparkSession
    .builder()
    .appName("TimeSeries")
    .config("spark.master", "local[*]")
    .getOrCreate()
  spark.sparkContext.setLogLevel("ERROR")

  import spark.implicits._

  @Test
  def testPredict(): Unit = {
    val data = Array.range(1, 8).map(_.toDouble)
    val vectorSize = 4

    val initVector = Vectors.dense(data.take(vectorSize))
    val targetValues = data.takeRight(data.length - vectorSize)
    val df = targetValues.toSeq.toDF("t")

    val stubErrors = Array(0.3, 0.6, 0.9)

    val expectedErrors = (stubErrors.sum / stubErrors.length) +: Seq.fill(df.count().toInt - 1) {stubErrors.length / stubErrors.map(1 / _).sum}
    val expectedRes = (targetValues, expectedErrors).zipped.map(_ + _)

    val mockedModels =
      for ((value, i) <- stubErrors.zipWithIndex) yield {
        val model = mock(classOf[RegressionModel[Vector, _ <: RegressionModel[Vector, _]]])
        when(model.predict(any(classOf[Vector])))
          .thenAnswer(i => i.getArgument(0).asInstanceOf[Vector].toArray.last + 1 + value)
        when(model.uid).thenReturn("model_" + i)
        model
      }

    val adaptiveWeighting = new AdaptiveWeighting(mockedModels: _*)
      .setInitVector(initVector)
      .setPredictionInterval(1)
      .setOrderCol("t").setValueCol("t")
    val actualDF = adaptiveWeighting.predict(df)

    assertArrayEquals(expectedRes, actualDF.collect().map(_.getAs[Double]("prediction")), 1e-5)
  }

}

package org.example.timeseries.utils

import org.apache.spark.sql
import org.junit.jupiter.api.Test
import org.junit.jupiter.api.Assertions._
import com.github.mrpowers.spark.fast.tests.DatasetComparer
import org.apache.spark.ml.linalg
import org.apache.spark.ml.linalg.Vectors
import org.apache.spark.sql.{DataFrame, SparkSession}

class FeatureUtilsTest extends DatasetComparer {

  val spark: SparkSession = sql.SparkSession
    .builder()
    .appName("TimeSeries")
    .config("spark.master", "local[*]")
    .getOrCreate()

  import spark.implicits._

  val simpleData: DataFrame = Seq.range(1, 10).toDF("value")
  val dataWithFeatures: Seq[(Int, linalg.Vector)] = Seq(
    (1, Vectors.dense(Array.emptyDoubleArray)),
    (2, Vectors.dense(1)),
    (3, Vectors.dense(1, 2)),
    (4, Vectors.dense(1, 2, 3)),
    (5, Vectors.dense(2, 3, 4)),
    (6, Vectors.dense(3, 4, 5)),
    (7, Vectors.dense(4, 5, 6)),
    (8, Vectors.dense(5, 6, 7)),
    (9, Vectors.dense(6, 7, 8))
  )

  @Test
  def testWithFeatures(): Unit = {
    val expectedDF = dataWithFeatures.toDF("value", "features")
    val actualDF = FeatureUtils.withFeatures(simpleData, 3, "value")
    assertSmallDatasetEquality(expectedDF, actualDF)
  }

  @Test
  def testSplitData(): Unit = {
    val size1 = 6
    val size2 = simpleData.count() - size1
    val Array(df1, df2) = FeatureUtils.splitData(simpleData, size1, "value")

    assertAll(
      () => assertEquals(size1, df1.count()),
      () => assertEquals(size2, df2.count())
    )
  }

  @Test
  def testSplitTrainTest(): Unit = {
    val trainSize = 4
    val winSize = 3

    val expectedTrain = (for (i <- winSize until winSize + trainSize) yield dataWithFeatures(i)).toDF("value", "features")
    val Array(actualTrain, _) = FeatureUtils.splitTrainTest(simpleData, trainSize, 3, "value")

    assertSmallDatasetEquality(actualTrain, expectedTrain)
  }

}

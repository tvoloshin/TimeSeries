package org.example.timeseries.models

import org.apache.spark.ml.linalg.{Vector, Vectors}
import org.apache.spark.ml.regression.RegressionModel
import org.junit.jupiter.api.Assertions.assertEquals
import org.junit.jupiter.api.Test
import org.mockito.Mockito._

class VotingModelTest {

  @Test
  def testPredict(): Unit = {
    val stubVector = Vectors.dense(Array.emptyDoubleArray)

    val stubPredictions = Array(1, 2, 3)
    val stubWeights = Array(0.25, 0.25, 0.5)
    val expectedRes = (stubPredictions, stubWeights).zipped.map(_ * _).sum

    val mockedModels =
      for (value <- stubPredictions) yield {
        val model = mock(classOf[RegressionModel[Vector, _ <: RegressionModel[Vector, _]]])
        when(model.predict(stubVector)).thenReturn(value)
        model
      }
    val votingModel = new VotingModel(mockedModels).setWeights(stubWeights: _*)

    assertEquals(expectedRes, votingModel.predict(stubVector))
  }
}

package org.example.timeseries.models

import org.apache.spark.sql

trait MyModel {

  def predict(data: sql.DataFrame): sql.DataFrame

}

package org.example.timeseries.utils

import breeze.plot.{Figure, plot}

class PlotUtils(name: String) {

  private var data: Option[Array[Array[Double]]] = None
  private var legend: Option[Array[String]] = None

  def addPlots(valArrays: Array[Double]*): this.type = {
    data = Some(valArrays.toArray)
    this
  }

  def addLegend(names: String*): this.type = {
    legend = Some(names.toArray)
    this
  }

  def create(): Unit = {
    if (data.isDefined) {
      val p = Figure(name).subplot(0)

      if (legend.isEmpty)
        for (arr <- data.get) p += plot(arr.indices.map(_.toDouble), arr)
      else {
        for ((arr, name) <- data.get zip legend.get) p += plot(arr.indices.map(_.toDouble), arr, name = name)
        p.legend = true
      }
    }
  }
}

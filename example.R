
# install.packages("devtools", repos = "http://cran.us.r-project.org")
# install.packages("devtools")
# library(devtools)
# load_all("/rqPen")
# devtools::load_all("/rqPen")
# .libPaths("/rqPen")

# library(rqPen)

#install.packages("/Users/evanwang/Desktop/eecs/eecs553/553-Project/rqPen", repos = NULL, type = "source")

#library(rqPen, lib.loc = "/Users/evanwang/Desktop/eecs/eecs553/553-Project/rqPen")
.libPaths(c("/Users/evanwang/Desktop/eecs/eecs553/553-Project/rqPen", .libPaths()))
install.packages("rqPen", lib = "C:/Users/evanwang/Desktop/eecs/eecs553/553-Project/rqPen")
library(rqPen, lib.loc = "/Users/evanwang/Desktop/eecs/eecs553/553-Project/rqPen")

rq.pen.printhi()


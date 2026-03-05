### Part -[1]-
#' load data.
setwd("/WData/WData2/Multimodal")

load("./Data/Cox/Training.RFS.Feature.RData")
data <- readRDS("./Data/Exprs/Training.DL.r3d50_KM.rds")
Training.Groups <- readRDS("./Data/Group/Training.Groups.rds")

data <- data[rownames(data) %in% Training.RFS.Feature$Character, ]
data <- data[order(rownames(data)), ] ## 


### Part -[2]-
#' RRS.

Score <- log2(Training.RFS.Feature$HR) * data ## 
Training.Groups$RFS.Score.log2 <- apply(Score, 2, sum) ## 


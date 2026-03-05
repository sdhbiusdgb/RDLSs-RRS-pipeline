### Part --[1]---

setwd("/WData/WData2/Multimodal")

library(ConsensusClusterPlus)

#' r3d50_KM

data <- readRDS("./Data/Exprs/Training.DL.r3d50_KM.rds")
con <- ConsensusClusterPlus(as.matrix(data), # 1867*931
    maxK = 10, 
    reps = 500, 
    pItem = 0.8,
    pFeature = 1,
    title = "ConsensusClusterPlus.Training.r3d50_KM", 
    clusterAlg = "km", 
    distance = "euclidean", 
    seed = 1234, 
    plot = "pdf", 
    writeTable = TRUE
)

k3_groups <- con[[3]]$consensusClass 
table(k3_groups) 
Training.Groups <- data.frame(r3d50_KM = k3_groups)
Training.Groups$r3d50_KM <- paste("cluster", Training.Groups$r3d50_KM, sep = "")

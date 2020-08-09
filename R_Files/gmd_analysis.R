library(caret)

gw_data <- read.csv('../Outputs/Output_Apr_Sept/GW_Raster.csv')
postResample(gw_data$Actual_GW, gw_data$Pred_GW)

gmd1 <- gw_data[which(gw_data$GMD=='GMD10' & gw_data$YEAR<2019),]
postResample(gmd1$Pred_GW, gmd1$Actual_GW)
gmd4 <- gw_data[which(gw_data$GMD=='GMD4' & gw_data$YEAR<2019),]
postResample(gmd4$Pred_GW, gmd4$Actual_GW)

gmd1_4 <- subset(gw_data, (GMD %in% c('GMD1', 'GMD4')) & (YEAR < 2019))
postResample(gmd1_4$Pred_GW, gmd1_4$Actual_GW)
t <- gmd1_4[gmd1_4$YEAR != 2012,]
t <- gmd1_4[gmd1_4$YEAR != 2007,]
postResample(t$Pred_GW, t$Actual_GW)

gmd_train <- subset(gw_data, (GMD %in% c('GMD2', 'GMD3', 'GMD5')) & (YEAR < 2019))
postResample(gmd_train$Pred_GW, gmd_train$Actual_GW)

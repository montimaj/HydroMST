library(caret)

gw_data <- read.csv('../Outputs/Output_Apr_Sept/gw_yearly_new.csv')
gw_data <- na.omit(gw_data)
postResample(gw_data$Actual_GW, gw_data$Pred_GW)

gw_data <- read.csv('../Outputs/Output_Apr_Sept/GW_Raster.csv')
postResample(gw_data$Actual_GW, gw_data$Pred_GW)

gw_data <- read.csv('../Outputs/Output_Apr_Sept/gw_yearly_new.csv')
grace_data <- read.csv('../Inputs/Data/GRACE/TWS_GRACE.csv')
par(mfrow=c(3, 2))
plot(gw_data)

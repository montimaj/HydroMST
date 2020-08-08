library(raster)
library(colorRamps)

err.raster <- list()
pred.raster.list <- list()
years <- seq(2011, 2018)
k <- 1
for (i in years) {
  pred.raster <- raster(paste("../Outputs/Output_Apr_Sept/Predicted_Rasters/pred_", i, ".tif", sep=""))
  actual.raster <- raster(paste("../Inputs/Files_Apr_Sept/RF_Data/GW_", i, ".tif", sep=""))
  wgs84 = "+proj=longlat +ellps=WGS84 +datum=WGS84 +no_defs"
  actual.raster = projectRaster(actual.raster, crs = wgs84, method = "ngb")
  pred.raster = projectRaster(pred.raster, crs = wgs84, method = "ngb")
  err.raster[[k]] <- actual.raster - pred.raster
  pred.raster.list[[k]] <- pred.raster
  k <- k + 1
}
err.raster.stack <- stack(err.raster)
pred.raster.stack <- stack(pred.raster.list)
err.mean.raster <- mean(err.raster.stack)
plot(err.mean.raster, col = matlab.like2(255), ylab='Latitude (Degree)', xlab='Longitude (Degree)', 
     legend.args=list(text='Error (mm)', side = 2))
err.df <- as.data.frame(err.mean.raster, na.rm = T)
err <- err.df$layer
err.mean <- mean(err)
err.sd <- sd(err)
std.err <- err / err.sd
std.err.df <- as.data.frame(std.err)
names(std.err.df) <- c('STD.ERR')
hist(std.err.df$STD.ERR, breaks=50, freq = F, main="", xlab='Standardized Residuals')
x <- seq(min(std.err.df$STD.ERR), max(std.err.df$STD.ERR), length.out=length(std.err.df$STD.ERR))
dist <- dnorm(x, mean(std.err.df$STD.ERR), sd(std.err.df$STD.ERR))
lines(x, dist, col = 'red')

pred.raster.df <- as.data.frame(mean(pred.raster.stack),na.rm=T)
names(pred.raster.df) <- c('pred')
plot(pred.raster.df$pred, std.err, xlab = 'Mean Predicted GW Pumping (mm)', ylab = 'Standardized Residuals')
abline(h = 0, col = "red")
qqnorm(std.err, main = "")
qqline(std.err, col = "red")

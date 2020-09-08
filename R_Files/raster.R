library(raster)
library(colorRamps)

pred.raster <- raster('../Outputs/Output_Apr_Sept/Predicted_Rasters/pred_2018.tif')
actual.raster <- raster('../Inputs/Files_Apr_Sept/RF_Data/GW_2018.tif')

wgs84 <- "+proj=longlat +ellps=WGS84 +datum=WGS84 +no_defs"
actual.raster <- projectRaster(actual.raster, crs = wgs84, method = "ngb")
pred.raster <- projectRaster(pred.raster, crs = wgs84, method = "ngb")

min_value  <- round(min(minValue(actual.raster), minValue(pred.raster)))
max_value <- round(max(maxValue(actual.raster), maxValue(pred.raster)))
max_value <- ceiling(max_value / 100) * 100
breaks <- seq(min_value, max_value, by=100)
col <- rev(terrain.colors(length(breaks) - 1))
plot(actual.raster, ylab='Latitude (Degree)', xlab='Longitude (Degree)', yaxt='n',
     legend.args=list(text='GW Pumping (mm)', side = 2, font = 0.5, cex = 0.8), breaks=breaks, zlim=c(min_value, max_value), col=col)
axis(side=2, at=c(37, 38, 39, 40))
plot(pred.raster, ylab='Latitude (Degree)', xlab='Longitude (Degree)', yaxt='n',
     legend.args=list(text='GW Pumping (mm)', side = 2, font = 0.5, cex = 0.8), breaks=breaks, zlim=c(min_value, max_value), col=col)
axis(side=2, at=c(37, 38, 39, 40))

plot(pred.raster, actual.raster, xlab="Predicted GW Pumping (mm)", ylab="Actual GW Pumping (mm)")
legend(0, 600, bty = 'n', legend = c("1:1 relationship"),
       col = c("red"), lty = 1, cex = 0.8)

err.raster <- actual.raster - pred.raster
plot(err.raster, col = matlab.like2(255), ylab='Latitude (Degree)', xlab='Longitude (Degree)', 
     legend.args=list(text='Error (mm)', side = 2, font = 1, cex = 1), yaxt='n')
axis(side=2, at=c(37, 38, 39, 40))
err.df <- as.data.frame(err.raster, na.rm = T)
err <- err.df$layer
err.mean <- mean(err)
err.sd <- sd(err)
std.err <- err / err.sd
std.err.df <- as.data.frame(std.err)
names(std.err.df) <- c('STD.ERR')
std.err.df$STD.ERR[std.err.df$STD.ERR < -2] <- NA
std.err.df$STD.ERR[std.err.df$STD.ERR > 2] <- NA
std.err.df <- na.omit(std.err.df)
hist(std.err.df$STD.ERR, freq = F, main="", xlab='Standardized Residuals')
x <- seq(min(std.err.df$STD.ERR), max(std.err.df$STD.ERR), length.out=length(std.err.df$STD.ERR))
dist <- dnorm(x, mean(std.err.df$STD.ERR), sd(std.err.df$STD.ERR))
lines(x, dist, col = 'red')

s <- stack(actual.raster, pred.raster)
s.df <- as.data.frame(s, na.rm=T)
names(s.df) <- c('actual', 'pred')
plot(s.df$pred, std.err, xlab = 'Predicted GW Pumping (mm)', ylab = 'Standardized Residuals')
abline(h = 0, col = "red")
qqnorm(std.err, main = "")
qqline(std.err, col = "red")

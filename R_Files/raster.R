library(raster)
library(colorRamps)

pred.raster <- raster('../Outputs/Output_Apr_Sept/Predicted_Rasters/pred_2014.tif')
actual.raster <- raster('../Inputs/Files_Apr_Sept/RF_Data/GW_2014.tif')

wgs84 = "+proj=longlat +ellps=WGS84 +datum=WGS84 +no_defs"
actual.raster = projectRaster(actual.raster, crs = wgs84, method = "ngb")
pred.raster = projectRaster(pred.raster, crs = wgs84, method = "ngb")

par(mfrow = c(1, 2))
plot(actual.raster, ylab='Latitude (Degree)', xlab='Longitude (Degree)', 
     legend.args=list(text='GW Pumping (mm)', side = 2, font = 0.5, cex = 0.8))
plot(pred.raster, ylab='Latitude (Degree)', xlab='Longitude (Degree)', 
     legend.args=list(text='GW Pumping (mm)', side = 2, font = 0.5, cex = 0.8))


par(mfrow = c(1, 1))
s <- stack(actual.raster, pred.raster)
s.df <- as.data.frame(s, na.rm=T)
names(s.df) <- c('actual', 'pred')
plot(pred.raster, actual.raster, log='xy', xlab="Predicted GW Pumping", ylab="Actual GW Pumping")
gw.fit <- lm(actual ~ pred, data = s.df)
abline(gw.fit, col='red')
abline(a = 0, b = 1, col = 'blue')
legend(1e-10, 1e+2, bty = 'n', legend = c("1:1 relationship", "Fitted regression line"),
       col = c("blue", "red"), lty = 1, cex = 0.8)
summary(gw.fit)
confint(gw.fit)

par(mfrow = c(2, 2))
err.raster <- actual.raster - pred.raster
plot(err.raster, col = matlab.like2(255), ylab='Latitude (Degree)', xlab='Longitude (Degree)', 
     legend.args=list(text='Error (mm)', side = 2, font = 0.5, cex = 0.55))
err.df <- as.data.frame(err.raster, na.rm = T)
err <- err.df$layer
hist(err, freq = F, main="", xlab='Residuals (mm)')
lines(density(err), col = 'red')
err.mean <- mean(err)
err.sd <- sd(err)
std.err <- err / err.sd
hist(std.err, freq = F, main="", xlab='Standardized Residuals')
lines(density(std.err), col = 'red')
plot(s.df$pred, err, xlab = 'Predicted GW Pumping (mm)', ylab = 'Residuals (mm)')
abline(h = 0, col = "red")
plot(s.df$pred, std.err, xlab = 'Predicted GW Pumping (mm)', ylab = 'Standardized Residuals')
abline(h = 0, col = "red")
qqnorm(std.err, main = "")
qqline(std.err, col = "red")


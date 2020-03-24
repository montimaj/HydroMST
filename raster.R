library(raster)
library(colorRamps)

pred.raster <- raster('C:\\Users\\sayan\\PycharmProjects\\HydroMST\\Output\\Cropped_Rasters_All\\pred_2012.tif')
actual.raster <- raster('C:\\Users\\sayan\\PycharmProjects\\HydroMST\\Output\\Cropped_Rasters_All\\GW_KS_2012.tif')

par(mfrow = c(1, 2))
plot(actual.raster, ylab='Latitude (m)', xlab='Longitude (m)', 
     legend.args=list(text='GW Pumping (mm)', side = 2, font = 0.5, cex = 0.8))
plot(pred.raster, ylab='Latitude (m)', xlab='Longitude (m)', 
     legend.args=list(text='GW Pumping (mm)', side = 2, font = 0.5, cex = 0.8))


par(mfrow = c(1, 1))
plot(actual.raster, pred.raster, xlab="Predicted GW Pumping (mm)", ylab="Actual GW Pumping (mm)")
s <- stack(actual.raster, pred.raster)
s.df <- as.data.frame(s, na.rm=T)
names(s.df) <- c('actual', 'pred')
gw.fit <- lm(actual ~ pred, data = s.df)
abline(gw.fit, col = 'red')
abline(a = 0, b = 1, col = 'blue')
legend(0, 450, bty = 'n', legend = c("True relationship", "Fitted regression line"),
       col = c("blue", "red"), lty = 1, cex = 0.8)
summary(gw.fit)
confint(gw.fit)

par(mfrow = c(3, 2))
err.raster <- actual.raster - pred.raster
plot(err.raster, col = matlab.like(255), ylab='Latitude (m)', xlab='Longitude (m)', 
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


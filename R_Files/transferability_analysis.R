library(raster)
library(colorRamps)
library(sf)

pred_raster_list <- list()
actual_raster_list <- list()
years <- seq(2002, 2018)
k <- 1
for (year in years) {
  pred_raster_list[[k]] <- raster(paste('../Outputs/Output_Apr_Sept/Pred_GMD_Rasters/GMD1/pred_',year,'.tif', sep=''))
  actual_raster_list[[k]] <- raster(paste('../Outputs/Output_Apr_Sept/Actual_GMD_Rasters/GMD1/GW_',year,'.tif', sep=''))
  k <- k + 1
}
actual.raster.stack <- stack(actual_raster_list)
pred.raster.stack <- stack(pred_raster_list)
actual.raster <- mean(actual.raster.stack)
pred.raster <- mean(pred.raster.stack)

boundary <- extent(485025, 556012, 4090170, 4436007)
actual.raster <- crop(actual.raster, boundary)
pred.raster <- crop(pred.raster, boundary)
wgs84 <- "+proj=longlat +ellps=WGS84 +datum=WGS84 +no_defs"
actual.raster <- projectRaster(actual.raster, crs = wgs84, method = "ngb")
pred.raster <- projectRaster(pred.raster, crs = wgs84, method = "ngb")


par(mfrow = c(1, 2))
plot(actual.raster, ylab='Latitude (Degree)', xlab='Longitude (Degree)', 
     legend.args=list(text='GW Pumping (mm)', side = 2, font = 0.5, cex = 0.8))
plot(pred.raster, ylab='Latitude (Degree)', xlab='Longitude (Degree)', 
     legend.args=list(text='GW Pumping (mm)', side = 2, font = 0.5, cex = 0.8))


par(mfrow = c(1, 1))
plot(pred.raster, actual.raster, xlab="Predicted GW Pumping (mm)", ylab="Actual GW Pumping (mm)")
s <- stack(actual.raster, pred.raster)
s.df <- as.data.frame(s, na.rm=T)
names(s.df) <- c('actual', 'pred')
plot(s.df$pred, s.df$actual, xlab="Predicted GW Pumping (mm)", ylab="Actual GW Pumping (mm)")
gw.fit <- lm(actual ~ pred, data = s.df)
abline(gw.fit, col = 'red')
abline(a = 0, b = 1, col = 'red')
s.log.df <- s.df[s.df$actual != 0,]
s.log.df <- s.log.df[s.log.df$pred !=0, ]
s.log.df$actual <- log(s.log.df$actual)
s.log.df$pred <- log(s.log.df$pred)
plot(s.log.df$pred, s.log.df$actual, xlab="Predicted GW Pumping", ylab="Actual GW Pumping")
gw.fit.log <- lm(actual ~ pred, data=s.log.df)
abline(gw.fit.log, col = 'red')
abline(gw.fit, col = 'red')
abline(a = 0, b = 1, col = 'blue')
legend(0, 600, bty = 'n', legend = c("1:1 relationship"),
       col = c("red"), lty = 1, cex = 0.8)
summary(gw.fit)
confint(gw.fit)

par(mfrow = c(2, 2))
err.raster <- actual.raster - pred.raster
plot(err.raster, col = matlab.like2(255), ylab='Latitude (Degree)', xlab='Longitude (Degree)', 
     legend.args=list(text='Error (mm)', side = 2, font = 0.5, cex = 0.55))
plot(err.raster, col = matlab.like2(255), ylab='Latitude (Degree)', xlab='Longitude (Degree)', 
     legend.args=list(text='Error (mm)', side = 2))
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
plot(s.df$pred, std.err, xlab = 'Predicted GW Pumping (mm)', ylab = 'Standardized Residuals')
abline(h = 0, col = "red")
qqnorm(std.err, main = "")
qqline(std.err, col = "red")


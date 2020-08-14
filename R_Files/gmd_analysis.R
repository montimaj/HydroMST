library(caret)

gw_data <- read.csv('../Outputs/Output_Apr_Sept/gw_yearly_new.csv')
gw_data <- na.omit(gw_data)
postResample(gw_data$Actual_GW, gw_data$Pred_GW)

gw_data <- read.csv('../Outputs/Output_Apr_Sept/GW_Raster.csv')
postResample(gw_data$Actual_GW, gw_data$Pred_GW)

## Pixelwise GMD Analysis for Transferability

gw_data <- read.csv('../Outputs/Output_Apr_Sept/GMD_Train_Test.csv')
gmd_names <- c('GMD1', 'GMD2', 'GMD3', 'GMD4', 'GMD5')
df_names = c('TRAIN', 'TEST', 'FORECAST')
result_df <- data.frame()
for (gmd in gmd_names) {
  gmd_df <- gw_data[gw_data$GMD == gmd,]
  for (df_name in df_names) {
    gmd_sub_df <- gmd_df[gmd_df$DATA == df_name,]
    if (df_name == 'FORECAST') {
      gmd_sub_df <- gmd_df[gmd_df$DATA != df_name,]
      df_name <- 'TRAIN+TEST'
    }
    actual_gw = gmd_sub_df$Actual_GW
    pred_gw = gmd_sub_df$Pred_GW
    res <- postResample(pred_gw, actual_gw)
    r2 <- round(res[['Rsquared']], 2)
    rmse <- res[['RMSE']]
    mae <- res[['MAE']]
    nmae <- round(mae / mean(actual_gw), 2)
    nrmse <- round(rmse / mean(actual_gw), 2)
    df <- data.frame(gmd, df_name, r2, round(rmse, 2), round(mae, 2), nmae, nrmse)
    names(df) <- c('GMD', 'DATA', 'R2', 'RMSE', 'MAE', 'NMAE', 'NRMSE')
    result_df <- rbind(result_df, df)
  }
}
result_df <- result_df[order(result_df$DATA, result_df$GMD),]
result_df

## Mean GMD Analysis for Transferability

gw_data <- read.csv('../Outputs/Output_Apr_Sept/GMD_Metrics_Train_Test_Yearly.csv')
gmd_names <- c('GMD1', 'GMD2', 'GMD3', 'GMD4', 'GMD5')
df_names = c('TRAIN', 'TEST', 'TRAIN+TEST')
result_df <- data.frame()
for (gmd in gmd_names) {
  gmd_df <- gw_data[gw_data$GMD == gmd,]
  for (df_name in df_names) {
    gmd_sub_df <- gmd_df[gmd_df$DATA == df_name,]
    actual_gw = gmd_sub_df$Mean_Actual_GW
    pred_gw = gmd_sub_df$Mean_Pred_GW
    res <- postResample(pred_gw, actual_gw)
    r2 <- round(res[['Rsquared']], 2)
    rmse <- res[['RMSE']]
    mae <- res[['MAE']]
    nmae <- round(mae / mean(actual_gw), 2)
    nrmse <- round(rmse / mean(actual_gw), 2)
    df <- data.frame(gmd, df_name, r2, round(rmse, 2), round(mae, 2), nmae, nrmse)
    names(df) <- c('GMD', 'DATA', 'R2', 'RMSE', 'MAE', 'NMAE', 'NRMSE')
    result_df <- rbind(result_df, df)
  }
}
result_df <- result_df[order(result_df$DATA, result_df$GMD),]
result_df

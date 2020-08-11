library(corrplot)
library(PerformanceAnalytics)

df <- read.csv('../Outputs/Output_Apr_Sept/X_Train.csv')
df['CC'] <- df['Crop']
df['Crop'] <- NA
test_df <- read.csv('../Outputs/Output_Apr_Sept/Y_Train.csv')
cor(df$AGRI, df$SW)
cor(df$AGRI, df$URBAN)
cor(df$SW, df$URBAN)
cor(df$AGRI, df$ET)
cor(df$AGRI, df$P)
cor(df$ET, df$P)

sub_df <- subset(df, select=c('AGRI', 'CC', 'ET', 'P', 'SW', 'URBAN'))
sub_df['GW'] <- test_df$X0
corrplot(cor(sub_df), type = "upper", order="hclust", tl.col = "black")

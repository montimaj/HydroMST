# load in cropscape data, mask and simplify to urban areas and crops
## MAIN AUTHOR: Dr. Ryan Smith
## MODIFICATIONS BY: Sayantan Majumdar

### Set working directory accordingly ###
setwd('/Users/smxnv/Documents/MST/Data/watersheds')

library(raster)
library(sp)
library(rgdal)
library(ggplot2)

ag=5

########### ARIZONA DATA ###############
az_watershed=readOGR('az_merged/az_watershed.shp')
az_crop=raster('../cropscape/CDL_2015_clip_20190812153756_568423369/CDL_2015_clip_20190812153756_568423369.tif')
az_crop=mask(az_crop,az_watershed)
# 
class_mat=c(0,59.5,1,67.5,75.5,1,203.5,250,1,110.5,111.5,2,120.5,124.5,3,60.5,61.5,NA,130.5,195.5,NA)
class_mat=matrix(class_mat,ncol=3,byrow=T)
az_reclass=reclassify(az_crop,class_mat)

plot(az_crop)
plot(az_reclass)
# 
writeRaster(az_reclass,filename = '../cropscape/az_reclass.tiff',overwrite=T)

########## KANSAS DATA #############
class_mat=c(0,59.5,1,66.5,77.5,1,203.5,255,1,110.5,111.5,2,111.5,112.5,NA,
            120.5,124.5,3,59.5,61.5,NA,130.5,195.5,NA)
class_mat=matrix(class_mat,ncol=3,byrow=T)

ks_watershed=readOGR('ks_merged/ks_watershed.shp')
ks_crop=raster('../cropscape/polygonclip_20190306140312_392696635/CDL_2015_clip_20190306140312_392696635.tif')
#ks_crop=mask(ks_crop, ks_watershed)
ks_reclass=reclassify(ks_crop,class_mat)
ks_reclass2=aggregate(ks_reclass,fact=3,fun=modal)
writeRaster(ks_reclass2,filename = '../cropscape/ks_reclass2.tiff',overwrite=T)
browser()
az_reclass=raster('../cropscape/az_reclass.tif')
save(az_reclass,ks_reclass,file='../crop_reclass.Rda')
load('../crop_reclass.Rda')

# load in ET and Precip data
setwd('../ET_precip/')
ET_files=list.files('.',pattern='ET')
P_files=list.files('.',pattern='precip')
ET=stack(ET_files)
P=stack(P_files)
# demand=ET-P # use ET-P for demand
# class_mat=c(-1000,0,0)
# class_mat=matrix(class_mat,ncol=3,byrow=T)
# demand2=reclassify(demand,class_mat)
demand2=ET # only use ET for demand
names(demand2)=c('august','july','june')
demand_all=sum(demand2)
ks_resamp=projectRaster(ks_reclass,demand_all,method='ngb')
filt=ks_resamp!=1|is.na(ks_resamp)
demand_all[filt]=0

gw_pumping=raster('../2015_smoothed/wuse_st_data/wuse_density_0515_5mile_clip.img')
# convert to mm
gw_pumping=gw_pumping*1233.48*1000/2.59e6
ks_watershed2=spTransform(ks_watershed,crs(gw_pumping))
# gw_pumping=projectRaster(gw_pumping,demand_all)
demand_all_reproj=projectRaster(demand_all,gw_pumping)
merge=stack(demand_all_reproj,gw_pumping)
names(merge)=c('ET_P','gw_Q')
merge=aggregate(merge,fact=ag)

# look at surface water
rivers=readOGR('../watersheds/ks_rivers/ks_rivers.shp')
rivers=spTransform(rivers,crs(merge))
water=ks_resamp==2
water[is.na(water)]=0
water2=projectRaster(water,gw_pumping,method='ngb')
water3=aggregate(water2,fact=ag,fun=max)
r <- raster(ncols=ncol(water3), nrows=nrow(water3), xmn=0)
wt=focalWeight(r, 5, "Gauss")
water4=focal(water3,w=wt)
#matrix of distance to water
# dist_water=matrix(nrow = nrow(water2),ncol=ncol(water2))
# lons=matrix(coordinates(water2)[,1],nrow=nrow(water2),ncol=ncol(water2),byrow=T)
# lats=matrix(coordinates(water2)[,2],nrow=nrow(water2),ncol=ncol(water2),byrow=T)
# water2_matrix=matrix(water2,nrow=nrow(water2),ncol=ncol(water2),byrow=T)
# water2_inds=which(water2_matrix==1,arr.ind=T)
# for(x in 1:ncol(water2)){
#   for(y in 1:nrow(water2)){
#     lon=lons[y,x];lat=lats[y,x]
#     dist=sqrt((lon-lons)^2+(lat-lats)^2)
#     dist_wat=dist[water2_inds]
#     dist_water[y,x]=min(dist_wat,na.rm=T)
#   }
# }
# dwat=as.vector(t(dist_water))
# dwat_r=water2
# dwat_r@data@values=dwat
# plot(dwat_r)

# extract urban areas, sum precip
urban1=ks_resamp==3
urban=projectRaster(urban1,gw_pumping,method='ngb')
urban[is.na(urban)]=0;#urban=mask(urban,ks_watershed2)
urban=aggregate(urban,fact=5,fun=mean,na.rm=T)
P_all1=sum(P)
P_all=projectRaster(P_all1,gw_pumping)
P_all=aggregate(P_all,fact=5,fun=mean,na.rm=T)

# merge and plot
merge=stack(merge,water4,urban,P_all)
filt=is.na(merge$ET_P)|is.na(merge$gw_Q)
merge[filt]=NA
names(merge)=c('ET_P','gw_Q','sw','urban','P')
wt1=focalWeight(r, 3, "Gauss")
merge$ET_filt=focal(merge$ET_P,wt1)
merge$diff=merge$ET_filt-merge$gw_Q
pumping.df=as.data.frame(merge)
g=ggplot(pumping.df,aes(x=ET_P,y=gw_Q,col=sw))+geom_point(alpha=.3)+geom_abline()
print(g)
merge=mask(merge,ks_watershed2)
plot(merge)

# random forest to predict gw_pumping
library(randomForest)
ind=match('diff',names(pumping.df))
pumping.df2=pumping.df[complete.cases(pumping.df),-ind]
train_filt=runif(nrow(pumping.df2))>.5
train=pumping.df2[train_filt,]
val=pumping.df2[train_filt==0,]

rf=randomForest(gw_Q~.,data=train,importance=T)
val$gw_Q_predict=predict(rf,val)
plot(val$gw_Q_predict,val$gw_Q)
pumping.df$gw_Q_predict=predict(rf,pumping.df)
merge$gw_predict=pumping.df$gw_Q_predict
merge=mask(merge,ks_watershed2)
plot(merge[[c(2,8)]])
partialPlot(rf,train,'ET_filt')
partialPlot(rf,train,'P')
partialPlot(rf,train,'urban')
partialPlot(rf,train,'sw')

# now create the same dataset for the whole watershed
# ET_filt=
water2=projectRaster(water,res=res(gw_pumping),crs=crs(gw_pumping),method='ngb')
water3=aggregate(water2,fact=ag,fun=max)
water4=focal(water3,w=wt)
full_area=stack(demand_all,urban1,P_all1)
full_area2=projectRaster(full_area,res=res(gw_pumping),crs=crs(gw_pumping))
names(full_area2)=c('ET_P','urban','P')
full_area2$urban[is.na(full_area2$urban)]=0
full_area2=aggregate(full_area2,fact=ag,fun=mean,na.rm=T)
full_area2$sw=water4
full_area2$ET_filt=focal(full_area2$ET_P,wt1)
full_area2=mask(full_area2,ks_watershed2)

df.full=as.data.frame(full_area2)
df.full$gw_Q_predict=predict(rf,df.full)
full_area2$gw_predict=df.full$gw_Q_predict
# writeRaster(full_area2,filename = '../gw_predictions/KS_gw_predictions.tif')
save(rf,wt,wt1,file='rf_model.Rda')

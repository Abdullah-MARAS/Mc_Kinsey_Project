setwd("~/Personal/Study/Hackathon/20 Jul 2018 Mckinsey")

library(readxl)

traindata=read.csv("~/Personal/Study/Hackathon/20 Jul 2018 Mckinsey/train_ZoGVYWq.csv")
testdata=read.csv("~/Personal/Study/Hackathon/20 Jul 2018 Mckinsey/test_66516Ee.csv")

FullData=rbind(traindata,testdata)
nrow(traindata)
nrow(testdata)
nrow(FullData)
FullData$sourcing_channel <-factor(FullData$sourcing_channel)
FullData$residence_area_type <-factor(FullData$residence_area_type)

plot(FullData$sourcing_channel)

hist(FullData$age_in_days)
y=dnorm(FullData$age_in_days, mean = mean(FullData$age_in_days), sd = sd(FullData$age_in_days))
plot(FullData$age_in_days,y)
boxplot(FullData$age_in_days)
boxplot.stats(FullData$age_in_days)$out
a=min(boxplot.stats(FullData$age_in_days)$out)
FullData$age_in_days[FullData$age_in_days>=a] <- mean(FullData$age_in_days, na.rm = T)

FullDataU<-FullData[!is.na(FullData$application_underwriting_score)>0,]
FullDataNU<-FullData[is.na(FullData$application_underwriting_score)>0,]
nrow(FullDataU)
nrow(FullData)
cor(FullDataU$Income,FullDataU$application_underwriting_score)
cor.test(FullDataU$age_in_days,FullDataU$application_underwriting_score)
plot(FullDataU$Income,FullDataU$application_underwriting_score)
abline(lm(FullDataU$Income~FullDataU$application_underwriting_score))
#linear regression model
reg1=lm(application_underwriting_score~Income+age_in_days+no_of_premiums_paid+premium+
          perc_premium_paid_by_cash_credit,data=FullDataU)
summary(reg1)
FullDataNU$application_underwriting_score<-predict(reg1,FullDataNU)

FullData=rbind(FullDataU,FullDataNU)
FullData$application_underwriting_score[(FullData$application_underwriting_score)>99.99]<-mean(FullData$application_underwriting_score)

boxplot(FullData$age_in_days)

FullData$Count_3.6_months_late[is.na(FullData$Count_3.6_months_late)]<-0
FullData$Count_6.12_months_late[is.na(FullData$Count_6.12_months_late)]<-0
FullData$Count_more_than_12_months_late[is.na(FullData$Count_more_than_12_months_late)]<-0
View(FullData)

Traindata<-FullData[!is.na(FullData$renewal)>0,]
Testdata<-FullData[is.na(FullData$renewal)>0,]
Testdata<-Testdata[,1:12]
View(Traindata)
View(Testdata)

write.csv(Traindata,'~/Personal/Study/Hackathon/20 Jul 2018 Mckinsey/Traindata.csv',row.names=FALSE)
write.csv(Testdata,'~/Personal/Study/Hackathon/20 Jul 2018 Mckinsey/Testdata.csv',row.names=FALSE)

library(caret)
library(caTools)
require(gbm)

set.seed(999)
split_data=sample.split(Traindata,SplitRatio =0.8)
Traindata_Train=subset(Traindata,split_data=="TRUE")
nrow(Traindata_Train)
Traindata_Test=subset(Traindata,split_data=='FALSE')
nrow(Traindata_Test)

fitControl <- trainControl(method = "repeatedcv", number = 4, repeats = 4)

set.seed(33)
gbmFit1 <- train(as.factor(renewal) ~ perc_premium_paid_by_cash_credit+age_in_days+Income+
                   Count_3.6_months_late+Count_6.12_months_late+
                   Count_more_than_12_months_late+application_underwriting_score+
                   no_of_premiums_paid+sourcing_channel+residence_area_type+premium,
                 data = Traindata_Train, method = "gbm", trControl = fitControl,verbose = FALSE)

gbm_dev <- predict(gbmFit1, Traindata_Test,type= "prob")[,2] 
library(Metrics)
auc(Traindata_Test$renewal,gbm_dev)

testdata=read.csv("~/Personal/Study/Hackathon/20 Jul 2018 Mckinsey/test_66516Ee.csv")

gbm_dev <- predict(gbmFit1, Testdata,type= "prob")[,2] 

a_1 <- predict(gbmFit1, Testdata,type= "prob")[,1] 

a_11=log(a_1)+log(1/20)
ab=(a_11/2)
ab_power_400=ab^400
incentives1_a=log(ab_power_400)

a_11=log(gbm_dev)+log(1/20)
ab=(a_11/2)
ab_power_400=ab^400
incentivesa=log(ab_power_400)


Result <- data.frame(Testdata,Prob_p = gbm_dev,Prob_p_1=a_1,
                     incentives1_a=incentives1_a,incentivesa=incentivesa)
nrow(Result)
write.csv(Result,file ="~/Personal/Study/Hackathon/20 Jul 2018 Mckinsey/Testdata_out.csv",row.names = FALSE)





setwd("D:/Fraud_POC")

#########################Data Preprocessing######################
# load required packages
library(data.table)
library(psych)
library(mvtnorm)
library(caret)
library(PRROC)
library(ggplot2)
library(caTools)
library(pROC)
library(dplyr)
# Read the csv file
creditcard_details <- read.csv('creditcard.csv')
# Print the structure of the dataframe
str(creditcard_details)


#####################Exploratory Data Analysis##########################################

################Finding the Imbalance in Dependent variable###########


# Now we will group the datas based on the Class value and 
# For this we will use dplyr package which contans group by function.

library(dplyr)
creditcard_details$Class <- as.factor(creditcard_details$Class)
creditcardDF <- creditcard_details %>% group_by(Class) %>% summarize(Class_count = n())
print(head(creditcardDF))

table(creditcard_details$Class)

# Finding the percentage of each Class category.
creditcardDF$Class_count <- 100 * creditcardDF$Class_count/nrow(creditcard_details)
creditcardDF$ncount_p <- paste(round(creditcardDF$Class_count,2),"%")
head(creditcardDF)



# Applying ggplot2 to visualize the results for Number of transaction under each Class Status, 
# Class 0 are Good transactions and Class 1 represents Fraud transactions.
# In Xaxis we have Class Status and In yaxis we have Percentage of transaction.

ggplot(creditcardDF,aes(x=Class,y=Class_count,fill=Class)) +
  geom_bar(stat="identity") + geom_text(aes(label=ncount_p),vjust = 2) +
  ggtitle("Transaction by status") + xlab("Class") + ylab("Percentage of transaction")


#####################Transaction by Hour####################

# In this dataset 'Time' variable contains the seconds elapsed between each transaction and the first transaction in the datase. Normalizing the time by day and category them into four quarters according to time of day.
# separate transactions by day
creditcard_details$day <- ifelse(creditcard_details$Time > 3600 * 24, "day2", "day1")

# make transaction relative to day
creditcard_details$Time_day <- ifelse(creditcard_details$day == "day2", creditcard_details$Time - 86400, creditcard_details$Time)

# ggplot to see number of transactions per day
ggplot(creditcard_details) + geom_bar(aes(x=day), color = "blue", fill = "blue")+ theme_bw()


# Categorize the transactions into four quarters according to time of day
creditcard_details$Time_qtr <- as.factor(ifelse(creditcard_details$Time_day <= 21600, "0-6 hr",
                                                ifelse(creditcard_details$Time_day > 21600 & creditcard_details$Time_day <= 43200, "7-12 hr",
                                                       ifelse(creditcard_details$Time_day > 43200 & creditcard_details$Time_day <= 64800, "13-18 hr",
                                                              "19-24 hr"))))
# Here we change the order of factor variable Time_qtr, to plot the below chart in the order of time interval in X-axis
#creditcard_details$Time_qtr <- factor(creditcard_details$Time_qtr, levels = creditcard_details$Time_qtr[order(creditcard_details$Time_day)])

# ggplot to see max number of transactions occured in time qtr(it shows that more fraud gransactions occured at 13-18 hr)
ggplot(creditcard_details) + geom_bar(aes(x = Time_qtr), color = "blue", fill = "blue") + theme_bw() + facet_wrap( ~ Class, scales = "free", ncol = 2) 


#############################Anomaly Detection##################

# Remove the fields created for Exploratory Data Analysis before Anomaly Detection
creditcard_details <- subset(creditcard_details, select = -c(day, Time_day, Time_qtr))


# Exploratory Data Analysis for Anomaly Detection using Mean
rownames(creditcard_details) <- 1:nrow(creditcard_details)
non_anom <- creditcard_details[creditcard_details$Class == 0,]
anomaly <- creditcard_details[creditcard_details$Class == 1,]

mean_non_anom <- apply(non_anom[sample(rownames(non_anom), size = 492), -c(1, 30, 31)], 2, mean)
mean_anomaly <- apply(anomaly[, -c(1, 30, 31)], 2, mean)
plot(mean_anomaly, col = "blue", xlab = "Features", ylab = "Mean")
lines(mean_anomaly, col = "blue", lwd = 2)
points(mean_non_anom, col = "black")
lines(mean_non_anom, col = "black", lwd = 2)
legend("topright", legend = c("Good", "Anomalous"), lty = c(1,1), col = c("black", "blue"), lwd = c(2,2))


######################Splitting the dataset into train and test datasets################
# Splitting the dataset into train and test datasets
#install.packages("caTools")
library(caTools)
set.seed(123)
split = sample.split(creditcard_details$Class, SplitRatio =0.8)
training_set = subset(creditcard_details, split == TRUE)
test_set = subset(creditcard_details, split == FALSE)

#Feature sclaing the fields
training_set[-31] = scale(training_set[-31])
test_set[-31] = scale(test_set[-31])

head(training_set)


#################################Apply model to Imbalanced Data#######################
# Apply Logistic classifier on training set
normal_classifier = glm(formula = Class ~ ., family = binomial, data = training_set)
# Predicting the test set using Under sampling classifier
normal_probability_predict = predict(normal_classifier, type = 'response', newdata = test_set[-31])
y_pred_normal = ifelse(normal_probability_predict>0.5, 1, 0)

# To check the model accuracy using confusionMatrix
confusionMatrix(table(test_set[, 31], y_pred_normal))

# To check the accuracy of this model using ROC curve.
roc_over <- roc.curve(test_set$Class, y_pred_normal, plotit = F)
print(roc_over)



#################################Apply Sample methods to Imbalanced Data#######################

# As the data has less Fraud transactions(less than 1%), we have to apply sample methods to balance the data
# We applied Over, Upper, Mixed(both) and ROSE sampling methods using ROSE package and SMOTE sampling method using DMwR package
#install.packages('ROSE')
#install.packages('DMwR')
library(DMwR)
library(ROSE)

print('Number of transactions in train dataset before applying sampling methods')
print(table(training_set$Class))

# Oversampling, as Fraud transactions(1) are having less occurrence, so this Over sampling method will increase the Fraud records untill matches good records 227452
# Here N= 227452*2
over_sample_train_data <- ovun.sample(Class ~ ., data = training_set, method="over", N=454904)$data
print('Number of transactions in train dataset after applying Over sampling method')
print(table(over_sample_train_data$Class))

# Undersampling,as Fraud transactions(1) are having less occurrence, so this Under sampling method will descrease the Good records untill matches Fraud records, But, you see that we’ve lost significant information from the sample. 
under_sample_train_data <- ovun.sample(Class ~ ., data = training_set, method="under", N=788)$data
print('Number of transactions in train dataset after applying Under sampling method')
print(table(under_sample_train_data$Class))

# Mixed Sampling, apply both under sampling and over sampling on this imbalanced data
both_sample_train_data <- ovun.sample(Class ~ ., data = training_set, method="both", p=0.5, seed=222, N=227846)$data
print('Number of transactions in train dataset after applying Mixed sampling method')
print(table(both_sample_train_data$Class))

# ROSE Sampling, this helps us to generate data synthetically. It generates artificial datas instead of dulicate data.
rose_sample_train_data <- ROSE(Class ~ ., data = training_set,  seed=111)$data
print('Number of transactions in train dataset after applying ROSE sampling method')
print(table(rose_sample_train_data$Class))

# SMOTE(Synthetic Minority Over-sampling Technique) Sampling
# formula - relates how our dependent variable acts based on other independent variable.
# data - input data
# perc.over - controls the size of Minority class
# perc.under - controls the size of Majority class
# since my data has less Majority class, increasing it with 200 and keeping the minority class to 100.
smote_sample_train_data <- SMOTE(Class ~ ., data = training_set, perc.over = 100, perc.under=200)
print('Number of transactions in train dataset after applying SMOTE sampling method')
print(table(smote_sample_train_data$Class))


###################Apply Logistic classifier on balanced data###########################

# Now we have five different types of inputs which are balanced and ready for prediction.
# We can appply Logistic classifier to all these five datasets and calculate the performance of each.

# Logistic classifier for Over sampling dataset
over_classifier = glm(formula = Class ~ ., family = binomial, data = over_sample_train_data)

# Logistic classifier for Under sampling dataset
under_classifier = glm(formula = Class ~ ., family = binomial, data = under_sample_train_data)

# Logistic classifier for Mixed sampling dataset
both_classifier = glm(formula = Class ~ ., family = binomial, data = both_sample_train_data)

#Logistic classifier for ROSE sampling dataset
rose_classifier = glm(formula = Class ~ ., family = binomial, data = rose_sample_train_data)

# Logistic classifier for SMOTE dataset
smote_classifier = glm(formula = Class ~ ., family = binomial, data = smote_sample_train_data)



#########################Prediction on test set#############################

# Prediction on test set using sampling classifiers

# Predicting the test set using Over sampling classifier
over_probability_predict = predict(over_classifier, type = 'response', newdata = test_set[-31])
y_pred_over = ifelse(over_probability_predict>0.5, 1, 0)

# Predicting the test set using Under sampling classifier
under_probability_predict = predict(under_classifier, type = 'response', newdata = test_set[-31])
y_pred_under = ifelse(under_probability_predict>0.5, 1, 0)

# Predicting the test set using Mixed sampling classifier
both_probability_predict = predict(both_classifier, type = 'response', newdata = test_set[-31])
y_pred_both = ifelse(both_probability_predict>0.5, 1, 0)

# Predicting the test set using ROSE classifier
rose_probability_predict = predict(rose_classifier, type = 'response', newdata = test_set[-31])
y_pred_rose = ifelse(rose_probability_predict>0.5, 1, 0)

# Predicting the test set using SMOTE classifier
smote_probability_predict = predict(smote_classifier, type = 'response', newdata = test_set[-31])
y_pred_smote = ifelse(smote_probability_predict>0.5, 1, 0)




############################ROC Curve###########################

# roc.curve function from ROSE package returns the ROC curve and AUC value.
# We can see the AUC value by making the plotit as FALSE and print the curve.
# It takes dependent variable as the first parameter and the class to be evaluated
# plotit is logical for plotting the ROC curve. color of the curve can be given in col.

# ROC curve of over sampling data
roc_over <- roc.curve(test_set$Class, y_pred_over)
print(roc_over)
# ROC curve of Under sampling data
roc_under <- roc.curve(test_set$Class, y_pred_under)
print(roc_under)
# ROC curve of both sampling data
roc_both <- roc.curve(test_set$Class, y_pred_both)
print(roc_both)
# ROC curve of ROSE sampling data
roc_rose <- roc.curve(test_set$Class, y_pred_rose)
print(roc_rose)
# ROC curve of SMOTE sampling data
roc_smote <- roc.curve(test_set$Class, y_pred_smote)
print(roc_smote)







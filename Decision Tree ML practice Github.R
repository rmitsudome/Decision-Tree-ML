BreastCancer = read.table("https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/breast-cancer-wisconsin.data",
                          sep = ",")

dim(BreastCancer)
head(BreastCancer)
names(BreastCancer) = c("ID", "Clump_Thick", "Cell_Size", "Cell_Shape"
                        , "Adhesion", "Epithelial_Size", "Nuclei",
                        "Chromatin", "Nucleoli", "Mitoses", "Class")

sum(is.na(BreastCancer))

#delete all ? in data set
Question_row = which(BreastCancer == "?", arr.ind = TRUE)
BreastCancer = BreastCancer[-c(Question_row[1:16,]), ]
nrow(BreastCancer)

#change all predictors to numeric
for(i in 2:10) {
  BreastCancer[, i] <- as.numeric(as.character(BreastCancer[, i]))
}

#make Class out of 0 and 1
BreastCancer$Class <- ifelse(BreastCancer$Class == "4", 1, 0)

#change dependent variable to factor
BreastCancer[,11] = as.factor(BreastCancer[,11])

#need to randomly split the data into training and test samples
#Since response variable is binary categorical variable need to
#make sure training data has approximately = proportion of classes.

table(BreastCancer$Class)

library(caret)
'%ni%' <- Negate('%in%')  # define 'not in' func
options(scipen=999)  # prevents printing scientific notations.


#randomly put 70% of orig. data in train, the rest in test
set.seed(100)
trainIndex = createDataPartition(BreastCancer$Class, 
                                 p=0.7, list = F)  # 70% training data
train = BreastCancer[trainIndex, ]
test = BreastCancer[-trainIndex, ]

table(train$Class) #around 2x of 0 than 1

#Down sampling
#Majority class randomly down sampled to same size as smaller class

set.seed(100)
#Selects all columns but Class for x
#y must be factor variable
down_train <- downSample(x = train[, colnames(train) %ni% "Class"],
                         y = train$Class)
table(down_train$Class)

#Up Sampling
#rows from minority class repeatedly sampled till reaches 
#equal size as majority class

set.seed(100)
up_train <- upSample(x = train[, colnames(train) %ni% "Class"],
                     y = train$Class)
table(up_train$Class)

#for this example, will use down_train as training data
train = down_train
#model decision tree
library("rpart")
dtm = rpart(Class ~ Clump_Thick + Cell_Size + Cell_Shape
            + Adhesion + Epithelial_Size + Nuclei
            + Chromatin + Nucleoli + Mitoses, 
            data = BreastCancer, method = "class")
#plot decision tree
library("rpart.plot")
par(mfrow = (c(1,1)))
rpart.plot(dtm)
p = predict(dtm, test, type = "class")
summary(p) #result from algo
table(test[,11], p) #0.9655 success rate

#prune tree to best Complexity Parameter
bestcp = dtm$cptable[which.min(dtm$cptable[,"xerror"]),"CP"]
pruned = prune(dtm, cp = bestcp)

prp(pruned)

#Scoring
install.packages("ROCR")
library(ROCR)
val1 = predict(pruned, test, type = "prob")
pred_val <-prediction(val1[,2],test$Class)


# Calculating Area under Curve
perf_val <- performance(pred_val,"auc")
perf_val

# Plotting Lift curve
plot(performance(pred_val, measure="lift", x.measure="rpp"), colorize=TRUE)

# Calculating True Positive and False Positive Rate
perf_val <- performance(pred_val, "tpr", "fpr")

# Plot the ROC curve
#closer the ROC curve is to the upper left corner
#the higher the overall accuracy of the test
plot(perf_val, col = "green", lwd = 1.5)

#Calculating KS statistics
#higher the value the better the model is at separating 
#the positive from negative cases.
ks1.tree <- max(attr(perf_val, "y.values")[[1]] - (attr(perf_val, "x.values")[[1]]))
ks1.tree
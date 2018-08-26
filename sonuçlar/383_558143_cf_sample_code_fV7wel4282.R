# Packages
library(xgboost)
library(magrittr)
library(dplyr)
library(Matrix)
library(DiagrammeR)
library(DMwR)

# Data
str(train)
train <- knnImputation(train)
summary(train)

# Partition data
set.seed(1234)
ind <- sample(2, nrow(train), replace = T, prob = c(0.8, 0.2))
traindata <- train[ind==1,]
testdata <- train[ind==2,]

# Create matrix - One-Hot Encoding for Factor variables
trainm <- sparse.model.matrix(traindata$renewal ~ .-1, data = traindata)
head(trainm)
train_label <- traindata[,"renewal"]
train_matrix <- xgb.DMatrix(data = as.matrix(trainm), label = train_label)

testm <- sparse.model.matrix(renewal~.-1, data = testdata)
test_label <- testdata[,"renewal"]
test_matrix <- xgb.DMatrix(data = as.matrix(testm), label = test_label)

# Parameters
nc <- length(unique(train_label))
xgb_params <- list("objective" = "multi:softprob",
                   "eval_metric" = "mlogloss",
                   "num_class" = nc)
watchlist <- list(train = train_matrix, test = test_matrix)

# eXtreme Gradient Boosting Model
bst_model <- xgb.train(params = xgb_params,
                       data = train_matrix,
                       nrounds = 100000,
                       watchlist = watchlist,
                       eta = 0.0001,
                       max.depth = 4,
                       gamma = 0,
                       subsample = 1,
                       colsample_bytree = 1,
                       missing = NA,
                       seed = 333)

# Training & test error plot
e <- data.frame(bst_model$evaluation_log)
plot(e$iter, e$train_mlogloss, col = 'blue')
lines(e$iter, e$test_mlogloss, col = 'red')

min(e$test_mlogloss)
e[e$test_mlogloss == 0.306851,]

# Feature importance
imp <- xgb.importance(colnames(train_matrix), model = bst_model)
print(imp)
xgb.plot.importance(imp)

# Prediction & confusion matrix - test data
p <- predict(bst_model, newdata = test_matrix)
pred <- matrix(p, nrow = nc, ncol = length(p)/nc) %>%
         t() %>%
         data.frame() %>%
         mutate(label = test_label, max_prob = max.col(., "last")-1)
table(Prediction = pred$max_prob, Actual = pred$label)

xgb.plot.tree(model = bst_model)
xgb.plot.tree(model = bst_model, n_first_tree = 1)


test <- knnImputation(test)

testreal <- sparse.model.matrix(~.-1, data = test)
test_label <- testdata[,"renewal"]
test_matrix <- xgb.DMatrix(data = as.matrix(testreal))

preal <- predict(bst_model, newdata = test_matrix)

predreal <- matrix(preal, nrow = nc, ncol = length(preal)/nc) %>%
  t() %>%
  data.frame() %>%
  mutate(label = "renewal", max_prob = max.col(., "last")-1)

test_final <- cbind(test,predreal$X2)

colnames(test_final)[13] <- paste("Propensity")

test_final$incentive <- -400*(log(1+5*log(1-test_final$Propensity/20)))

write.csv(test_final, file = "test_final_2.csv")



# load libraries
library(tidyverse)
library(caret)

# read in data
df <- read_csv('jop_data.csv')

# quick look at dataset 
df %>%
  glimpse()

# pre-processing ------------------------------------------------------------

# look at missing values 
df %>%
  map_df(function(x) sum(is.na(x))) %>%
  gather(var.name, num_nulls) %>%
  arrange(-num_nulls)

# filter rows where voteprival is missing and first column
df <- df %>%
  filter(!is.na(voteprival)) %>%
  select(-X1)

# impute median for inc and drop remaining missing samples 

# select variables to impute 
df.medimpute <- df %>%
  select(inc)

# median impute
medimpute_vars <- preProcess(df.medimpute, method = c("medianImpute"))

# Use predict to transform data
df <- predict(medimpute_vars, df)

# drop remaining missing values 
df <- df %>%
  drop_na()

# drop non-informative and problematic variables
df <- df %>%
  select(-pid3z, -weight)

# one hot encoding for categorical variables
df.onehot <- df %>%
  select(year, race, educ, pid7, state) %>%
  mutate(year = as.factor(year))

# convert year to factor variable type 
df <- df %>%
  mutate(year = as.factor(year))

df.onehot <- dummyVars( ~ ., data = df.onehot, fullRank=T)
df.onehot <- data.frame(predict(df.onehot, df))

# bind cols and drop variables
df<- df %>%
  bind_cols(df.onehot) %>%
  select(-year, -year.2010, -state, -pid7, -race, -educ, -inputstate)

# center and scale numeric variables 
df.normalize <- df %>%
  select(age, days_before_election, inc, libconnew, marginpnew, libcon3)

normalized_vars <- preProcess(df.normalize, method = c('center', 'scale'))

df <- predict(normalized_vars, df)
  
# glimpse at dataframe 
df %>% glimpse()

# baseline h2o model --------------------------------------------------------

# initialize h2o
library(h2o)
h2o.init()

# convert dataframe to h2o type
train_data <- as.h2o(df)

# create target and learning variables
y <- "voteprival"
x <- setdiff(colnames(train_data), y)

# set y to factor
train_data[, y] <- as.factor(train_data[, y])

# test and train split
sframe <- h2o.splitFrame(train_data, ratios=0.80, seed = 42)
train <- sframe[[1]]
valid <- sframe[[2]]

# automatic ml based on cross validation
automl_model <- h2o.automl(x = x, 
                           y = y,
                           training_frame = train,
                           max_models = 30,
                           sort_metric = 'logloss',
                           nfolds = 3,
                           seed = 42)

# view leaderboard 
lb <- automl_model@leaderboard
head(lb)

# predictions with top model
pred_top <- predict(automl_model@leader, valid)

# look at performance of top model
perf <- h2o.performance(automl_model@leader, valid)

# save leaderboard to file 
lb.df <- as.data.frame(lb)
write.csv(lb.df, file = 'leaderboard_h2o_9_8_19.csv')

# most important features of xgboost model 

# get model ids for all models in the AutoML Leaderboard
model_ids <- as.data.frame(automl_model@leaderboard$model_id)[,1]

# grab the top model
top_model <- h2o.getModel(grep("XGBoost_1_AutoML_20190908_081747", 
                               model_ids, value = TRUE)[1])

# variable importance
h2o.varimp_plot(top_model)

# model without votegenval ----------------------------------------------------

df2 <- df %>%
  select(-votegenval)

# convert dataframe to h2o type
train_data <- as.h2o(df2)

# create target and learning variables
y <- "voteprival"
x <- setdiff(colnames(train_data), y)

# set y to factor
train_data[, y] <- as.factor(train_data[, y])

# test and train split
sframe <- h2o.splitFrame(train_data, ratios=0.80, seed = 42)
train <- sframe[[1]]
valid <- sframe[[2]]

# automatic ml based on cross validation
automl_model <- h2o.automl(x = x, 
                           y = y,
                           training_frame = train,
                           max_models = 30,
                           sort_metric = 'logloss',
                           nfolds = 3,
                           seed = 43)

# view leaderboard 
lb2 <- automl_model@leaderboard
head(lb2)

# predictions with top model
pred_top2 <- predict(automl_model@leader, valid)

# look at performance of top model
perf2 <- h2o.performance(automl_model@leader, valid)

# save leaderboard to file 
lb.df2 <- as.data.frame(lb2)
write.csv(lb.df2, file = 'leaderboard_h2o_9_8_19(2).csv')

# most important features of xgboost model 

# get model ids for all models in the AutoML Leaderboard
model_ids2 <- as.data.frame(automl_model@leaderboard$model_id)[,1]

# grab the top model
top_model2 <- h2o.getModel(grep("XGBoost_1_AutoML_20190908_094208", 
                                model_ids2, value = TRUE)[1])

# variable importance
h2o.varimp_plot(top_model2)


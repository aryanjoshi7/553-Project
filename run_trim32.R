library(devtools)
devtools::load_all("/Users/micahwilliamson/code/EECS553/553-Project/rqpen")

norm_dat <- function(features, y){
    avg <- colMeans(features)
    std_dev <- apply(features, 2, sd)
    normal_features <- sweep(sweep(features, 2, avg), 2, std_dev, "/")

    log_fat <- log(y)
    avg_log_fat <- mean(log_fat)
    std_log_fat <- sd(log_fat)
    normal_fat <- (log_fat - avg_log_fat) / std_log_fat

    list(features = normal_features, fat = normal_fat)
}


applyPCA <- function(X_train, X_test) {
  # Step 1: PCA with 30 components
  pca <- prcomp(X_train, center = TRUE, scale. = TRUE, rank. = 30)

  # Step 2: Transform train and test data
  X_train_pca <- predict(pca, X_train)
  X_test_pca <- predict(pca, X_test)

  # Step 3: Scale and center PCs
  X_train_scaled <- scale(X_train_pca)
  # Use mean and std from training set to transform test set
  center_vals <- attr(X_train_scaled, "scaled:center")
  scale_vals <- attr(X_train_scaled, "scaled:scale")
  X_test_scaled <- scale(X_test_pca, center = center_vals, scale = scale_vals)

  list(X_train = X_train_scaled, X_test = X_test_scaled)
}

load("meatspec.rda")
X <- meatspec[, -ncol(meatspec)]  # All columns except the last one (features)
y <- meatspec[, ncol(meatspec)]   # The last column (labels)
n <- nrow(meatspec)
normed_data <- norm_dat(X, y)
X_normed <- normed_data$features
y_normed <- normed_data$fat

train_size <- round(0.8 * n)
train_index <- sample(1:n, size = train_size)

X_train <- X_normed[train_index, ]
X_test  <- X_normed[-train_index, ]
Y_train <- y_normed[train_index]
Y_test  <- y_normed[-train_index]

pca_results <- applyPCA(X_train, X_test)
X_train_pca <- pca_results$X_train
X_test_pca <- pca_results$X_test

fit3 <- rq.group.pen(X_train_pca,Y_train,penalty="gSCAD")
print(coefficients(fit3)[-1])
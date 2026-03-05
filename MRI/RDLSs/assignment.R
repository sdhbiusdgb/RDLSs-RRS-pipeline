# =========================================================
# RDLS deployable assignment + stability + concordance check
# =========================================================

suppressPackageStartupMessages({
    library(ConsensusClusterPlus)
    library(mclust) # adjustedRandIndex
    library(clue) # solve_LSAP (Hungarian algorithm)
})

set.seed(1234)

# -----------------------------
# User-configurable parameters
# -----------------------------
K_FINAL <- 3
REPS <- 500
PITEM <- 0.8
PFEATURE <- 1

TRAIN_GROUPS_PATH <- "./Data/Group/Training.Groups.rds"
TRAIN_FEATURES_PATH <- "./Data/Exprs/Training.DL.r3d50_KM.rds"

VAL_FEATURES_PATH <- "./Data/Exprs/VALIDATION.DL.r3d50_KM.rds" # <- 改成你的验证队列特征文件
OUT_DIR <- "./Data/RDLS"
dir.create(OUT_DIR, showWarnings = FALSE, recursive = TRUE)

# -----------------------------
# Helpers
# -----------------------------
# Ensure matrix is in features x samples for ConsensusClusterPlus
to_features_by_samples <- function(M) {
    M <- as.matrix(M)
    # typical: features ~ 1867, samples usually smaller than features
    # if rows < cols, likely samples x features -> transpose
    if (nrow(M) < ncol(M)) {
        return(t(M))
    }
    return(M)
}

# Centroids: input X_sf is samples x features, y length = samples
get_centroids <- function(X_sf, y) {
    labs <- sort(unique(as.integer(y)))
    centroids <- lapply(labs, function(j) colMeans(X_sf[y == j, , drop = FALSE]))
    C <- do.call(rbind, centroids)
    rownames(C) <- paste0("RDLS", labs)
    return(C) # K x features
}

# Nearest centroid assignment (Euclidean; squared distances OK)
assign_nearest_centroid <- function(X_sf, C_kf) {
    # returns labels in 1..K corresponding to rows of C_kf
    d <- sapply(1:nrow(C_kf), function(j) {
        rowSums((X_sf - matrix(C_kf[j, ], nrow = nrow(X_sf), ncol = ncol(X_sf), byrow = TRUE))^2)
    })
    # d: n x K
    apply(d, 1, which.min)
}

# Confidence margin: d2 - d1
assignment_margin <- function(X_sf, C_kf) {
    d <- sapply(1:nrow(C_kf), function(j) {
        rowSums((X_sf - matrix(C_kf[j, ], nrow = nrow(X_sf), ncol = ncol(X_sf), byrow = TRUE))^2)
    })
    ds <- t(apply(d, 1, sort))
    margin <- ds[, 2] - ds[, 1]
    return(margin)
}

# Hungarian matching based on centroid distances
match_labels_by_centroids <- function(C_from, C_to) {
    # C_from: K x p (e.g., validation de novo centroids)
    # C_to:   K x p (e.g., training centroids)
    # returns perm where perm[i] = matched label in "to" for label i in "from"
    K <- nrow(C_from)
    D <- as.matrix(dist(rbind(C_from, C_to)))
    D <- D[1:K, (K + 1):(2 * K)]
    perm <- solve_LSAP(D)
    as.integer(perm)
}

# -----------------------------
# 1) Load training data
# -----------------------------
train_groups <- readRDS(TRAIN_GROUPS_PATH) # not used directly here but keep for your pipeline
train_raw <- readRDS(TRAIN_FEATURES_PATH)

X_train_fx <- to_features_by_samples(train_raw) # features x samples
# ConsensusClusterPlus will cluster samples (columns) using features (rows)
# It expects a numeric matrix; your input seems OK as you used it already.

# -----------------------------
# 2) Train: consensus clustering
# -----------------------------
con_train <- ConsensusClusterPlus(
    X_train_fx,
    maxK = 10,
    reps = REPS,
    pItem = PITEM,
    pFeature = PFEATURE,
    title = file.path(OUT_DIR, "ConsensusClusterPlus.TRAIN"),
    clusterAlg = "km",
    distance = "euclidean",
    seed = 1234,
    plot = "pdf",
    writeTable = TRUE
)

rdls_train <- con_train[[K_FINAL]]$consensusClass # length = n_samples_train
rdls_train <- as.integer(rdls_train)

# convert to samples x features for centroid math
X_train_sf <- t(X_train_fx)

C_train <- get_centroids(X_train_sf, rdls_train)

saveRDS(
    list(
        rdls_train = rdls_train,
        centroids_train = C_train
    ),
    file = file.path(OUT_DIR, "TRAIN_RDLS_k3_centroids.rds")
)

write.csv(
    data.frame(sample_id = colnames(X_train_fx), rdls_train = rdls_train),
    file = file.path(OUT_DIR, "TRAIN_RDLS_k3_labels.csv"),
    row.names = FALSE
)

# -----------------------------
# 3) Load validation data
# -----------------------------
val_raw <- readRDS(VAL_FEATURES_PATH)
X_val_fx <- to_features_by_samples(val_raw)
X_val_sf <- t(X_val_fx) # samples x features

# sanity: feature dimension must match
stopifnot(ncol(X_val_sf) == ncol(X_train_sf))

# -----------------------------
# 4) Deployable assignment: nearest training centroid
# -----------------------------
rdls_val_assigned <- assign_nearest_centroid(X_val_sf, C_train)
margin_val <- assignment_margin(X_val_sf, C_train)

saveRDS(
    list(
        rdls_val_assigned = rdls_val_assigned,
        margin_val = margin_val
    ),
    file = file.path(OUT_DIR, "VALIDATION_RDLS_assigned_by_train_centroids.rds")
)

write.csv(
    data.frame(sample_id = colnames(X_val_fx), rdls_assigned = rdls_val_assigned, margin = margin_val),
    file = file.path(OUT_DIR, "VALIDATION_RDLS_assigned_by_train_centroids.csv"),
    row.names = FALSE
)

# -----------------------------
# 5) Stability: bootstrap/subsampling in training
# -----------------------------
B <- REPS
ari_vec <- numeric(B)
acc_vec <- numeric(B)

for (b in 1:B) {
    idx <- sample(seq_len(nrow(X_train_sf)), size = floor(PITEM * nrow(X_train_sf)), replace = TRUE)
    C_b <- get_centroids(X_train_sf[idx, , drop = FALSE], rdls_train[idx])
    y_b <- assign_nearest_centroid(X_train_sf, C_b)

    ari_vec[b] <- adjustedRandIndex(rdls_train, y_b)
    acc_vec[b] <- mean(rdls_train == y_b)
}

stability_summary <- data.frame(
    metric = c("ARI_median", "ARI_IQR_low", "ARI_IQR_high", "ACC_median", "ACC_IQR_low", "ACC_IQR_high"),
    value = c(
        median(ari_vec), quantile(ari_vec, 0.25), quantile(ari_vec, 0.75),
        median(acc_vec), quantile(acc_vec, 0.25), quantile(acc_vec, 0.75)
    )
)

write.csv(stability_summary, file.path(OUT_DIR, "TRAIN_assignment_stability_summary.csv"), row.names = FALSE)

pdf(file.path(OUT_DIR, "TRAIN_assignment_stability_distributions.pdf"), width = 9, height = 4.5)
par(mfrow = c(1, 2))
hist(ari_vec, main = "Stability of RDLS assignment (ARI)", xlab = "ARI")
hist(acc_vec, main = "Stability of RDLS assignment (Agreement)", xlab = "Agreement")
dev.off()

# -----------------------------
# 6) Concordance with de novo clustering in validation (for comparison only)
# -----------------------------
con_val <- ConsensusClusterPlus(
    X_val_fx,
    maxK = 10,
    reps = REPS,
    pItem = PITEM,
    pFeature = PFEATURE,
    title = file.path(OUT_DIR, "ConsensusClusterPlus.VAL_DENOVO"),
    clusterAlg = "km",
    distance = "euclidean",
    seed = 1234,
    plot = "pdf",
    writeTable = TRUE
)

rdls_val_denovo <- as.integer(con_val[[K_FINAL]]$consensusClass)

# label matching using centroid distances
C_val_denovo <- get_centroids(X_val_sf, rdls_val_denovo)
perm <- match_labels_by_centroids(C_val_denovo, C_train) # perm[val_label] -> train_label
rdls_val_denovo_matched <- perm[rdls_val_denovo]

ARI_val <- adjustedRandIndex(rdls_val_assigned, rdls_val_denovo_matched)
ACC_val <- mean(rdls_val_assigned == rdls_val_denovo_matched)

concordance_summary <- data.frame(
    metric = c("ARI_assigned_vs_denovo_matched", "ACC_assigned_vs_denovo_matched"),
    value = c(ARI_val, ACC_val)
)

write.csv(concordance_summary, file.path(OUT_DIR, "VAL_concordance_assigned_vs_denovo_matched.csv"), row.names = FALSE)

conf_mat <- table(Assigned = rdls_val_assigned, DenovoMatched = rdls_val_denovo_matched)
write.csv(as.data.frame.matrix(conf_mat), file.path(OUT_DIR, "VAL_confusion_assigned_vs_denovo_matched.csv"))

# -----------------------------
# 7) Console outputs
# -----------------------------
cat("\n==== TRAIN: subtype counts ====\n")
print(table(rdls_train))

cat("\n==== VALIDATION: assigned subtype counts ====\n")
print(table(rdls_val_assigned))

cat("\n==== TRAIN: stability summary ====\n")
print(stability_summary)

cat("\n==== VALIDATION: concordance with de novo (matched) ====\n")
print(concordance_summary)

cat("\nAll outputs saved to: ", OUT_DIR, "\n")

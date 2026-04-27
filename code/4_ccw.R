#!/usr/bin/env Rscript
# List of required packages
packages <- c("tidyverse", "pscl", "ggplot2", "dplyr", "openxlsx", "tibble", "cobalt", "this.path")

# Install any packages that are not yet installed
installed <- packages %in% rownames(installed.packages())
if (any(!installed)) {
  install.packages(packages[!installed])
}

# Load all libraries
library(tidyverse)
library(pscl)
library(ggplot2)
library(dplyr)
library(openxlsx)
library(tibble)
library(cobalt)
library(this.path)

# Open config file for file path:
# File paths
setwd(dirname(this.path()))
work_dir <- normalizePath("..")
output_folder <- file.path(work_dir, "output")

data <- read.csv(file.path(output_folder,"intermediate","block_and_time_bins_for_stats.csv"))
unique(data$pt_pre24_IMV)
colnames(data)

df <- subset(data, select = -c(1, 2, 10, 13, 14, 20, 32, 35))
df <- df[order(df$encounter_block, df$time_bin), ]

## Identify the first row where pt_order == 1 within each encounter
pt_rows <- which(df$pt_order == 1)
pt1_idx <- pt_rows[!duplicated(df$encounter_block[pt_rows])]

## Map each encounter_block to its first PT time_bin
pt_enc <- df$encounter_block[pt1_idx]
pt_bin <- df$time_bin[pt1_idx]

## For every row, attach the first PT time_bin for that encounter
## If no PT within observed bins, this will be NA
df$first_pt_bin <- pt_bin[match(df$encounter_block, pt_enc)]

## Clone N: censored from first PT bin onward = PT_order
## Clone E: censored only on the final bin if no PT occurred within 48h
#df$PT_censor_N <- ifelse(!is.na(df$first_pt_bin) & df$time_bin >= df$first_pt_bin, 1, 0)
df$PT_censor_N <- df$pt_order
df$PT_censor_E <- ifelse(is.na(df$first_pt_bin) & df$bin_end == 48, 1, 0)

df1 <- subset(df, select = -c(9, 10, 11, 12, 13, 14, 15, 19, 20))

df1$race_category[df1$race_category == ""] <- NA
df1$ethnicity_category[df1$ethnicity_category == ""] <- NA
df1$language_category[df1$language_category == ""] <- NA
df1$ICU_type[df1$ICU_type == ""] <- NA

fac_vars <- c("sex_category", "race_category", "ethnicity_category",
              "language_category", "ICU_type")
for (v in fac_vars) { df1[[v]] <- as.factor(df1[[v]]) }

df1$time_bin_f <- factor(df1$time_bin)

## For clone N, keep rows through first PT bin (inclusive), or all rows if no PT
keep_N <- is.na(df1$first_pt_bin) | df1$time_bin <= df1$first_pt_bin
df_N <- df1[keep_N, ]
colnames(df_N)
df_N <- subset(df_N, select=-c(19, 21))

df_E <- subset(df1, select=-c(19, 20))

#rm(data)
#rm(df)

#### Censoring Model ####
# Baseline covariates
base_vars <- c("age", "sex_category", "race_category", "ethnicity_category",
               "weight_kg", "language_category", "ICU_type")

# Time-varying covariates
tv_vars <- c("heart_rate_mean", "map_mean", "fio2_set_mean", "peep_set_mean", 
             "RASS_min","pressor_flag", "paralytics_flag")

##### Clone N #####
df_N$PT_uncensor_N <- 1-df_N$PT_censor_N
vars_all_N <- c("PT_uncensor_N", "time_bin_f", base_vars, tv_vars)
df_N <- df_N[complete.cases(df_N[, vars_all_N]), ]
df_N <- df_N[order(df_N$encounter_block, df_N$time_bin), ]

colSums(is.na(df_N))


##### Clone E #####
df_E$PT_uncensor_E <- 1-df_E$PT_censor_E
vars_all_E <- c("PT_uncensor_E", "time_bin_f", base_vars, tv_vars)
df_E <- df_E[complete.cases(df_E[, vars_all_E]), ]
df_E <- df_E[order(df_E$encounter_block, df_E$time_bin), ]

colSums(is.na(df_E))

form_num_E <- as.formula(
  paste("PT_uncensor_E", "~", "time_bin_f", "+", paste(base_vars, collapse = " + "))
)

## denominator model: baseline + time + time-varying
form_den_E <- as.formula(
  paste("PT_uncensor_E", "~", "time_bin_f", "+",
        paste(c(base_vars, tv_vars), collapse = " + "))
)

form_num_N <- as.formula(
  paste("PT_uncensor_N", "~", "time_bin_f", "+", paste(base_vars, collapse = " + "))
)

## denominator model: baseline + time + time-varying
form_den_N <- as.formula(
  paste("PT_uncensor_N", "~", "time_bin_f", "+",
        paste(c(base_vars, tv_vars), collapse = " + "))
)

fit_num_N <- glm(form_num_N, data = df_N, family = binomial())
fit_den_N <- glm(form_den_N, data = df_N, family = binomial())
summary(fit_num_N)
summary(fit_den_N)

fit_num_E <- glm(form_num_E, data = df_E, family = binomial())
fit_den_E <- glm(form_den_E, data = df_E, family = binomial())
summary(fit_num_E)
summary(fit_den_E)

run_ccw <- function(data, response_var, base_vars, tv_vars, 
                    id_var = "encounter_block", time_factor_var = "time_bin_f") {
  df <- data
  
  ## numerator model: baseline + time
  form_num <- as.formula(
    paste(response_var, "~", time_factor_var, "+", paste(base_vars, collapse = " + "))
  )
  
  ## denominator model: baseline + time + time-varying
  form_den <- as.formula(
    paste(response_var, "~", time_factor_var, "+",
          paste(c(base_vars, tv_vars), collapse = " + "))
  )
  
  fit_num <- glm(form_num, data = df, family = binomial())
  fit_den <- glm(form_den, data = df, family = binomial())
  
  ## predicted probabilities
  df$p_num <- predict(fit_num, newdata = df, type = "response")
  df$p_den <- predict(fit_den, newdata = df, type = "response")
  
  ## one-step stabilized factor
  df$sw_step <- df$p_num / df$p_den
  
  ## cumulative stabilized weight by encounter
  df$SW <- ave(df$sw_step, df[[id_var]], FUN = cumprod)

  list(data = df, fit_num = fit_num, fit_den = fit_den)
}

res_N <- run_ccw(data = df_N, response_var = "PT_uncensor_N", base_vars = base_vars,
                 tv_vars = tv_vars)
df_N$sw_step <- res_N$data$sw_step
df_N$SW_final <- res_N$data$SW

res_E <- run_ccw(data = df_E, response_var = "PT_uncensor_E", base_vars = base_vars,
                 tv_vars = tv_vars)
df_E$sw_step <- res_E$data$sw_step
df_E$SW_final <- res_E$data$SW



#### Outcome Model
block_df <- read.csv(file.path(output_folder,"intermediate","block_for_stats.csv"))
colnames(block_df)

block_df <- block_df[block_df$pt_pre24_IMV=="False", ]
block_df$vent_free_days <- as.integer(block_df$vent_free_days)
block_df$icu_los_days <- as.integer(block_df$icu_los_days)
block_df$is_dead_hosp <- ifelse(block_df$is_dead_hosp == "True", 1, 0)
block_df$is_dead_30 <- ifelse(block_df$is_dead_30 == "True", 1, 0)
block_df$is_dead_365 <- ifelse(block_df$is_dead_365 == "True", 1, 0)

block_df$race_category[block_df$race_category == ""] <- NA
block_df$ethnicity_category[block_df$ethnicity_category == ""] <- NA
block_df$language_category[block_df$language_category == ""] <- NA
block_df$ICU_type[block_df$ICU_type == ""] <- NA

for (v in fac_vars) { block_df[[v]] <- as.factor(block_df[[v]]) }

# Get one final weight per encounter for each clone
#make_final_weight <- function(dat, clone_label) {
#  dat %>%
#    group_by(encounter_block) %>%
#    slice_tail(n = 1) %>%
#    ungroup() %>%
#    select(encounter_block, SW_final) %>%
#    rename(weight = SW_final) %>%
#    mutate(clone = clone_label)
#}

make_final_weight <- function(dat, clone_label, censor_col) {
  dat %>%
    group_by(encounter_block) %>%
    # Grab the final observed time-bin for this clone
    slice_tail(n = 1) %>% 
    # Keep ONLY if they successfully completed the strategy (uncensored)
    filter(!!sym(censor_col) == 0) %>% 
    ungroup() %>%
    select(encounter_block, SW_final) %>%
    rename(weight = SW_final) %>%
    mutate(clone = clone_label)
}

w_E <- make_final_weight(df_E, "E", "PT_censor_E")
w_N <- make_final_weight(df_N, "N", "PT_censor_N")

outcome_df <- bind_rows(
  block_df %>% mutate(clone = "E") %>% left_join(w_E, by = c("encounter_block", "clone")),
  block_df %>% mutate(clone = "N") %>% left_join(w_N, by = c("encounter_block", "clone"))
) %>%
  mutate(clone = factor(clone, levels = c("N", "E"))) 

summary(outcome_df$weight)
sum(is.na(outcome_df$weight))
sum(!is.finite(outcome_df$weight))

# Subset data first
missing_weights <- outcome_df[is.na(outcome_df$weight), ]
tapply(missing_weights$encounter_block, missing_weights$clone, function(x) length(unique(x)))

outcome_df <- outcome_df %>%
  filter(!is.na(weight), is.finite(weight), weight > 0)

#### Baseline Covariate Balance Check ####
# balance table
bal_ccw <- bal.tab(x = outcome_df[, base_vars], treat = outcome_df$clone,
                   weights = outcome_df$weight, method = "weighting",
                   estimand = "ATE", s.d.denom = "pooled", un = TRUE)
print(bal_ccw)

# love plot
p_balance <- love.plot(bal_ccw, stats = "mean.diffs", abs = TRUE, 
                       thresholds = c(m = 0.1), var.order = "unadjusted",
                       stars = "raw", sample.names = c("Unweighted", "Weighted"),
                       title = "Baseline Covariate Balance Before and After IPCW")
print(p_balance)

ggsave(file.path(output_folder,"final","graphs","balance_plot_with_RASS.pdf"), plot = p_balance,
       width = 8, height = 6)

rhs <- paste(c("clone", base_vars), collapse = " + ")
form_base <- as.formula(paste("~ clone +", paste(base_vars, collapse = " + ")))
# Or simplest version: form_base <- ~ clone

##### VFD: Zero-inflated NB #####
fit_vent <- zeroinfl(as.formula(paste("vent_free_days ~", rhs, "| 1")), 
                     data = outcome_df, dist = "negbin", weights = weight)
summary(fit_vent)

##### ICU LOS: Poisson #####
fit_icu_los <- glm(as.formula(paste("icu_los_days ~", rhs)), data = outcome_df,
                   family = poisson(), weights = weight)
summary(fit_icu_los)


##### Hospital mortality: Binary #####
fit_dead_hosp <- glm(as.formula(paste("is_dead_hosp ~", rhs)), data = outcome_df,
                     family = binomial(), weights = weight)
summary(fit_dead_hosp)


##### 30-day mortality: Binary #####
fit_dead_30 <- glm(as.formula(paste("is_dead_30 ~", rhs)), data = outcome_df,
                   family = binomial(), weights = weight)
summary(fit_dead_30)


##### 1-year mortality: Binary #####
fit_dead_365 <- glm(as.formula(paste("is_dead_365 ~", rhs)), data = outcome_df,
                    family = binomial(), weights = weight)
summary(fit_dead_365)


#### Results Organization ####
summary(outcome_df$weight)
table(outcome_df$clone)
quantile(outcome_df$weight, probs = c(0, .01, .05, .25, .5, .75, .95, .99, 1), na.rm = TRUE)

g <- ggplot(outcome_df, aes(x = weight, fill = clone)) + 
  geom_histogram(bins = 60, alpha = 0.5, position = "identity") + 
  theme_bw() + 
  labs(title = "Distribution of final stabilized weights by clone",
       x = "Final stabilized weight", 
       y = "Count",
       fill = "Clone Group")
g
ggsave(file.path(output_folder,"final","graphs","SW_plot_no_RASS.pdf"), plot = g, width = 7, height = 5)

extract_glm_table <- function(fit, model_name) {
  sm <- summary(fit)$coefficients
  out <- as.data.frame(sm)
  out$term <- rownames(out)
  rownames(out) <- NULL
  
  p_col <- grep("Pr\\(", names(out), value = TRUE)
  
  out %>%
    transmute(model = model_name, component = "main", term = term,
              estimate = Estimate, se = `Std. Error`, p_value = .data[[p_col]])
}

extract_zeroinfl_table <- function(fit, model_name) {
  sm <- summary(fit)
  
  count_tab <- as.data.frame(sm$coefficients$count)
  count_tab$term <- rownames(count_tab)
  rownames(count_tab) <- NULL
  
  zero_tab <- as.data.frame(sm$coefficients$zero)
  zero_tab$term <- rownames(zero_tab)
  rownames(zero_tab) <- NULL
  
  out_count <- count_tab %>%
    transmute(model = model_name, component = "count", term = term,
              estimate = Estimate, se = `Std. Error`, p_value = `Pr(>|z|)`)
  out_zero <- zero_tab %>%
    transmute(model = model_name, component = "zero", term = term,
              estimate = Estimate, se = `Std. Error`, p_value = `Pr(>|z|)`)
  
  bind_rows(out_count, out_zero)
}

tab_vfd <- extract_zeroinfl_table(fit_vent, "vent_free_days")
tab_icu_los <- extract_glm_table(fit_icu_los, "icu_los_days")
tab_dead_hosp <- extract_glm_table(fit_dead_hosp, "is_dead_hosp")
tab_dead_30 <- extract_glm_table(fit_dead_30, "is_dead_30")
tab_dead_365 <- extract_glm_table(fit_dead_365, "is_dead_365")


standardized_contrast <- function(fit, data, outcome_name, clone_var = "clone") {
  dE <- data; dN <- data
  
  dE[[clone_var]] <- factor("E", levels = levels(data[[clone_var]]))
  dN[[clone_var]] <- factor("N", levels = levels(data[[clone_var]]))
  
  pred_E <- predict(fit, newdata = dE, type = "response")
  pred_N <- predict(fit, newdata = dN, type = "response")
  
  tibble(
    outcome = outcome_name,
    mean_pred_E = mean(pred_E, na.rm = TRUE),
    mean_pred_N = mean(pred_N, na.rm = TRUE),
    diff_E_minus_N = mean(pred_E, na.rm = TRUE) - mean(pred_N, na.rm = TRUE),
    ratio_E_over_N = mean(pred_E, na.rm = TRUE) / mean(pred_N, na.rm = TRUE)
  )
}

pred_vfd <- standardized_contrast(fit_vent, outcome_df, "vent_free_days")
pred_icu <- standardized_contrast(fit_icu_los, outcome_df, "icu_los_days")
pred_hosp <- standardized_contrast(fit_dead_hosp, outcome_df, "is_dead_hosp")
pred_30 <- standardized_contrast(fit_dead_30, outcome_df, "is_dead_30")
pred_365 <- standardized_contrast(fit_dead_365, outcome_df, "is_dead_365")
pred_contrast_tab <- bind_rows(pred_vfd, pred_icu, pred_hosp, pred_30, pred_365)


wb <- createWorkbook()
addWorksheet(wb, "VFD")
writeData(wb, "VFD", tab_vfd)
addWorksheet(wb, "ICU_LOS")
writeData(wb, "ICU_LOS", tab_icu_los)
addWorksheet(wb, "Hosp")
writeData(wb, "Hosp", tab_dead_hosp)
addWorksheet(wb, "30Day")
writeData(wb, "30Day", tab_dead_30)
addWorksheet(wb, "1Year")
writeData(wb, "1Year", tab_dead_365)
addWorksheet(wb, "Predicted_Contrast")
writeData(wb, "Predicted_Contrast", pred_contrast_tab)
saveWorkbook(wb, file = file.path(output_folder,"final","ccw_results.xlsx"), overwrite = TRUE)


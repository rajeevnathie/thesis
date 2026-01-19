# NeuroLangControl project (EEG) 
# script for analysing group results  
# - picture naming task (RT)
# - picture naming task (coded accuracy)
# - probe performance
# - (in a different log file) WM task

# last updated by XYZheng 22 July 2025


rm(list = ls())
# cat("\f") # clear console
# comment out multiple line: ctrl + shift + c
# run lines: ctrl + enter

# [!!!] install packages if needed
# options(scipen = 999) # present results with decimals
# options(scioen) = 0) # reactivate all scientific notations
library(tidyverse)
#library(dplyr)
library(ggplot2)
library(ggdist)
library(readxl)
#library(data.table) # for setnames
library(lme4) # for the mixed effect models
library(lmerTest) # to get p-value estimations that are not part of the standard lme4 packages
library(multcomp) # for multicomp in lmer
library(car) # for Anova

#---- set up directory ----####
# [!!!] change this to your local directory
# Rajeev: change here
dirs = 'C:/Users/rajee/Documents/Thesis_code/processed_data'
setwd(dirs)

# [001] update participant number here (sub009: technical failure)
subjNr = c(1:8, 10:40) # subj 009 technical error

#---- prepare data file, add accuracy codes----####
all_data_list = list()

for (i in subjNr) {
  
  # subj_folder = paste0(dirs, '/', sprintf("%03d", i))
  
  # Locate main data file
  files = list.files(path = dirs,
                     pattern = paste0('^results_sub', sprintf("%03d", i), '.*\\.csv$'),
                     full.names = TRUE)
  
  if (length(files) > 1) {
    warning(paste("Multiple files found for subject", i, "- skipping"))
    
  } else if (length(files) == 1) {
    
    data_indiv = read.csv(files[1])

    # Locate coding file
    # Rajeev: change here
    filename = paste0(dirs, "/CodingSheet_sub", sprintf("%03d", i), ".xlsx")
    
    if (file.exists(filename)) {
      # Read coding sheet
      coding_data = read_excel(filename) %>%
        dplyr::select(trial, Coded = starts_with("Coded"))
      
      # Merge coding with main data
      data_indiv = merge(data_indiv, coding_data, by = "trial", all.x = TRUE)
      
    } else {
      # Coding sheet missing: assume all correct
      data_indiv$Coded = rep(1, nrow(data_indiv))
      cat("Coding file missing for subject", i, "- assuming all correct responses.\n")
    }
    
    all_data_list[[length(all_data_list) + 1]] = data_indiv
    
  } else {
    warning(paste("No data file found for subject", i))
  }
}

# Combine all participants into one dataset
data = bind_rows(all_data_list)

# check data structure and variables (some examples)
str(data)
View(data)
#table(data$subject_id, data$trial) # 180 trials
#table(data$practice) # try a few other variables
table(data$subject_id, data$probe_response)
table(data$subject_id, data$condition) 
data %>%
  group_by(subject_id) %>%
  summarise(max_trial = max(trial, na.rm = TRUE)) %>%
  arrange(subject_id)
table(data$subject_id, data$Coded)
#check NAs
any(is.na(data$Coded))
sum(is.na(data$Coded))
any(is.na(data$picture_rt))
sum(is.na(data$picture_rt))

# reorder condition
data$condition = factor(data$condition, levels = c("congruent", "neutral", "incongruent")) # reorder conditions

#write.csv(data, file = paste0(dirs, "/preproc_v3/beh_pp40.csv"), row.names = FALSE)

#---- check picture naming performance: accuracy + RT ----####
aggr_byCond = data%>% 
  filter(Coded == 1 | Coded == 0) %>%
  group_by(subject_id, condition) %>% 
  summarize(errRate = 1-mean(Coded))
View(aggr_byCond)

# Rajeev: change here
aggr_byCond_correcttrial = data%>% 
  filter(Coded == 1) %>%
  group_by(subject_id, condition) %>% 
  summarize(mean_RT = mean(picture_rt), mean_RT_CLIP = mean(cos_dis_text_clip), mean_RT_Text = mean(cos_dis_transformer), mean_RT_ELMo = mean(cos_dis_elmo_nonclip) )
View(aggr_byCond_correcttrial)

items_unique = data %>%
  filter(Coded == 1) %>% 
  dplyr::select(words, final_word_NL, condition, cos_dis_text_clip, cos_dis_transformer, cos_dis_elmo_nonclip, transformer_surprisal, elmo_surprisal) %>%
  distinct()
nrow(items_unique)

# plot RT data
items_unique %>%
  ggplot(aes(x = condition, y = elmo_surprisal, fill = condition)) +
  ggdist::stat_halfeye(
    adjust = 0.5, width = 0.6, .width = 0, justification = -0.1, alpha = 0.5
  ) +
  geom_boxplot(
    width = 0.2, outlier.shape = NA, 
  ) +
  geom_jitter(
    position = position_nudge(x = 0.1), alpha = 0.5, size = 1.5
  ) +
  stat_summary(fun=median, geom="line", aes(group=1))  + 
  stat_summary(fun=median, geom="point")+
  coord_cartesian(ylim = c(NA, NA)) + 
  labs(
    
    title = "Raincloud Plot of ELMo model suprisal", y = "ELMo Surprisal", x = "Condition"
  ) +
  theme_minimal() +
  theme(legend.position = "none")




#---- lmer (main task effect) ----------------------####
# as factors
data$subject_id = as.factor(data$subject_id)
data$final_word_NL = as.factor(data$final_word_NL) # in total 60 target pictures
data$condition = as.factor(data$condition)


# for condition (Rajeev)
lmer_RT_simple = data %>% 
  filter(Coded == 1)%>%
  lmer(log(picture_rt) ~ condition +  (1 |subject_id) + (1 |final_word_NL) ,data= ., 
       control = lmerControl(optimizer = "bobyqa"))  
summary(lmer_RT_simple)
Anova(lmer_RT_simple)
temp = glht(lmer_RT_simple, linfct = mcp(condition = "Tukey")) # multiple comparison from package MULTCOMP
summary(temp)
            
# for condition (Rajeev)
# !!! Rajeev: check demean/standardization
lmer_RT_simple_model1 = data %>% 
  filter(Coded == 1)%>%
  lmer(log(picture_rt) ~ scale(cos_dis_text_clip, scale=TRUE) +  (1 |subject_id) + (1 |final_word_NL) ,data= ., 
       control = lmerControl(optimizer = "bobyqa"))  
summary(lmer_RT_simple_model1)
Anova(lmer_RT_simple_model1)

lmer_RT_simple_model2 = data %>% 
  filter(Coded == 1)%>%
  lmer(log(picture_rt) ~ scale(cos_dis_transformer, scale=TRUE) +  (1 |subject_id) + (1 |final_word_NL) ,data= ., 
       control = lmerControl(optimizer = "bobyqa"))  
summary(lmer_RT_simple_model2)

lmer_RT_simple_model3 = data %>% 
  filter(Coded == 1)%>%
  lmer(log(picture_rt) ~ scale(cos_dis_elmo_nonclip, scale=TRUE) +  (1 |subject_id) + (1 |final_word_NL) ,data= ., 
       control = lmerControl(optimizer = "bobyqa"))  
summary(lmer_RT_simple_model3)

# ?
#AIC(lmer_RT_simple_model1, lmer_RT_simple_model2)
BIC(lmer_RT_simple_model1, lmer_RT_simple_model2)
BIC(lmer_RT_simple_model1, lmer_RT_simple_model3)
BIC(lmer_RT_simple_model2, lmer_RT_simple_model3)


#BIC(lmer_RT_simple_model1)

# maybe not ,,, hmmm should not
# (1) condition has three levels --> anova?
# (2) correlated predictors compete for shared variance --> consider orthogonalization
lmer_RT_simple_model2_clip = data %>% 
  filter(Coded == 1)%>%
  lmer(log(picture_rt) ~ scale(cos_dis_transformer, scale=TRUE) + condition + (1 |subject_id) + (1 |final_word_NL) ,data= ., 
       control = lmerControl(optimizer = "bobyqa"))  
summary(lmer_RT_simple_model2_clip)
Anova(lmer_RT_simple_model2_clip)
#anova(lmer_RT_simple_model2_clip)

lmer_RT_simple_model2_nonclip = data %>% 
  filter(Coded == 1)%>%
  lmer(log(picture_rt) ~ scale(cos_dis_text_nonclip, scale=TRUE) + demean(cos_dist_trans_clip) + (1 |subject_id) + (1 |final_word_NL) ,data= ., 
       control = lmerControl(optimizer = "bobyqa"))  
summary(lmer_RT_simple_model2_nonclip)
Anova(lmer_RT_simple_model2_nonclip)

item_summary <- data %>%
  filter(Coded == 1) %>%
  group_by(words, final_word_NL, cos_dis_text_clip, cos_dis_transformer, cos_dis_elmo_nonclip, transformer_surprisal, elmo_surprisal) %>% 
  summarize(
    mean_RT = mean(picture_rt, na.rm = TRUE),
    .groups = "drop"
  )

item_summary %>%
  ggplot(aes(x = cos_dis_transformer, y = transformer_surprisal)) +
  
  geom_point(alpha = 0.6, size = 2, color = "darkblue") +
  
  geom_smooth(method = "lm", color = "red", fill = "pink") +
  
  labs(
    title = "Correlation: Transformer cosine distance vs. Transformer surprisal",
    subtitle = "Each dot one item (averaged across participants)",
    x = "Transformer cosine distance",
    y = "Transformer surprisal"
  ) +
  theme_minimal()

spearman_result <- cor.test(
  item_summary$mean_RT, 
  item_summary$cos_dis_transformer, 
  method = "spearman"
)

# Print the full results
print(spearman_result)

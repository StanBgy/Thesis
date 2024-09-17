library('BayesCircIsotropy')
library('R.matlab')
library('REdaS')
library(tidyr)
library(dplyr)

# Computes the bayes factor in favor of the alternative hypothesis of von Mises
# distributed data.
# This version uses a conjugate prior.
# conj_prior has mu0, R0, and c0, in that order.

# prior 12 as in the article
# might slightly prefer Ha (von Mises) compared to prior 13 which slightly prefers H0 (uniform)
# The authors choose prior 12 in their example (with n = 15 and low expected 
# concentration), so I did that too here
mu0 = NA
R0 = 0
c0 = 1
conj_prior = c(mu0, R0, c0)


library(readr)
data <- read_csv("angle_roi_unfinished.csv")

rownames(data) <- data[[1]]
data <- data[ , -1]

data <- data %>% pivot_longer(cols = everything(), names_to = "map", values_to = "angles")

# View the result
View(data)


# data <- data.frame(Value = data)
data$angles = deg2rad(data$angles)

View(data)

names = unique(data$map)
BF = c()
pH0 = c()
pHa = c()
rois = c()

count = 0

for (name in names){
  count = count + 1
  
  th = data$angles[data$map == name]
  th = th[!is.nan(th)]
  
  if (length(th) == 0 || length(th) == 1){
    next
  }
  
  stat = computeIsotropyBFHaVMConj(th,conj_prior,kappaMax = 20)
  BF = c(BF, stat$BF)
  pH0 = c(pH0, stat$pH0)
  pHa = c(pHa, stat$pHa)
  rois = c(rois, name)
  
}


df <- data.frame(rois, pH0, pHa, BF)
write.csv(df, file="BF_angluar_test_preferedxy.csv")
#save(BF,conj_prior,rois,pH0,pHa,file="[save_dir]BF_wc4_x0x0.Rda")
#write.csv(df, "[save_dir]BF_wc4_x0x0.csv")
#save(BF,conj_prior,rois,pH0,pHa,file="[save_dir]BF_wc4_x0y0.Rda")
#write.csv(df, "[save_dir]BF_wc4_x0y0.csv")
#save(BF,conj_prior,rois,pH0,pHa,file="[save_dir]BF_wc2_x0x0.Rda")
#write.csv(df, "[save_dir]BF_wc2_x0x0.csv")
#save(BF,conj_prior,rois,pH0,pHa,file="[save_dir]BF_wc2_x0y0.Rda")
#write.csv(df, "[save_dir]BF_wc2_x0y0.csv")
#save(BF,conj_prior,rois,pH0,pHa,file="[save_dir]BF_wc4_x0x0_leftRight.Rda")
#write.csv(df, "[save_dir]BF_wc4_x0x0_leftRight.csv")
#save(BF,conj_prior,rois,pH0,pHa,file="[save_dir]BF_wc4_x0y0_leftRight.Rda")
#write.csv(df, "[save_dir]BF_wc4_x0y0_leftRight.csv")
#save(BF,conj_prior,rois,pH0,pHa,file="[save_dir]BF_wc4_x0x0_rm_timing.Rda")
#write.csv(df, "[save_dir]BF_wc4_x0x0_rm_timing.csv")
#save(BF,conj_prior,rois,pH0,pHa,file="[save_dir]BF_wc4_y0y0_rm_timing.Rda")
#write.csv(df, "[save_dir]BF_wc4_y0y0_rm_timing.csv")
#save(BF,conj_prior,rois,pH0,pHa,file="[save_dir]BF_wc4_x0x0_rm_numerosity.Rda")
#write.csv(df, "[save_dir]BF_wc4_x0x0_rm_numerosity.csv")
#save(BF,conj_prior,rois,pH0,pHa,file="C:/Users/Hendr076/OneDrive - Universiteit Utrecht/PhD/articles/numTiming/writings/near_submission/BF_wc4_x0x0_numerosityHalves.Rda")
#write.csv(df, "C:/Users/Hendr076/OneDrive - Universiteit Utrecht/PhD/articles/numTiming/writings/near_submission/BF_wc4_x0x0_numerosityHalves.csv")
#save(BF,conj_prior,rois,pH0,pHa,file="C:/Users/Hendr076/OneDrive - Universiteit Utrecht/PhD/articles/numTiming/writings/near_submission/BF_wc4_x0y0_numerosityHalves.Rda")
#write.csv(df, "C:/Users/Hendr076/OneDrive - Universiteit Utrecht/PhD/articles/numTiming/writings/near_submission/BF_wc4_x0y0_numerosityHalves.csv")
#save(BF,conj_prior,rois,pH0,pHa,file="C:/Users/Hendr076/OneDrive - Universiteit Utrecht/PhD/articles/numTiming/writings/near_submission/BF_wc4_x0x0_timingHalves.Rda")
#write.csv(df, "C:/Users/Hendr076/OneDrive - Universiteit Utrecht/PhD/articles/numTiming/writings/near_submission/BF_wc4_x0x0_timingHalves.csv")
#save(BF,conj_prior,rois,pH0,pHa,file="C:/Users/Hendr076/OneDrive - Universiteit Utrecht/PhD/articles/numTiming/writings/near_submission/BF_wc4_x0y0_timingHalves.Rda")
#write.csv(df, "C:/Users/Hendr076/OneDrive - Universiteit Utrecht/PhD/articles/numTiming/writings/near_submission/BF_wc4_x0y0_timingHalves.csv")
#save(BF,conj_prior,rois,pH0,pHa,file="C:/Users/Hendr076/OneDrive - Universiteit Utrecht/PhD/articles/numTiming/writings/near_submission/BF_wc5_x0x0_rm_numerosity.Rda")
#write.csv(df, "C:/Users/Hendr076/OneDrive - Universiteit Utrecht/PhD/articles/numTiming/writings/near_submission/BF_wc5_x0x0_rm_numerosity.csv")
#save(BF,conj_prior,rois,pH0,pHa,file="C:/Users/Hendr076/OneDrive - Universiteit Utrecht/PhD/articles/numTiming/writings/near_submission/BF_wc5_x0x0_rm_timing.Rda")
#write.csv(df, "C:/Users/Hendr076/OneDrive - Universiteit Utrecht/PhD/articles/numTiming/writings/near_submission/BF_wc5_x0x0_rm_timing.csv")
#save(BF,conj_prior,rois,pH0,pHa,file="C:/Users/Hendr076/OneDrive - Universiteit Utrecht/PhD/articles/numTiming/writings/near_submission/BF_wc5_y0y0_rm_timing.Rda")
#write.csv(df, "C:/Users/Hendr076/OneDrive - Universiteit Utrecht/PhD/articles/numTiming/writings/near_submission/BF_wc5_y0y0_rm_timing.csv")
#save(BF,conj_prior,rois,pH0,pHa,file="C:/Users/Hendr076/OneDrive - Universiteit Utrecht/PhD/articles/numTiming/writings/near_submission/BF_wc5_x0x0_numerosityHalves.Rda")
#write.csv(df, "C:/Users/Hendr076/OneDrive - Universiteit Utrecht/PhD/articles/numTiming/writings/near_submission/BF_wc5_x0x0_numerosityHalves.csv")
#save(BF,conj_prior,rois,pH0,pHa,file="C:/Users/Hendr076/OneDrive - Universiteit Utrecht/PhD/articles/numTiming/writings/near_submission/BF_wc5_x0y0_numerosityHalves.Rda")
#write.csv(df, "C:/Users/Hendr076/OneDrive - Universiteit Utrecht/PhD/articles/numTiming/writings/near_submission/BF_wc5_x0y0_numerosityHalves.csv")
#save(BF,conj_prior,rois,pH0,pHa,file="C:/Users/Hendr076/OneDrive - Universiteit Utrecht/PhD/articles/numTiming/writings/near_submission/BF_wc5_x0x0_timingHalves.Rda")
#write.csv(df, "C:/Users/Hendr076/OneDrive - Universiteit Utrecht/PhD/articles/numTiming/writings/near_submission/BF_wc5_x0x0_timingHalves.csv")
#save(BF,conj_prior,rois,pH0,pHa,file="C:/Users/Hendr076/OneDrive - Universiteit Utrecht/PhD/articles/numTiming/writings/near_submission/BF_wc5_x0y0_timingHalves.Rda")
#write.csv(df, "C:/Users/Hendr076/OneDrive - Universiteit Utrecht/PhD/articles/numTiming/writings/near_submission/BF_wc5_x0y0_timingHalves.csv")
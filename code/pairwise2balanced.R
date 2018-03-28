library(tidyverse)
library(here)
library(purrr)

inputs = c(
    here("data/training_set_deepchromaOTI_mfcc_sl=4.csv"), 
    here("data/training_set_deepchromaOTI_mfcc_sl=6.csv"), 
    here("data/training_set_deepchromaOTI_mfcc_sl=8.csv"), 
    here("data/training_set_deepchromaOTI_mfcc_sl=10.csv")
)

pairwise_2_balanced = function(i_file){
    dados = read_csv(i_file)    
    set.seed(123)
    balanceados = sample_n(dados, NROW(dados)) %>% 
        mutate(i = 1:n()) %>% 
        transmute(ab_mfcc = if_else(i < NROW(dados)/2, simple_mfcc_similar, simple_mfcc_dissimilar), 
                  ac_mfcc = if_else(i < NROW(dados)/2, simple_mfcc_dissimilar, simple_mfcc_similar), 
                  ab_chroma = if_else(i < NROW(dados)/2, simple_chroma_similar, simple_chroma_dissimilar), 
                  ac_chroma = if_else(i < NROW(dados)/2, simple_chroma_dissimilar, simple_chroma_similar), 
                  ab_gt_ac = if_else(i < NROW(dados)/2, TRUE, FALSE))
    
    balanceados %>% 
        write_csv(paste0(dirname(i_file), "/balanced_", basename(i_file)))
}

inputs %>% 
    walk(pairwise_2_balanced)

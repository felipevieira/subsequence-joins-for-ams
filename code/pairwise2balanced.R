library(tidyverse)
library(here)
library(purrr)

inputs = c(
    here("workspace/subsequence-joins-for-ams/data/threewise-distances_deepchromaOTI_sl10.csv")
)

pairwise_2_balanced = function(i_file){
    pairwise = read_csv(i_file, 
             col_types = "cccddddddddddddl")
    set.seed(123)
    pairwise = pairwise %>% 
        select(mfcc_similar = simple_mfcc_c1, 
               mfcc_dissimilar = simple_mfcc_c2, 
               mfcc_sd = simple_mfcc_c3,
               mfcc_ds = simple_mfcc_c4,
               mfcc_dissimilar_inv = simple_mfcc_c5,
               mfcc_similar_inv = simple_mfcc_c6,
               chroma_similar = simple_chroma_c1, 
               chroma_dissimilar = simple_chroma_c2, 
               chroma_sd = simple_chroma_c3,
               chroma_ds = simple_chroma_c4,
               chroma_dissimilar_inv = simple_chroma_c5,
               chroma_similar_inv = simple_chroma_c6
               ) %>% 
        mutate(mfcc_similar_simetric1 = mfcc_similar + mfcc_similar_inv, 
               mfcc_dissimilar_simetric1 = mfcc_dissimilar + mfcc_dissimilar_inv, 
               mfcc_similar_simetric2 = min(mfcc_similar, mfcc_similar_inv), 
               mfcc_dissimilar_simetric2 = min(mfcc_dissimilar, mfcc_dissimilar_inv), 
               chroma_similar_simetric1 = chroma_similar + chroma_similar_inv, 
               chroma_dissimilar_simetric1 = chroma_dissimilar + chroma_dissimilar_inv, 
               chroma_similar_simetric2 = min(chroma_similar, chroma_similar_inv), 
               chroma_dissimilar_simetric2 = min(chroma_dissimilar, chroma_dissimilar_inv)
               )
    
    balanceados = sample_n(pairwise, NROW(pairwise)) %>% 
        mutate(i = 1:n()) %>% 
        transmute(base_similar_mfcc = if_else(i < NROW(pairwise)/2, mfcc_similar, mfcc_dissimilar), 
                  base_dissimilar_mfcc = if_else(i < NROW(pairwise)/2, mfcc_dissimilar, mfcc_similar), 
                  similar_dissimilar_mfcc = if_else(i < NROW(pairwise)/2, mfcc_sd, mfcc_ds), 
                  dissimilar_similar_mfcc = if_else(i < NROW(pairwise)/2, mfcc_ds, mfcc_sd), 
                  dissimilar_base_mfcc = if_else(i < NROW(pairwise)/2, mfcc_dissimilar_inv, mfcc_similar_inv), 
                  similar_base_mfcc = if_else(i < NROW(pairwise)/2, mfcc_similar_inv, mfcc_dissimilar_inv), 
                  
                  base_similar_chroma = if_else(i < NROW(pairwise)/2, chroma_similar, chroma_dissimilar), 
                  base_dissimilar_chroma = if_else(i < NROW(pairwise)/2, chroma_dissimilar, chroma_similar), 
                  similar_dissimilar_chroma = if_else(i < NROW(pairwise)/2, chroma_sd, chroma_ds), 
                  dissimilar_similar_chroma = if_else(i < NROW(pairwise)/2, chroma_ds, chroma_sd), 
                  dissimilar_base_chroma = if_else(i < NROW(pairwise)/2, chroma_dissimilar_inv, chroma_similar_inv), 
                  similar_base_chroma = if_else(i < NROW(pairwise)/2, chroma_similar_inv, chroma_dissimilar_inv), 
                  # ab_mfcc_simetric1 = if_else(i < NROW(pairwise)/2, mfcc_similar_simetric1, mfcc_dissimilar_simetric1), 
                  # ab_mfcc_simetric2 = if_else(i < NROW(pairwise)/2, mfcc_similar_simetric2, mfcc_dissimilar_simetric2), 
                  # ab_chroma_simetric1 = if_else(i < NROW(pairwise)/2, chroma_similar_simetric1, chroma_dissimilar_simetric1), 
                  # ac_chroma_simetric1 = if_else(i < NROW(pairwise)/2, chroma_dissimilar_simetric1, chroma_similar_simetric1), 
                  # ab_chroma_simetric2 = if_else(i < NROW(pairwise)/2, chroma_similar_simetric2, chroma_dissimilar_simetric2),
                  # ac_chroma_simetric2 = if_else(i < NROW(pairwise)/2, chroma_dissimilar_simetric2, chroma_similar_simetric2),
                  ab_gt_ac = if_else(i < NROW(pairwise)/2, TRUE, FALSE))
    
    balanceados %>% 
        write_csv(paste0(dirname(i_file), "/balanced_", basename(i_file)))
}

inputs %>% 
    walk(pairwise_2_balanced)

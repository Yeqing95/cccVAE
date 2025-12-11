library(CellChat)
library(dplyr)
library(tidyr)
library(purrr)
library(stringr)
library(tibble)

db <- CellChatDB.human
interaction <- db$interaction
colnames(db$cofactor) <- str_replace_all(string = colnames(db$cofactor), pattern = "cofactor", replacement = "cofactor_")

complex_map <- db$complex %>%
  as.data.frame() %>%
  mutate(complex_name = rownames(.)) %>%
  pivot_longer(cols = starts_with("subunit_"), names_to = "subunit", values_to = "gene") %>%
  filter(!is.na(gene), gene != "") %>%
  group_by(complex_name) %>%
  summarize(genes = paste(gene, collapse = ";")) %>%
  deframe()

cofactor_map <- db$cofactor %>%
  as.data.frame() %>%
  mutate(cofactorname = rownames(.)) %>%
  pivot_longer(cols = starts_with("cofactor_"), names_to = "cofactor", values_to = "gene") %>%
  filter(!is.na(gene), gene != "") %>%
  group_by(cofactorname) %>%
  summarize(genes = paste(gene, collapse = ";")) %>%
  deframe()

replace_from_table <- function(x, table = c("complex", "cofactor"), 
                               complex_map = complex_map, cofactor_map = cofactor_map){
  table <- match.arg(table)
  x <- as.character(x)
  if (table == "complex") {
    if (!is.null(complex_map) && x %in% names(complex_map)) {
      return(complex_map[[x]])
    } else {
      return(x)
    }
  }
  if (table == "cofactor") {
    if (!is.null(cofactor_map) && x %in% names(cofactor_map)) {
      return(cofactor_map[[x]])
    } else {
      return(x)
    }
  }
  return(x)
}
  
df <- db$interaction %>%
  select(interaction_name, pathway_name, ligand, receptor, agonist, antagonist, co_A_receptor, co_I_receptor, annotation, version) %>%
  filter(version == "CellChatDB v1") %>%
  mutate(ligand  = sapply(ligand,  replace_from_table, table = "complex", complex_map = complex_map, cofactor_map = cofactor_map),
         receptor = sapply(receptor, replace_from_table, table = "complex", complex_map = complex_map, cofactor_map = cofactor_map),
         agonist = sapply(agonist, replace_from_table, table = "cofactor", complex_map = complex_map, cofactor_map = cofactor_map),
         antagonist = sapply(antagonist, replace_from_table, table = "cofactor", complex_map = complex_map, cofactor_map = cofactor_map),
         co_A_receptor = sapply(co_A_receptor, replace_from_table, table = "cofactor", complex_map = complex_map, cofactor_map = cofactor_map),
         co_I_receptor = sapply(co_I_receptor, replace_from_table, table = "cofactor", complex_map = complex_map, cofactor_map = cofactor_map))

write.csv(x = df, file = "CellChatDB.human.v1.csv", quote = F, row.names = F)





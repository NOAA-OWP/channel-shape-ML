library(hydrofabric)
library(arrow)
library(jsonlite)
library(dplyr)
library(powerjoin)
library(zonal)


p = read_parquet('data/coastal_main_flowlines.parquet')

net = open_dataset('data/conus_net.parquet') %>%
  filter(id %in% p$id) %>%
  select(id, hf_id, areasqkm) %>%
  collect()


xx = fromJSON(
  'data/variables.json') %>%
  filter(Source == 'StreamCat',
         !`Variable name` %in% c('CatAreaSqKm', 'WsAreaSqKm'))

l = list.files('data/scat', full.names = TRUE)
x = list()
for (i in 1:length(l)) {
  x[[i]] = open_dataset(l[i]) %>%
    select(c('COMID', any_of(xx$`Variable name`))) %>%
    filter(COMID %in% net$hf_id) %>%
    collect()
}

x[[length(x) + 1]] = open_dataset(l[1]) %>%
  select('COMID', 'CatAreaSqKm') %>%
  filter(COMID %in% net$hf_id) %>%
  collect()


full = powerjoin::power_full_join(x, by = "COMID") %>%
  rename(hf_id = COMID, s_areasqkm = CatAreaSqKm) %>%
  left_join(net, by = "hf_id")

column_names <- names(full)
print(column_names)


out = aggregate_zones(
  data = full,
  geom = NULL,
  crosswalk = select(full, hf_id, id, areasqkm, s_areasqkm),
  ID = "id"
)

write_parquet(out,"data/texas_mainstems_sc.parquet")
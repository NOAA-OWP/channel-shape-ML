library(RNetCDF)
library(dplyr)
library(arrow)

# Prep --------------------------------------------------------------------

nwis       = open.nc('/Volumes/Transcend/nwm_retro_files/mergednwis.nc')
nwm21      = open.nc('/Volumes/Transcend/nwm_retro_files/nwmv21_nwis.nc')

nwis_count = c(which((
  as.Date("1993-01-01") + 0:length(var.get.nc(nwis, "time"))
) == as.Date("1993-01-01")),
which((
  as.Date("1993-01-01") + 0:length(var.get.nc(nwis, "time"))
) == as.Date("2017-12-31")))


nwm21_count = c(which(
  utcal.nc(
    "minutes since 1970-01-01 00:00:00 UTC",
    var.get.nc(nwm21, "time"),
    type = "s"
  ) == "1993-01-01 00:00:00"
),
which(
  utcal.nc(
    "minutes since 1970-01-01 00:00:00 UTC",
    var.get.nc(nwm21, "time"),
    type = "s"
  ) == "2017-12-31 00:00:00"
))


# Execute -----------------------------------------------------------------

s3 = read_parquet('data/cleaned_training.parquet')
evals = list()

for (i in 1:nrow(s3)) {
  
  nwm21ID   = which(var.get.nc(nwm21, "feature_id")   == s3$comid[i])
  nwisID    = which(var.get.nc(nwis, "feature_id")   == s3$comid[i])
  
  if (!any(length(nwm21ID) == 0, length(nwisID) == 0)) {
    
    x = var.get.nc(
      nwm21,
      "streamflow",
      start = c(nwm21_count[1], nwm21ID),
      count = c(diff(nwm21_count) + 24, 1),
      unpack = TRUE
    )
    
    nwm21_daily = colMeans(matrix(x, 24, byrow = FALSE))
    
    nwis_daily = var.get.nc(
      nwis,
      "streamflow",
      start = c(nwis_count[1], nwisID),
      count = c(diff(nwis_count) + 1, 1),
      unpack = TRUE
    ) * 0.028316846592
    
    df = data.frame(nwis_daily,nwm21_daily)
    
    df[df < 0] = NA
    
    df = df[complete.cases(df), ]
    
    if (nrow(df) > 0){
      evals[[i]] = data.frame(
        comid     = s3$comid[i],
        nwis_min  = fivenum(df$nwis_daily, na.rm = TRUE)[1],
        nwis_25   = fivenum(df$nwis_daily, na.rm = TRUE)[2],
        nwis_50   = fivenum(df$nwis_daily, na.rm = TRUE)[3],
        nwis_75   = fivenum(df$nwis_daily, na.rm = TRUE)[4],
        nwis_max  = fivenum(df$nwis_daily, na.rm = TRUE)[5],
        nwm21_min = fivenum(df$nwm21_daily, na.rm = TRUE)[1],
        nwm21_25  = fivenum(df$nwm21_daily, na.rm = TRUE)[2],
        nwm21_50  = fivenum(df$nwm21_daily, na.rm = TRUE)[3],
        nwm21_75  = fivenum(df$nwm21_daily, na.rm = TRUE)[4],
        nwm21_max = fivenum(df$nwm21_daily, na.rm = TRUE)[5]
      )
    } else {
      evals[[i]] = NULL
    }
  }
  
  message(i, " of ", nrow(s3))
}


bind_rows(evals) %>% 
  arrow::write_parquet("/Users/mjohnson/github/conus-fhg/data/summary_fivnum_AHG.parquet")

library(arrow)

xx = open_dataset('/Volumes/Transcend/nwm21-ff') %>%
  filter(comid %in% s3$feature_id) %>% 
  collect()

yy = bind_rows(evals) %>% 
  left_join(xx) %>% 
  arrow::write_parquet("data/summary_flows_AHG.parquet")


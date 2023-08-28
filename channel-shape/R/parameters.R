pacman::p_load(terra, dplyr, sf, arrow)

r = rast(
  c(
    '/vsicurl/http://hydrology.cee.duke.edu/POLARIS/PROPERTIES/v1.0/vrt/alpha_mean_0_5.vrt',
    '/vsicurl/http://hydrology.cee.duke.edu/POLARIS/PROPERTIES/v1.0/vrt/bd_mean_0_5.vrt',
    '/vsicurl/http://hydrology.cee.duke.edu/POLARIS/PROPERTIES/v1.0/vrt/clay_mean_0_5.vrt',
    '/vsicurl/http://hydrology.cee.duke.edu/POLARIS/PROPERTIES/v1.0/vrt/hb_mean_0_5.vrt',
    '/vsicurl/http://hydrology.cee.duke.edu/POLARIS/PROPERTIES/v1.0/vrt/ksat_mean_0_5.vrt',
    '/vsicurl/http://hydrology.cee.duke.edu/POLARIS/PROPERTIES/v1.0/vrt/lambda_mean_0_5.vrt',
    '/vsicurl/http://hydrology.cee.duke.edu/POLARIS/PROPERTIES/v1.0/vrt/n_mean_0_5.vrt',
    '/vsicurl/http://hydrology.cee.duke.edu/POLARIS/PROPERTIES/v1.0/vrt/om_mean_0_5.vrt',
    '/vsicurl/http://hydrology.cee.duke.edu/POLARIS/PROPERTIES/v1.0/vrt/ph_mean_0_5.vrt',
    '/vsicurl/http://hydrology.cee.duke.edu/POLARIS/PROPERTIES/v1.0/vrt/sand_mean_0_5.vrt',
    '/vsicurl/http://hydrology.cee.duke.edu/POLARIS/PROPERTIES/v1.0/vrt/silt_mean_0_5.vrt',
    '/vsicurl/http://hydrology.cee.duke.edu/POLARIS/PROPERTIES/v1.0/vrt/theta_r_mean_0_5.vrt',
    '/vsicurl/http://hydrology.cee.duke.edu/POLARIS/PROPERTIES/v1.0/vrt/theta_s_mean_0_5.vrt'
  )
)

elev      = rast("/vsicurl/https://prd-tnm.s3.amazonaws.com/StagedProducts/Elevation/13/TIFF/USGS_Seamless_DEM_13.vrt")
landcover = rast('/Volumes/Transcend/nlcd_2019_land_cover_l48_20210604/nlcd_2019_land_cover_l48_20210604.img')


pts = read_parquet('data/fhg_hydroswot.parquet') %>% 
  group_by(siteID) %>% 
  filter(viable, c > 0, between(f, 0, 1), a > 0, between(b, 0, 1), k > 0, between(m, 0, 1)) %>% 
  slice_min(tot_error) %>% 
  filter(!is.na(X)) %>% 
  st_as_sf(coords = c('X', "Y.y"), crs = 4326, remove = FALSE) %>% 
  st_transform(st_crs(r)) %>% 
  ungroup()
  

system.time({ t = extract(r, pts) })
t$siteID = pts$siteID[t$ID]
t$comid = pts$comid[t$ID]
write_parquet(t, "data/polaris_data.parquet")

system.time({ t2 = extract(elev, pts) })
t2 = read_parquet("data/3dep_13_data.parquet")
t2$siteID = pts$siteID[t2$ID]
t2$comid = pts$comid[t2$ID]
write_parquet(t2, "data/3dep_13_data.parquet")

system.time({ t3 = extract(landcover, project(vect(pts), crs(landcover)))})
t3$siteID = pts$siteID[t3$ID]
t3$comid = pts$comid[t3$ID]
write_parquet(t3, "data/nlcd_2019_data.parquet")

st_drop_geometry(pts) %>% 
  rename(Y = Y.x, lng = X, lat = Y.y)  %>% 
  write_parquet("data/training_data.parquet")

summary(xx)

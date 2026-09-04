library(arrow)
library(dplyr)

# Specify the path to the Parquet file
parquet_file_path <- "D:/Lynker/FEMA_HECRAS/bankfull_W_D/deployment/data/ml_exports.parquet"

# Read the Parquet file into a DataFrame
df <- arrow::read_parquet(parquet_file_path)

# Drop the "__index_level_0__" column
df <- df[, -ncol(df)]

# Specify the path for the new Parquet file
new_parquet_file_path <- "D:/Lynker/FEMA_HECRAS/bankfull_W_D/deployment/data/ml_exports_vpu12.parquet"


# Write the modified DataFrame to a new Parquet file
arrow::write_parquet(df, new_parquet_file_path)
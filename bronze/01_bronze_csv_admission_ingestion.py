# Databricks notebook source
# MAGIC %md
# MAGIC # Bronze Ingestion — CSV Admission Feed
# MAGIC Reads raw admission CSV files per active tenant (source_type = 'csv_admission')
# MAGIC from ADLS and lands them as a Delta table in Unity Catalog, unchanged except for
# MAGIC standard bronze metadata columns. No cleansing, no type casting, no masking here —
# MAGIC bronze is the raw/audit copy.

# COMMAND ----------

#gittest test dev to main
from pyspark.sql import functions as F
from functools import reduce

config_table = "healthcare_dev.control.tenant_config"

# ------------------------------------------------------------
# 1. Read active tenant configuration for the CSV admission source
# ------------------------------------------------------------

tenant_configs = (
    spark.table(config_table)
    .filter(
        (F.col("active_flag") == True)
        & (F.col("source_type") == "csv_admission")
    )
    .select(
        "tenant_id",
        "display_name",
        "source_path"
    )
    .collect()
)

if not tenant_configs:
    raise ValueError(
        "No active csv_admission tenants found in "
        f"{config_table}"
    )

# COMMAND ----------

# ------------------------------------------------------------
# 2. Read each tenant's admission CSV(s)
# ------------------------------------------------------------
#
# NOTE: the CSV files themselves already contain a tenant_id column
# (exported that way by the source system). We do NOT trust that as
# the tenant boundary — the control-table tenant_id (driven by which
# folder we were told to read) is authoritative, same as the HL7
# pattern. The file's own tenant_id is preserved as
# source_file_tenant_id so a mismatch (wrong file in wrong tenant's
# folder) is visible as a data-quality signal instead of silently
# trusted.

# Source files have NO header row and are stored with a .txt extension
# (not .csv) — column order below matches the known admissions export
# schema exactly. pathGlobFilter keeps this read scoped to the admission
# export files only, in case other file types ever land in the same folder.

ADMISSION_COLUMN_ORDER = [
    "source_file_tenant_id",
    "mrn",
    "patient_first_name",
    "patient_last_name",
    "date_of_birth",
    "gender",
    "admit_datetime",
    "discharge_datetime",
    "admit_type",
    "department",
    "attending_provider_id",
    "admit_diagnosis_code",
    "discharge_diagnosis_code",
    "discharge_disposition",
    "length_of_stay_days"
]


def read_tenant_admissions(config):

    tenant_id = config["tenant_id"]
    tenant_name = config["display_name"]
    source_path = config["source_path"]

    print(f"Reading tenant: {tenant_id}")
    print(f"Source path: {source_path}")

    raw_df = (
        spark.read
        .format("csv")
        .option("header", "false")
        .option("inferSchema", "false")
        .option("pathGlobFilter", "*.txt")
        .load(source_path)
        .toDF(*ADMISSION_COLUMN_ORDER)
    )

    admission_df = (
        raw_df

        .select(
            F.lit(tenant_id).alias("tenant_id"),
            F.lit(tenant_name).alias("tenant_name"),

            "source_file_tenant_id",

            "mrn",
            "patient_first_name",
            "patient_last_name",
            "date_of_birth",
            "gender",
            "admit_datetime",
            "discharge_datetime",
            "admit_type",
            "department",
            "attending_provider_id",
            "admit_diagnosis_code",
            "discharge_diagnosis_code",
            "discharge_disposition",
            "length_of_stay_days",

            F.col("_metadata.file_path").alias("source_path"),
            F.current_timestamp().alias("ingest_ts")
        )
    )

    return admission_df

# COMMAND ----------

# ------------------------------------------------------------
# 3. Create one DataFrame per active tenant, union together
# ------------------------------------------------------------

tenant_dfs = [
    read_tenant_admissions(config)
    for config in tenant_configs
]

bronze_df = reduce(
    lambda df1, df2: df1.unionByName(
        df2,
        allowMissingColumns=True
    ),
    tenant_dfs
)

# COMMAND ----------

# ------------------------------------------------------------
# 4. Review before writing
# ------------------------------------------------------------

display(
    bronze_df.select(
        "tenant_id",
        "source_file_tenant_id",
        "mrn",
        "patient_first_name",
        "patient_last_name",
        "admit_datetime",
        "source_path"
    )
)

# COMMAND ----------

# Flag any mismatch between control-plane tenant_id and the file's own tenant_id
display(
    bronze_df
    .filter(
        F.upper(F.col("tenant_id")) != F.upper(F.col("source_file_tenant_id"))
    )
    .select(
        "tenant_id",
        "source_file_tenant_id",
        "source_path"
    )
)

# COMMAND ----------

display(
    bronze_df
    .groupBy("tenant_id", "source_path")
    .agg(F.count("*").alias("row_count"))
)

# COMMAND ----------

# ------------------------------------------------------------
# 5. Write to bronze
# ------------------------------------------------------------

bronze_table = "healthcare_dev.bronze.csv_admission_raw"

(
    bronze_df.write
    .format("delta")
    .mode("overwrite")
    .option("overwriteSchema", "true")
    .saveAsTable(bronze_table)
)

print(f"Bronze table written: {bronze_table}")

# COMMAND ----------

# ------------------------------------------------------------
# 6. Verify
# ------------------------------------------------------------

bronze_check_df = spark.table(bronze_table)

bronze_check_df.printSchema()

print("Row count:", bronze_check_df.count())

display(
    bronze_check_df.select(
        "tenant_id",
        "tenant_name",
        "mrn",
        "patient_first_name",
        "patient_last_name",
        "admit_datetime",
        "discharge_datetime",
        "department"
    )
)

# COMMAND ----------

# Duplicate check (same tenant + mrn + admit_datetime landing more than once)
display(
    bronze_check_df
    .groupBy("tenant_id", "mrn", "admit_datetime")
    .count()
    .filter(F.col("count") > 1)
)

# COMMAND ----------

# MAGIC %md
# MAGIC Yes, bronze is good — duplicates in bronze are expected/fine since bronze is meant to be a raw mirror; that gets cleaned up in silver instead. Here's the cell-by-cell summary:
# MAGIC
# MAGIC Cell 0 — Markdown title cell. States the notebook's purpose: bronze ingestion of the CSV admission feed, per active tenant, unchanged except for standard metadata columns; explicitly notes no cleansing/casting/masking happens here.
# MAGIC
# MAGIC Cell 1 — Imports pyspark.sql.functions as F and reduce. Sets config_table = "healthcare_dev.control.tenant_config". Queries that table, filtering to rows where active_flag == True AND source_type == "csv_admission", selects just tenant_id, display_name, source_path, and .collect()s the result into a Python list called tenant_configs. Raises a ValueError and stops execution if that list comes back empty (safety check against a misconfigured control table).
# MAGIC
# MAGIC Cell 2 — Contains inline comments explaining two design decisions: (a) the file's own embedded tenant_id is not trusted as the tenant boundary — the control table's value is authoritative, and the file's value is kept separately as source_file_tenant_id so mismatches are visible; (b) the source files have no header row and a .txt extension, so column names must be supplied manually. Defines ADMISSION_COLUMN_ORDER, a hardcoded list of the 15 real column names in the exact order they appear in the files. Defines the function read_tenant_admissions(config), which: reads source_path as CSV with header=false, inferSchema=false, pathGlobFilter="*.txt"; applies ADMISSION_COLUMN_ORDER via .toDF() since there's no header to infer names from; then selects out a final set of columns — tenant_id and tenant_name from the control-table config (via F.lit()), source_file_tenant_id from the file itself, all 15 admission fields, source_path derived from the UC-safe _metadata.file_path (not the blocked input_file_name()), and ingest_ts via F.current_timestamp().
# MAGIC
# MAGIC Cell 3 — Calls read_tenant_admissions() once per tenant in tenant_configs, producing a list of DataFrames (tenant_dfs). Uses functools.reduce with unionByName(..., allowMissingColumns=True) to stack all three tenants' DataFrames into one combined bronze_df.
# MAGIC
# MAGIC Cell 4 — A review display() call showing tenant_id, source_file_tenant_id, mrn, patient_first_name, patient_last_name, admit_datetime, and source_path from bronze_df, so you can visually sanity-check the data before writing anything.
# MAGIC
# MAGIC Cell 5 — The mismatch-detection check: filters bronze_df to rows where UPPER(tenant_id) != UPPER(source_file_tenant_id), displaying tenant_id, source_file_tenant_id, source_path for any offending rows. Returned zero rows for you, confirming every file's internal tenant_id matches its folder.
# MAGIC
# MAGIC Cell 6 — Groups bronze_df by tenant_id and source_path, counts rows per group (row_count), and displays it — lets you see exactly how many admission records came from each individual file.
# MAGIC
# MAGIC Cell 7 — Sets bronze_table = "healthcare_dev.bronze.csv_admission_raw". Writes bronze_df to that table using .format("delta").mode("overwrite").option("overwriteSchema", "true").saveAsTable(...). Prints a confirmation message.
# MAGIC
# MAGIC Cell 8 — Reads the freshly-written table back into bronze_check_df, calls .printSchema() to show the resulting column types, prints the total row count, and displays a sample selecting tenant_id, tenant_name, mrn, patient_first_name, patient_last_name, admit_datetime, discharge_datetime, department.
# MAGIC
# MAGIC Cell 9 — The duplicate check: groups bronze_check_df by tenant_id, mrn, admit_datetime, counts occurrences, filters to count > 1, and displays those — this is the cell that surfaced the duplicate-file problem in hospitalB/hospitalC that we just decided to handle via dedup in the silver layer.
# MAGIC
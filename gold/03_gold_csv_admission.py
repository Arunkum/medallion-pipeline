# Databricks notebook source
# MAGIC %md
# MAGIC # Gold — CSV Admission: Row-Level Table + Business Metrics
# MAGIC Reads validated records from `silver.csv_admission_valid` and produces two gold
# MAGIC tables, kept separate from the HL7 gold table (`gold.patient_admissions`) by design:
# MAGIC
# MAGIC 1. `gold.csv_admissions` — row-level, one row per admission. PHI columns
# MAGIC    (mrn, patient_first_name, patient_last_name, date_of_birth) are written
# MAGIC    unmasked here; masking is enforced separately at the Unity Catalog layer
# MAGIC    via column masks (next notebook), not baked into this write.
# MAGIC 2. `gold.csv_admission_metrics` — aggregated business metrics. PHI-free by
# MAGIC    construction (aggregation strips patient-level identity), so no masking
# MAGIC    is needed on this table at all.

# COMMAND ----------

dbutils.widgets.text("catalog", "healthcare_dev", "Catalog")
catalog = dbutils.widgets.get("catalog")

# COMMAND ----------

from pyspark.sql import functions as F

silver_valid_table = f"{catalog}.silver.csv_admission_valid"
gold_admissions_table = f"{catalog}.gold.csv_admissions"
gold_metrics_table = f"{catalog}.gold.csv_admission_metrics"

# COMMAND ----------

# ------------------------------------------------------------
# 1. Read silver valid records
# ------------------------------------------------------------

silver_df = spark.table(silver_valid_table)

print("Silver valid row count:", silver_df.count())

display(
    silver_df.select(
        "tenant_id", "mrn", "admit_datetime", "discharge_datetime",
        "department", "length_of_stay_days"
    )
)

# COMMAND ----------

# ------------------------------------------------------------
# 2. Build gold.csv_admissions (row-level, published copy)
# ------------------------------------------------------------
# Straight pass-through of the validated silver columns — gold's job here
# is to be the trusted, published row-level dataset, not to re-derive
# anything. A derived admission_id is added as a stable surrogate key
# since (tenant_id, mrn, admit_datetime) is unwieldy to join on repeatedly.

gold_admissions_df = (
    silver_df
    .withColumn(
        "admission_id",
        F.sha2(
            F.concat_ws("||", "tenant_id", "mrn", F.col("admit_datetime").cast("string")),
            256
        )
    )
    .select(
        "admission_id",
        "tenant_id",
        "tenant_name",
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
    )
)

# COMMAND ----------

(
    gold_admissions_df.write
    .format("delta")
    .mode("overwrite")
    .option("overwriteSchema", "true")
    .saveAsTable(gold_admissions_table)
)

print(f"Gold table written: {gold_admissions_table}")
print("Row count:", spark.table(gold_admissions_table).count())

# COMMAND ----------

# ------------------------------------------------------------
# 3. Build gold.csv_admission_metrics (aggregated, PHI-free)
# ------------------------------------------------------------
# No mrn/name/date_of_birth anywhere in this table — aggregation removes
# patient-level identity by construction, so this table needs no masking.

metrics_by_tenant_dept = (
    silver_df
    .groupBy("tenant_id", "tenant_name", "department")
    .agg(
        F.count("*").alias("admission_count"),
        F.round(F.avg("length_of_stay_days"), 2).alias("avg_length_of_stay_days"),
        F.min("length_of_stay_days").alias("min_length_of_stay_days"),
        F.max("length_of_stay_days").alias("max_length_of_stay_days")
    )
)

metrics_by_admit_type = (
    silver_df
    .groupBy("tenant_id", "tenant_name", "admit_type")
    .agg(
        F.count("*").alias("admission_count"),
        F.round(F.avg("length_of_stay_days"), 2).alias("avg_length_of_stay_days")
    )
)

metrics_by_disposition = (
    silver_df
    .groupBy("tenant_id", "tenant_name", "discharge_disposition")
    .agg(
        F.count("*").alias("discharge_count")
    )
)

# COMMAND ----------

# Union the three metric breakdowns into one table with a metric_type
# discriminator column, rather than three separate tables — keeps gold
# schema count manageable while still supporting different slice-and-dice
# views. Each metric_type has its own relevant non-null dimension column.

metrics_df = (
    metrics_by_tenant_dept
    .withColumn("metric_type", F.lit("by_department"))
    .withColumnRenamed("department", "dimension")
    .select("metric_type", "tenant_id", "tenant_name", "dimension",
            "admission_count", "avg_length_of_stay_days",
            "min_length_of_stay_days", "max_length_of_stay_days")

    .unionByName(
        metrics_by_admit_type
        .withColumn("metric_type", F.lit("by_admit_type"))
        .withColumnRenamed("admit_type", "dimension")
        .select("metric_type", "tenant_id", "tenant_name", "dimension",
                "admission_count", "avg_length_of_stay_days"),
        allowMissingColumns=True
    )

    .unionByName(
        metrics_by_disposition
        .withColumn("metric_type", F.lit("by_discharge_disposition"))
        .withColumnRenamed("discharge_disposition", "dimension")
        .withColumnRenamed("discharge_count", "admission_count")
        .select("metric_type", "tenant_id", "tenant_name", "dimension", "admission_count"),
        allowMissingColumns=True
    )
)

display(metrics_df)

# COMMAND ----------

(
    metrics_df.write
    .format("delta")
    .mode("overwrite")
    .option("overwriteSchema", "true")
    .saveAsTable(gold_metrics_table)
)

print(f"Gold metrics table written: {gold_metrics_table}")
print("Row count:", spark.table(gold_metrics_table).count())

# COMMAND ----------

# ------------------------------------------------------------
# 4. Verify both tables
# ------------------------------------------------------------

print("--- gold.csv_admissions schema ---")
spark.table(gold_admissions_table).printSchema()

print("--- gold.csv_admission_metrics schema ---")
spark.table(gold_metrics_table).printSchema()

display(spark.table(gold_admissions_table).limit(10))
display(spark.table(gold_metrics_table))
# Databricks notebook source
# MAGIC %md
# MAGIC # Silver — CSV Admission: Schema Enforcement, Cleansing, Comprehensive Validation & Quarantine
# MAGIC
# MAGIC Reads raw bronze CSV admission data, enforces schema via StructType, parses dual date formats
# MAGIC generically (ANSI-safe via try_to_date/try_to_timestamp), runs healthcare quality/chronology
# MAGIC checks, deduplicates deterministically, and writes to Silver Valid and Silver Quarantine tables.

# COMMAND ----------

dbutils.widgets.text("catalog", "healthcare_dev", "Catalog")
catalog = dbutils.widgets.get("catalog")

# COMMAND ----------

from pyspark.sql import functions as F
from pyspark.sql.window import Window
from pyspark.sql.types import (
    StructType,
    StructField,
    StringType,
    DateType,
    TimestampType,
    IntegerType
)

bronze_table = f"{catalog}.bronze.csv_admission_raw"
silver_valid_table = f"{catalog}.silver.csv_admission_valid"
silver_quarantine_table = f"{catalog}.silver.csv_admission_quarantine"

# COMMAND ----------

# ------------------------------------------------------------
# 1. Declare the target silver schema explicitly
# ------------------------------------------------------------
# All fields nullable=True: casting/select() does not itself enforce
# not-null (Delta requires an explicit constraint for that), so real
# not-null enforcement happens via the validation rules below (e.g. the
# missing-mrn check), not via this StructType. Schema comparison below
# checks name + dtype only, not nullable, to avoid false failures from
# nullability inferred off the bronze source columns.

silver_schema = StructType([
    StructField("tenant_id",              StringType(),    True),
    StructField("tenant_name",            StringType(),    True),
    StructField("source_file_tenant_id",  StringType(),    True),

    StructField("mrn",                    StringType(),    True),
    StructField("patient_first_name",     StringType(),    True),
    StructField("patient_last_name",      StringType(),    True),
    StructField("date_of_birth",          DateType(),      True),
    StructField("gender",                 StringType(),    True),

    StructField("admit_datetime",         TimestampType(), True),
    StructField("discharge_datetime",     TimestampType(), True),
    StructField("admit_type",             StringType(),    True),
    StructField("department",             StringType(),    True),
    StructField("attending_provider_id",  StringType(),    True),
    StructField("admit_diagnosis_code",   StringType(),    True),
    StructField("discharge_diagnosis_code", StringType(),  True),
    StructField("discharge_disposition",  StringType(),    True),
    StructField("length_of_stay_days",    IntegerType(),   True),

    StructField("source_path",            StringType(),    True),
    StructField("ingest_ts",              TimestampType(), True)
])

# COMMAND ----------

# ------------------------------------------------------------
# 2. Read bronze (all strings, untouched)
# ------------------------------------------------------------

bronze_df = spark.table(bronze_table)

display(
    bronze_df.select(
        "tenant_id", "mrn", "date_of_birth",
        "admit_datetime", "discharge_datetime", "length_of_stay_days"
    )
)

# COMMAND ----------

# ------------------------------------------------------------
# 3. Generic dual-format date/timestamp parsing (ANSI-safe)
# ------------------------------------------------------------
# Known formats: yyyy-MM-dd (hospitalA) and dd-MM-yyyy (hospitalB/C).
# Tries both for every row and coalesces to whichever parses — no
# tenant-based branching, so it keeps working if a new tenant shows up
# using either of these two formats.
#
# IMPORTANT: uses try_to_date/try_to_timestamp, not to_date/to_timestamp.
# Under ANSI mode (on by default in this workspace), to_date/to_timestamp
# RAISE on a non-matching format instead of returning NULL, which breaks
# coalesce before it ever reaches the second format attempt. The try_
# variants return NULL on failure instead, which is what coalesce needs.
# Also note: try_to_date/try_to_timestamp require the format arg wrapped
# in F.lit(...) — passing a bare Python string gets misresolved as a
# column reference.

DATE_FORMATS = ["yyyy-MM-dd", "dd-MM-yyyy"]
TIMESTAMP_FORMATS = [
    "yyyy-MM-dd HH:mm:ss",
    "dd-MM-yyyy HH:mm:ss",
    "yyyy-MM-dd HH:mm",
    "dd-MM-yyyy HH:mm"
]


def parse_date_multi_format(col_name):
    return F.coalesce(
        *[F.try_to_date(F.col(col_name), F.lit(fmt)) for fmt in DATE_FORMATS]
    )


def parse_timestamp_multi_format(col_name):
    return F.coalesce(
        *[F.try_to_timestamp(F.col(col_name), F.lit(fmt)) for fmt in TIMESTAMP_FORMATS]
    )


typed_df = (
    bronze_df
    .withColumn("date_of_birth_parsed", parse_date_multi_format("date_of_birth"))
    .withColumn("admit_datetime_parsed", parse_timestamp_multi_format("admit_datetime"))
    .withColumn("discharge_datetime_parsed", parse_timestamp_multi_format("discharge_datetime"))
)

# COMMAND ----------

# Sanity check: any row where the raw value was non-blank but BOTH
# formats failed to parse (shows up as NULL despite source data existing)
# — a real data-quality signal, not just a missing value. Checked here,
# against the RAW string column, before it gets dropped by the cast below.

display(
    typed_df
    .filter(
        (F.trim(F.col("date_of_birth")) != "")
        & F.col("date_of_birth_parsed").isNull()
    )
    .select("tenant_id", "mrn", "date_of_birth")
)

display(
    typed_df
    .filter(
        (F.trim(F.col("admit_datetime")) != "")
        & F.col("admit_datetime_parsed").isNull()
    )
    .select("tenant_id", "mrn", "admit_datetime")
)

# COMMAND ----------

# ------------------------------------------------------------
# 4. Cast remaining columns per StructType, assemble final schema
# ------------------------------------------------------------
# Keep the raw date_of_birth string alongside the parsed value (as
# date_of_birth_raw) so the validation rules below can check "was there
# raw data that failed to parse" without calling string functions on an
# already-cast DateType column (trim() on a Date column throws).

cleansed_df = (
    typed_df
    .select(
        F.trim(F.col("tenant_id")).cast(StringType()).alias("tenant_id"),
        F.trim(F.col("tenant_name")).cast(StringType()).alias("tenant_name"),
        F.trim(F.col("source_file_tenant_id")).cast(StringType()).alias("source_file_tenant_id"),

        F.trim(F.col("mrn")).cast(StringType()).alias("mrn"),
        F.trim(F.col("patient_first_name")).cast(StringType()).alias("patient_first_name"),
        F.trim(F.col("patient_last_name")).cast(StringType()).alias("patient_last_name"),
        F.col("date_of_birth_parsed").alias("date_of_birth"),
        F.trim(F.col("date_of_birth")).alias("date_of_birth_raw"),
        F.upper(F.trim(F.col("gender"))).cast(StringType()).alias("gender"),

        F.col("admit_datetime_parsed").alias("admit_datetime"),
        F.col("discharge_datetime_parsed").alias("discharge_datetime"),
        F.trim(F.col("admit_type")).cast(StringType()).alias("admit_type"),
        F.trim(F.col("department")).cast(StringType()).alias("department"),
        F.trim(F.col("attending_provider_id")).cast(StringType()).alias("attending_provider_id"),
        F.trim(F.col("admit_diagnosis_code")).cast(StringType()).alias("admit_diagnosis_code"),
        F.trim(F.col("discharge_diagnosis_code")).cast(StringType()).alias("discharge_diagnosis_code"),
        F.trim(F.col("discharge_disposition")).cast(StringType()).alias("discharge_disposition"),
        F.col("length_of_stay_days").cast(IntegerType()).alias("length_of_stay_days"),

        F.col("source_path").cast(StringType()).alias("source_path"),
        F.col("ingest_ts").cast(TimestampType()).alias("ingest_ts")
    )
)

# COMMAND ----------

# Robust schema check: compares name + dtype only (not nullable, since
# nullable is inferred from the bronze source and isn't meaningfully
# enforced by select()/cast() anyway). Checks against silver_schema's
# columns only — date_of_birth_raw is a working column dropped before
# the final write, so it's intentionally excluded here.

def validate_schema(df, expected_schema):
    expected_names = [f.name for f in expected_schema.fields]
    df_subset = df.select(*expected_names)
    diffs = [
        f"Col: {f1.name} (Actual: {f1.dataType}, Expected: {f2.dataType})"
        for f1, f2 in zip(df_subset.schema.fields, expected_schema.fields)
        if f1.dataType != f2.dataType or f1.name != f2.name
    ]
    return diffs


schema_diffs = validate_schema(cleansed_df, silver_schema)
assert len(schema_diffs) == 0, f"Schema assertion failed! Details: {schema_diffs}"

print("Schema validation successful.")
cleansed_df.printSchema()

# COMMAND ----------

# ------------------------------------------------------------
# 4b. Deterministic deduplication
# ------------------------------------------------------------
# Bronze intentionally mirrors whatever landed in ADLS as-is, including
# accidental duplicate file uploads (confirmed via dbutils.fs.ls — some
# tenant folders had more than one file). Silver applies the business
# rule: one row per tenant + mrn + admit_datetime. Order by ingest_ts
# desc, then source_path desc as a tiebreaker in case ingest_ts ties
# (same batch write) — keeps the result deterministic across reruns.

dedup_window = Window.partitionBy(
    "tenant_id", "mrn", "admit_datetime"
).orderBy(F.col("ingest_ts").desc(), F.col("source_path").desc())

before_count = cleansed_df.count()

deduped_df = (
    cleansed_df
    .withColumn("_row_num", F.row_number().over(dedup_window))
    .filter(F.col("_row_num") == 1)
    .drop("_row_num")
)

after_count = deduped_df.count()

print(f"Rows before dedup: {before_count}")
print(f"Rows after dedup:  {after_count}")
print(f"Duplicates removed: {before_count - after_count}")

# COMMAND ----------

# Review exactly which (tenant_id, mrn, admit_datetime) combos had duplicates
display(
    cleansed_df
    .groupBy("tenant_id", "mrn", "admit_datetime")
    .count()
    .filter(F.col("count") > 1)
)

# COMMAND ----------

# ------------------------------------------------------------
# 5. Healthcare validation rules (comprehensive, multi-error visible)
# ------------------------------------------------------------
# Each rule is a F.when(...) that returns an error string or NULL.
# All rules run against every row; failures collect into an array so a
# single row can surface more than one issue instead of only the first
# one matched (unlike a chained .when()/.otherwise()).

validation_rules = [
    # --- Identifiers & critical demographics ---
    F.when(F.col("mrn").isNull() | (F.col("mrn") == ""), "Missing or blank MRN"),
    F.when(~F.col("gender").isin("M", "F", "O", "U"), "Invalid gender value"),

    # --- Date parsing checks (against the preserved raw string) ---
    F.when(F.col("admit_datetime").isNull(), "Unparseable or missing admit_datetime"),
    F.when(
        (F.col("date_of_birth_raw").isNotNull())
        & (F.col("date_of_birth_raw") != "")
        & F.col("date_of_birth").isNull(),
        "Unparseable date_of_birth"
    ),

    # --- Chronology & logical consistency ---
    F.when(F.col("admit_datetime") > F.current_timestamp(), "Future admit_datetime"),
    F.when(F.col("date_of_birth") > F.current_date(), "Future date_of_birth"),
    F.when(
        F.col("date_of_birth").isNotNull()
        & F.col("admit_datetime").isNotNull()
        & (F.col("date_of_birth") > F.col("admit_datetime").cast("date")),
        "date_of_birth is after admit_datetime"
    ),
    F.when(
        F.col("discharge_datetime").isNotNull()
        & F.col("admit_datetime").isNotNull()
        & (F.col("discharge_datetime") < F.col("admit_datetime")),
        "discharge_datetime is prior to admit_datetime"
    ),

    # --- Clinical coding format (ICD-10 syntax) ---
    # NOTE: verify against your actual sample data before trusting this —
    # if the fabricated diagnosis codes aren't real ICD-10-shaped values,
    # this will quarantine rows that are otherwise fine.
    F.when(
        F.col("admit_diagnosis_code").isNotNull()
        & (F.col("admit_diagnosis_code") != "")
        & ~F.col("admit_diagnosis_code").rlike(r"^[A-Z][0-9][0-9A-Z](\.[0-9A-Z]{1,4})?$"),
        "Invalid admit_diagnosis_code format (ICD-10)"
    ),
    F.when(
        F.col("discharge_diagnosis_code").isNotNull()
        & (F.col("discharge_diagnosis_code") != "")
        & ~F.col("discharge_diagnosis_code").rlike(r"^[A-Z][0-9][0-9A-Z](\.[0-9A-Z]{1,4})?$"),
        "Invalid discharge_diagnosis_code format (ICD-10)"
    )
]

validated_df = (
    deduped_df
    .withColumn("_raw_errors", F.array(*validation_rules))
    .withColumn("_cleaned_errors", F.expr("filter(_raw_errors, x -> x IS NOT NULL)"))
    .withColumn(
        "validation_status",
        F.when(F.size(F.col("_cleaned_errors")) > 0, "INVALID").otherwise("VALID")
    )
    .withColumn("validation_error", F.concat_ws("; ", F.col("_cleaned_errors")))
    .drop("_raw_errors", "_cleaned_errors")
)

# valid_df drops the working date_of_birth_raw column too, so it matches silver_schema exactly
valid_df = (
    validated_df
    .filter(F.col("validation_status") == "VALID")
    .drop("validation_status", "validation_error", "date_of_birth_raw")
)

quarantine_df = validated_df.filter(F.col("validation_status") == "INVALID")

# COMMAND ----------

display(
    validated_df.select(
        "tenant_id", "mrn", "gender", "admit_datetime",
        "validation_status", "validation_error"
    )
)

# COMMAND ----------

# ------------------------------------------------------------
# 6. Write silver tables
# ------------------------------------------------------------

(
    valid_df.write
    .format("delta")
    .mode("overwrite")
    .option("overwriteSchema", "true")
    .saveAsTable(silver_valid_table)
)

(
    quarantine_df.write
    .format("delta")
    .mode("overwrite")
    .option("overwriteSchema", "true")
    .saveAsTable(silver_quarantine_table)
)

print("Valid records written:", valid_df.count())
print("Quarantined records written:", quarantine_df.count())

# COMMAND ----------

# ------------------------------------------------------------
# 7. Verify
# ------------------------------------------------------------

spark.table(silver_valid_table).printSchema()

display(
    spark.table(silver_valid_table).select(
        "tenant_id", "mrn", "date_of_birth", "admit_datetime",
        "discharge_datetime", "length_of_stay_days"
    )
)

# COMMAND ----------

from pyspark.sql import functions as F

display(
    spark.table(silver_quarantine_table)
    .withColumn("error", F.explode(F.split(F.col("validation_error"), "; ")))
    .groupBy("error")
    .count()
    .orderBy(F.col("count").desc())
)
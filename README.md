# ✈️ Flight Delay Prediction Platform on Google Cloud

> An end-to-end data engineering, ML engineering, and MLOps platform built on GCP to predict flight delays using BTS On-Time Performance data and NOAA weather data.

---

## 📌 Problem Statement

Flight delays cost the U.S. aviation industry over **$33 billion annually**. This project builds a production-grade platform that:

- Migrates historical flight and weather data from on-premise sources to Google Cloud
- Constructs scalable ETL/ELT pipelines using a Medallion architecture
- Models data using Kimball star schema in BigQuery
- Trains an ML model (XGBoost + TabNet) to **predict flight delays 3 hours before departure**
- Implements full MLOps — versioning, batch prediction, drift monitoring, and automated retraining

---

## 🏗️ Architecture Overview

```
On-Premise / BTS Download
        │
        ▼
  ┌─────────────┐      Apache Beam / Dataflow
  │  GCS Bronze │ ──────────────────────────────► GCS Silver ──► GCS Gold
  │  (raw data) │      (cleanse, validate,          (joined,       (feature
  └─────────────┘       dead-letter queue)          enriched)       ready)
                                                        │
                                                        ▼
                                                  BigQuery
                                          (Star Schema + Partitioning)
                                                        │
                              ┌─────────────────────────┤
                              ▼                         ▼
                        EDA & SQL               Vertex AI Training
                        Analysis                (XGBoost + TabNet)
                                                        │
                                                        ▼
                                              Vertex AI Pipelines
                                           (KFP + Model Registry)
                                                        │
                              ┌─────────────────────────┤
                              ▼                         ▼
                       Batch Prediction          Online Endpoint
                     (SHAP Explanations)      (90/10 Traffic Split)
                                                        │
                                                        ▼
                                            Monitoring + CI/CD
                                       (Cloud Build + Pub/Sub Trigger)
```

---

## 📦 Tech Stack

| Layer | Technology |
|---|---|
| Infrastructure | Terraform, GCP (GCS, BigQuery, Composer, Vertex AI, Cloud Run) |
| Ingestion | Apache Beam / Cloud Dataflow |
| Orchestration | Cloud Composer (Apache Airflow 2.x) |
| Data Quality | Great Expectations |
| Storage & Modeling | BigQuery (Medallion + Kimball Star Schema) |
| ML Training | Vertex AI Custom Training, XGBoost, TabNet |
| Hyperparameter Tuning | Vertex AI Vizier |
| MLOps | Vertex AI Pipelines (KFP), Model Registry |
| Monitoring | Vertex AI Model Monitoring (drift + skew) |
| CI/CD | Cloud Build, Pub/Sub, Cloud Run |
| Security | VPC-SC, CMEK, Dataplex |

---

## 📂 Project Structure

```
flight-delay-platform/
├── infra/
│   ├── main.tf                  # Terraform — GCS, BigQuery, Composer, IAM
│   ├── variables.tf
│   └── outputs.tf
├── pipelines/
│   ├── ingestion/
│   │   └── beam_bronze_silver.py     # Dataflow ETL with dead-letter queue
│   ├── dags/
│   │   └── flight_pipeline_dag.py    # Airflow DAG with sensors + retries
│   └── kfp/
│       └── vertex_pipeline.py        # KFP pipeline — train → eval → deploy
├── sql/
│   ├── ddl/
│   │   ├── fact_flights.sql
│   │   ├── dim_airport.sql
│   │   ├── dim_carrier.sql
│   │   ├── dim_date.sql
│   │   └── dim_weather.sql
│   ├── transforms/
│   │   └── scd_type2_merge.sql       # SCD Type 2 MERGE for dim tables
│   └── analysis/
│       ├── eda_delay_distribution.sql
│       ├── eda_weather_correlation.sql
│       └── feature_engineering_view.sql
├── ml/
│   ├── train_xgboost.py
│   ├── train_tabnet.py
│   ├── evaluate.py
│   └── batch_predict_shap.py
├── monitoring/
│   └── drift_monitor_config.yaml
├── cloudbuild/
│   └── cloudbuild.yaml               # CI/CD with retraining trigger
├── tests/
│   ├── unit/
│   ├── integration/
│   └── great_expectations/
└── README.md
```

---

## 📊 Data Sources

| Source | Description | Rows (Sample) |
|---|---|---|
| [BTS On-Time Performance](https://www.transtats.bts.gov/OT_Delay/OT_DelayCause1.asp) | Flight delay records by airline/route | ~544K (AA, Jan 2026) |
| [NOAA ASOS](https://www.ncei.noaa.gov/products/land-based-station/automated-surface-observing-systems) | Hourly surface weather observations | Joined by airport + hour |

### Key Columns Used (30 of 110+)

<details>
<summary>Expand column list</summary>

**Identity & Time:** `FlightDate`, `Reporting_Airline`, `Tail_Number`, `Flight_Number_Reporting_Airline`, `OriginAirportID`, `DestAirportID`

**Origin/Destination:** `Origin`, `OriginCityName`, `Dest`, `DestCityName`

**Schedule & Delays:** `CRSDepTime`, `DepTime`, `DepDelayMinutes`, `CRSArrTime`, `ArrTime`, `ArrDelayMinutes`, `ArrDel15`, `DepDel15`

**Flight Status:** `Cancelled`, `CancellationCode`, `Diverted`

**Performance:** `TaxiOut`, `TaxiIn`, `Distance`, `CRSElapsedTime`

**Delay Causes:** `CarrierDelay`, `WeatherDelay`, `NASDelay`, `LateAircraftDelay`, `SecurityDelay`

**Weather (NOAA join):** `wind_speed_knots`, `ceiling_ft`, `visibility_miles`, `thunder_flag`

</details>

> ⚠️ **Data Leakage Warning:** `ActualElapsedTime`, `ArrTime`, and `ArrDel15` are post-flight values. They are **excluded from training features** — only information available ≥3h before departure is used.

---

## 🗄️ Data Model (Star Schema)

```
                    ┌──────────────┐
                    │  dim_date    │
                    └──────┬───────┘
                           │
  ┌──────────────┐   ┌─────┴──────────┐   ┌──────────────┐
  │  dim_airport ├───┤  fact_flights  ├───┤  dim_carrier │
  └──────────────┘   └─────┬──────────┘   └──────────────┘
                           │
                    ┌──────┴───────┐
                    │  dim_weather │
                    └──────────────┘
```

- **Partitioned** on `flight_date` (daily)
- **Clustered** on `reporting_airline`, `origin_airport_id`
- SCD Type 2 MERGE for slowly changing dimensions

---

## 🤖 ML Model

### Target Variable
`is_delayed` — binary flag: arrival delay ≥ 15 minutes (cancelled flights → `NULL`, excluded from training)

### Feature Engineering
| Feature | Source | Signal Strength |
|---|---|---|
| `dep_hour` | Schedule | High |
| `day_of_week` | Schedule | High |
| `month` | Schedule | Medium |
| `distance_miles` | Flight | Medium |
| `taxi_out_minutes` | Operations | High |
| `tail_chain_delay` | Tail_Number cascade | Very High |
| `wind_speed_knots` | NOAA ASOS | High |
| `ceiling_ft` | NOAA ASOS | High |
| `visibility_miles` | NOAA ASOS | High |
| `thunder_flag` | NOAA ASOS | Medium |

### Models Trained
- **XGBoost** — baseline, fast iteration, SHAP explanations
- **TabNet** — neural attention-based tabular model
- **Ensemble** — soft-vote final predictor

### Target Metrics
| Metric | Target |
|---|---|
| AUC-ROC | ≥ 0.882 |
| Precision@80% Recall | ≥ 0.74 |

---

## 🔄 MLOps Pipeline (Vertex AI)

```
Data Validation → Feature Engineering → Train → Evaluate
                                                    │
                                          ┌─────────┴──────────┐
                                    Pass eval gate?
                                    │ Yes               │ No
                                    ▼                   ▼
                             Push to Registry     Alert + Stop
                             (champion alias)
                                    │
                         ┌──────────┴──────────┐
                         ▼                     ▼
                  Batch Prediction        Online Endpoint
                 (SHAP explanations)   (90/10 traffic split)
                         │
                         ▼
                  Drift Monitoring
                  (feature + prediction skew thresholds)
                         │
                         ▼
                  Pub/Sub → Cloud Run → Retrain Trigger
```

---

## 🚀 Getting Started

### Prerequisites

- GCP Project with billing enabled
- `gcloud` CLI authenticated
- Terraform ≥ 1.5
- Python ≥ 3.10

### 1. Clone the repo

```bash
git clone https://github.com/your-username/flight-delay-platform.git
cd flight-delay-platform
```

### 2. Provision infrastructure

```bash
cd infra
terraform init
terraform plan -var="project_id=YOUR_PROJECT_ID"
terraform apply
```

### 3. Download BTS data

Go to [transtats.bts.gov](https://www.transtats.bts.gov), select the 30 columns listed above, download as pre-zipped CSV, and upload to your GCS Bronze bucket:

```bash
gsutil cp *.zip gs://your-project-bronze/flights/raw/
```

### 4. Trigger the pipeline

```bash
gcloud composer environments run your-composer-env \
  --location us-central1 dags trigger \
  -- flight_pipeline_dag
```

### 5. Train the model

```bash
python ml/train_xgboost.py \
  --project YOUR_PROJECT_ID \
  --region us-central1 \
  --dataset bq://YOUR_PROJECT_ID.gold.feature_view
```

---

## 💰 Estimated Monthly Cost

| Service | Cost |
|---|---|
| BigQuery (storage + queries) | ~$320 |
| Cloud Composer | ~$800 |
| Dataflow | ~$600 |
| Vertex AI Training | ~$1,200 |
| Vertex AI Endpoints | ~$900 |
| GCS Storage | ~$85 |
| Monitoring & Logging | ~$550 |
| **Total** | **~$4,455/month** |

> Costs reduce significantly when using committed use discounts or scaling down non-prod environments.

---

## 🔒 Security

- **VPC Service Controls** — perimeter around BigQuery and GCS
- **CMEK** — customer-managed encryption keys for all storage
- **Dataplex** — data governance, lineage, and tagging
- **IAM** — least-privilege service accounts per pipeline stage
- **Audit Logging** — all data access logged to Cloud Logging

---

## 📅 Delivery Timeline

| Week | Milestone |
|---|---|
| 1–2 | IaC setup, GCS buckets, IAM, Bronze ingestion |
| 3–4 | Dataflow ETL, data quality checks, Silver layer |
| 5–6 | BigQuery star schema, SCD Type 2, Gold layer |
| 7–8 | EDA, feature engineering view, partitioning optimization |
| 9–10 | XGBoost + TabNet training, hyperparameter tuning |
| 11–12 | Vertex AI Pipelines, Model Registry, batch prediction |
| 13–14 | Monitoring, CI/CD, online endpoint, documentation |

---

## 🧪 Testing

```bash
# Unit tests
pytest tests/unit/

# Integration tests (requires GCP credentials)
pytest tests/integration/

# Data quality checks
great_expectations checkpoint run flight_data_checkpoint
```

---

## 📈 Key Findings (Jan 2026 AA Data)

- **6.9% cancellation rate** — ~2–3× national average, driven by January winter storms
- **LateAircraftDelay** accounts for **38% of all delay minutes** (avg 39 min)
- The `tail_chain_delay` feature (cascade via `Tail_Number` within same day) is the single highest-signal ML feature
- Adding NOAA weather features lifts AUC from ~0.74 → **0.88+**

---

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/your-feature`)
3. Commit your changes (`git commit -m 'Add your feature'`)
4. Push to the branch (`git push origin feature/your-feature`)
5. Open a Pull Request

---

## 📄 License

This project is licensed under the MIT License — see [LICENSE](LICENSE) for details.

---

## 🙏 Acknowledgements

- [Bureau of Transportation Statistics](https://www.transtats.bts.gov) — BTS On-Time Performance dataset
- [NOAA NCEI](https://www.ncei.noaa.gov) — ASOS surface weather observations
- Google Cloud documentation for Vertex AI Pipelines and Dataflow

---

*Built as an end-to-end reference implementation covering data engineering, ML engineering, and MLOps on Google Cloud Platform.*

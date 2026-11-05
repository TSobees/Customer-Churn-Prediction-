# Dataset source and use

The existing `Telco_customer_churn.csv` is retained from the original project.
Its cited source is [Telco Customer Churn on Kaggle](https://www.kaggle.com/datasets/abdallahwagih/telco-customer-churn).
This repository does not independently verify original collection methods or the
dataset's redistribution terms. The code's MIT license does not assign a license
to third-party data. Check the upstream terms before reusing or redistributing it.

Training excludes customer IDs, geographic identifiers, churn outcome fields,
Churn Score and CLTV. No customer-level rows are returned by the API. Generated
evaluation contains aggregate metrics and a source fingerprint. A random holdout
measures performance on this supplied snapshot; there are no event timestamps
establishing a prospective prediction task.

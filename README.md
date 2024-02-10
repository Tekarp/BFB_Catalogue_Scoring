gcloud builds submit --tag gcr.io/symmetric-hash-413516/catscore-bucket01 --project=symmetric-hash-413516
gcloud run deploy --image gcr.io/symmetric-hash-413516/catscore-bucket01 --platform managed  --project=symmetric-hash-413516 --allow-unauthenticated

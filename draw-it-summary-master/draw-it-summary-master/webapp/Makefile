install_requirements:
	@pip install -r requirements.txt

streamlit:
	@streamlit run canvas.py

deploy_cloud_run:
	@docker build -t $GCR_MULTI_REGION/$PROJECT/$IMAGE .
	@docker push $GCR_MULTI_REGION/$PROJECT/$IMAGE
	@gcloud run deploy --image $GCR_MULTI_REGION/$PROJECT/$IMAGE --memory $MEMORY  --timeout=$TIMEOUT --region $REGION --env-vars-file .env.yaml

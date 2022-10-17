# ----------------------------------
#          Key make commands
# ----------------------------------

install_all:
	@pip install -e .

reinstall_all:
	@pip uninstall -y draw_it_model || :
	@pip install -e .

run_create_output_folders:
	python -c 'from draw_it_model.ml_logic.data import create_output_folders; create_output_folders()'

run_data_generation:
	python -c 'from draw_it_model.ml_logic.data import create_img_data; create_img_data()'

run_data_split:
	python -c 'from draw_it_model.ml_logic.data import create_split_folders; create_split_folders()'

run_setup_and_data_storage:
	run_create_output_folders run_data_generation run_data_split

run_train_model:
	python -c 'from draw_it_model.interface.main import train; train()'

run_evaluate_model:
	python -c 'from draw_it_model.interface.main import evaluate; evaluate()'

run_predict:
	python -c 'from draw_it_model.interface.main import predict; predict()'

run_test:
	python -c 'from draw_it_model.interface.main import test; test()'

run_api:
	uvicorn draw_it_model.api.fast:app --reload

deploy_cloud_run:
	@docker build -t $GCR_MULTI_REGION/$PROJECT/$IMAGE .
	@docker push $GCR_MULTI_REGION/$PROJECT/$IMAGE
	@gcloud run deploy --image $GCR_MULTI_REGION/$PROJECT/$IMAGE --memory $MEMORY  --timeout=$TIMEOUT --region $REGION --env-vars-file .env.yaml
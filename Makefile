.PHONY: environment remove-env
ENVIRONMENT=data_project

environment: remove-env
		conda create -n $(ENVIRONMENT) "python=3.10" --yes
		conda run -n $(ENVIRONMENT) conda install -c conda-forge matplotlib numpy pandas scipy scikit-image seaborn scikit-learn tensorflow tqdm --yes
		conda run -n $(ENVIRONMENT) conda install -c pytorch pytorch torchvision --yes

remove-env:
		conda remove --name $(ENVIRONMENT) --all --yes
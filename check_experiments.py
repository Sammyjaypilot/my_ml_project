import mlflow

for exp in mlflow.search_experiments():
    print(exp.name)

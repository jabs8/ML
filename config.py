import os

def project_dir():
    # Vérifier si le "PycharmProject"" est dans le chemin
    if "PycharmProjects" in os.getcwd():
        project_dir = ""
    else:
        project_dir = "PycharmProjects/ML"
    return project_dir
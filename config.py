import os

def project_dir():
    # Vérifier si le "PycharmProject"" est dans le chemin
    # if "PycharmProjects" in os.getcwd():
    #     project_dir = ""
    # else:
    #     project_dir = "PycharmProjects/ML"
    # return project_dir
    projetct_dir = os.getcwd()
    return projetct_dir

import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.keras.models import Model
from scipy.spatial.distance import cosine
from tkinter import Tk, Label, Button, filedialog, Frame, Canvas, Scrollbar, PhotoImage
from PIL import Image, ImageTk

# Initialisation du modèle ResNet50 avec les poids pré-entraînés sur ImageNet.
base_model = ResNet50(weights='imagenet')
model = Model(inputs=base_model.input, outputs=base_model.get_layer('avg_pool').output)

def load_and_preprocess_img(img_path):
    """ Charge et prétraite une image pour qu'elle soit compatible avec ResNet50. """
    img = image.load_img(img_path, target_size=(224, 224))#charge et redimensionne l'image
    img_array = image.img_to_array(img)
    img_array_expanded = np.expand_dims(img_array, axis=0)
    return preprocess_input(img_array_expanded)

def extract_features(img_path, model):
    """ Extrait les caractéristiques d'une image en utilisant le modèle ResNet50. """
    img_preprocessed = load_and_preprocess_img(img_path)#appel de 1ere fonction pour preparer l'image
    features = model.predict(img_preprocessed)
    return features.flatten()

def create_feature_index(img_folder, model):
    """ Indexe toutes les images d'un dossier pour une recherche rapide des similarités. """
    feature_index = {}
    for img_name in os.listdir(img_folder):
        img_path = os.path.join(img_folder, img_name)
        features = extract_features(img_path, model)
        feature_index[img_name] = features
    return feature_index

def find_similar_images(user_img_path, feature_index, model, top_n=3):
    """ Trouve les images les plus similaires à l'image de l'utilisateur. """
    user_features = extract_features(user_img_path, model)
    similarities = {}
    for img_name, features in feature_index.items():
        sim = cosine(user_features, features)
        similarities[img_name] = sim
    sorted_similarities = sorted(similarities.items(), key=lambda item: item[1])
    return sorted_similarities[:top_n]

def browse_file():
    """ Ouvre un dialogue pour permettre a l'utilisateur de choisir un fichier image et affichier resultat """
    f_path = filedialog.askopenfilename()
    if f_path:
        similar_images = find_similar_images(f_path, feature_index, model)
        show_results(similar_images, f_path)

def show_results(similar_images, query_path):
    """ Affiche l'image requête et les images similaires trouvées. """
    for widget in frame.winfo_children():
        widget.destroy()#pour nettoyer fenetre avant d'afficher resultat
    query_img = Image.open(query_path)
    query_photo = ImageTk.PhotoImage(query_img.resize((256, 256)))#charge l'image requete et la redimensionne 
    query_label = Label(frame, image=query_photo)#creer label pour afficher l'image requete
    query_label.image = query_photo
    query_label.grid(row=0, column=0, columnspan=len(similar_images))#positionne l'image requete dans la grille

    row = 1
    col = 0
    for index, (img, sim) in enumerate(similar_images):
        img_path = os.path.join(img_folder, img)#construire chemin de l'image similaire et le charge
        load_img = Image.open(img_path)
        photo = ImageTk.PhotoImage(load_img.resize((256, 256)))
        img_label = Label(frame, image=photo)
        img_label.image = photo
        img_label.grid(row=row, column=col)
        sim_label = Label(frame, text=f"Similarité: {1 - sim:.4f}")#creer label pour chaque image similaire et son score
        sim_label.grid(row=row + 1, column=col)
        col += 1
    canvas.configure(scrollregion=canvas.bbox("all"))

if __name__ == "__main__":
    img_folder = 'base'
    feature_index = create_feature_index(img_folder, model)
    root = Tk()
    root.title("Fashion Search Engine")  # Ajout du titre
    root.geometry('800x600')
    root.resizable(True, True)

    title_label = Label(root, text="Fashion Search Engine", font=("Helvetica", 16))  # Titre ajouté
    title_label.pack()

    button_frame = Frame(root)
    button_frame.pack(fill="x", pady=10)  # Ajout d'un peu d'espace en bas du titre

    btn_load = Button(button_frame, text="Choisir une image requête", command=browse_file, height=2, width=30)  # Bouton agrandi
    btn_load.pack(padx=10)  # Ajout de marge autour du bouton

    scrollbar = Scrollbar(root)
    scrollbar.pack(side='right', fill='y')
    canvas = Canvas(root, yscrollcommand=scrollbar.set)
    frame = Frame(canvas)
    canvas_frame = canvas.create_window((0, 0), window=frame, anchor='nw')
    canvas.pack(side="left", fill="both", expand=True)
    scrollbar.config(command=canvas.yview)
    root.mainloop()

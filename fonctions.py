"""LES MODULES"""

import os
from pathlib import Path
from vosk import Model, KaldiRecognizer
import sounddevice as sd
import queue
import json
import subprocess
import sys
from openai import OpenAI
import requests
import time
from PIL import Image, ImageFilter, ImageDraw
import numpy as np
import trimesh
import pygame
import smtplib
import ssl
import shutil
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
from email import encoders
import zipfile


"""FIN MODULES"""


"""LES BRUITAGES"""
# Initialiser pygame et le module mixer pour l'audio
pygame.init()
pygame.mixer.init(frequency=22050, size=-16, channels=2, buffer=512)

def jouer_effet_sonore(chemin_fichier):
    """
    Joue un fichier MP3 comme effet sonore.
    """
    try:
        # Charger et jouer le son
        son = pygame.mixer.Sound(chemin_fichier)
        son.play()
        
        # Attendre la fin du son
        while pygame.mixer.get_busy():
            time.sleep(0.1)
            
    except pygame.error as e:
        print(f"Erreur lors du chargement du fichier {chemin_fichier}: {e}")
    except FileNotFoundError:
        print(f"Fichier non trouvé: {chemin_fichier}")




client = OpenAI(
    api_key="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",
    base_url="https://openrouter.ai/api/v1"
)

def poems_to_image_prompt(poems: str) -> str:
    response = client.chat.completions.create(
        model="mistralai/mistral-7b-instruct",
        messages=[
            {
                "role": "system",
                "content": (
                    "The user gives you one or more poems.\n"
                    "Consider these poems as one universe: combine them.\n"
                    "You create a detailed image generation prompt based on this universe.\n"
                    "Output ONLY the image prompt in English.\n"
                    "Focus on shapes, art style.\n"
                    "No explanations."
                )
            },
            {
                "role": "user",
                "content": poems
            }
        ],
        temperature=0.8,
        max_tokens=200
    )

    return response.choices[0].message.content.strip()




def repondre(msg: str):
    response = client.chat.completions.create(
        model="mistralai/mistral-7b-instruct",
        messages=[
            {
                "role": "system",
                "content": (
"L'utilisateur lit un poème. Ne pas l'interrompre, sauf si :\n"
"Si il semble vouloir arrêter la dictation : tu renvoies 'STOP'.\n"
"Si il semble vouloir générer une image, tu renvoies 'GENERATION'.\n"
"Sinon, si l'utilisateur semble continuer de lire son poème, tu renvoies 'OK', comme un signe que tu l'écoutes toujours.\n"
                            )
            },
            {
                "role": "user",
                "content": msg
            }
        ],
        temperature=0.8,
        max_tokens=200
    )
    
    return response.choices[0].message.content.strip()


def finir_poeme(msg, nom_de_plume):
    return msg.lower() == nom_de_plume.lower()



def get_nom_fichier_suivant():
    """Trouve le prochain nom: poeme1.txt, poeme2.txt..."""
    
    dossier = Path("poemes")
    
    # Compter les fichiers existants poemeX.txt
    fichiers = list(dossier.glob("poeme*.txt"))
    if not fichiers:
        return dossier / "poeme1.txt"
    
    # Extraire les numéros et trouver le max
    numeros = []
    for fichier in fichiers:
        nom = fichier.stem  # sans extension
        if nom.startswith("poeme"):
            try:
                num = int(nom[5:])  # après "poeme"
                numeros.append(num)
            except ValueError:
                pass
    
    prochain_num = max(numeros) + 1 if numeros else 1
    return dossier / f"poeme{prochain_num}.txt"


def sauvegarder_texte(texte):
    """Sauvegarde avec nom auto-généré"""
    nom_fichier = get_nom_fichier_suivant()
    with open(nom_fichier, 'w', encoding='utf-8') as fichier:
        fichier.write(texte)
    print(f"✅ Sauvegardé: {nom_fichier}")
    
def lire_poeme(nom_poeme):
    with open(nom_fichier, 'r', encoding='utf-8') as fichier:
        texte = fichier.read()
    return texte

def reunir_poemes():
    """Récupère TOUS les poèmes du dossier en 1 seule string"""
    dossier = Path("poemes")
    if not dossier.exists():
        return "❌ Dossier 'poemes' introuvable"
    
    # Récupère tous les fichiers poeme*.txt
    fichiers = list(dossier.glob("poeme*.txt"))
    if not fichiers:
        return "❌ Aucun poème trouvé dans 'poemes/'"
    
    tous_les_poemes = []
    for fichier in fichiers:
        try:
            with open(fichier, 'r', encoding='utf-8') as f:
                contenu = f.read().strip()
                if contenu:  # Ignore les fichiers vides
                    tous_les_poemes.append(f"=== {fichier.name} ===\n{contenu}")
        except Exception:
            continue
    
    if not tous_les_poemes:
        return "❌ Aucun poème lisible trouvé"
    
    # Concatène avec 10 tirets entre chaque
    separateur = "\n" + "----------" * 10 + "\n"
    return separateur.join(tous_les_poemes)


def supprimer_tous_poemes():
    """
    Supprime TOUS les fichiers du dossier "poemes" (poeme*.txt et autres).
    Garde le dossier vide. Affiche les fichiers supprimés.
    """
    dossier = Path("poemes")
    if not dossier.exists():
        print("❌ Dossier 'poemes' introuvable.")
        return
    
    # Liste tous les fichiers (pas les sous-dossiers)
    fichiers = [f for f in dossier.iterdir() if f.is_file()]
    if not fichiers:
        print("ℹ️  Dossier 'poemes' déjà vide.")
        return
    
    print(f"🗑️  Suppression de {len(fichiers)} fichier(s) :")
    for fichier in fichiers:
        print(f"  - {fichier.name}")
        fichier.unlink()  # Supprime le fichier
    
    print("✅ Tous les poèmes supprimés ! Dossier 'poemes' vidé.")




API_KEY = "FPSX8d28cab7913e5efb35b8af9552b87988"
BASE_URL = "https://api.freepik.com/v1/ai/mystic"
headers = {"x-freepik-api-key": API_KEY}

def generer_image(prompt):
    print(f"🎨 Génération de l'image pour: '{prompt}'")
    
    # Étape 1: Créer le job
    payload = {"prompt": prompt}
    response = requests.post(BASE_URL, headers=headers, json=payload)
    data = response.json()
    
    print("Réponse création:", json.dumps(data, indent=2))
    task_id = data['data']['task_id']
    
    # Étape 2: Polling
    status_url = f"{BASE_URL}/{task_id}"
    print(f"⏳ Suivi du job {task_id}...")
    
    while True:
        resp = requests.get(status_url, headers=headers)
        data = resp.json()
        status = data['data']['status']
        print(f"Statut: {status}")
        
        if status == "COMPLETED":
            generated = data['data']['generated']
            if generated:
                image_url = generated[0]
                print(f"✅ Image prête: {image_url}")
                
                # Téléchargement
                img_data = requests.get(image_url).content
                filename = Path("images générées")/"image.png"
                with open(filename, "wb") as f:
                    f.write(img_data)
                print(f"💾 Image sauvée: {filename}")
                return filename
        elif status in ["FAILED", "ERROR"]:
            print("❌ Échec:", data)
            return None
        
        time.sleep(5)



def extraire_contours(image_path, sortie_path, seuil=50, epaisseur=2):
    # Charger et convertir en gris
    img = Image.open(image_path).convert('L')
    img_array = np.array(img, dtype=np.float32)
    
    # Lissage gaussien avec PIL
    img_lisse = img.filter(ImageFilter.GaussianBlur(radius=1.5))
    img_lisse_array = np.array(img_lisse, dtype=np.float32)
    
    # Détecter les contours avec Sobel (manuel)
    sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float32)
    sobel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=np.float32)
    
    # Convolution manuelle pour gradients
    grad_x = np.zeros_like(img_lisse_array)
    grad_y = np.zeros_like(img_lisse_array)
    
    height, width = img_lisse_array.shape
    for i in range(1, height-1):
        for j in range(1, width-1):
            grad_x[i, j] = (sobel_x * img_lisse_array[i-1:i+2, j-1:j+2]).sum()
            grad_y[i, j] = (sobel_y * img_lisse_array[i-1:i+2, j-1:j+2]).sum()
    
    # Magnitude des gradients (contours)
    contours = np.sqrt(grad_x**2 + grad_y**2)
    contours_bin = contours > seuil
    
    # Épaissir les contours
    kernel = np.ones((epaisseur, epaisseur), dtype=np.uint8)
    contours_epais = np.pad(contours_bin, 1, mode='edge')
    for i in range(epaisseur):
        for j in range(epaisseur):
            contours_epais |= np.roll(np.roll(contours_epais, i-epaisseur//2, axis=0), 
                                    j-epaisseur//2, axis=1)
    
    # Créer image finale
    img_sortie = Image.new('L', img.size, 255)
    draw = ImageDraw.Draw(img_sortie)
    coords = np.column_stack(np.where(contours_epais[1:-1, 1:-1]))
    if len(coords) > 0:
        draw.point(list(zip(coords[:,1], coords[:,0])), fill=0)
    
    img_sortie.save(sortie_path)
    print(f"Contours sauvés dans {sortie_path}")
    return img_sortie







def lancement_cura():
    # ================= CONFIG =================
    DEPTH_MM = 30.0
    WIDTH_MM = 60.0
    # ==========================================

    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    IMAGE_PATH = os.path.join(BASE_DIR+"\images générées", "image_contours.png")
    STL_PATH = os.path.join(BASE_DIR+"\images générées", "lithophanie_Patricia.stl")

    # Load image in grayscale
    img = Image.open(IMAGE_PATH).convert("L")
    heightmap = np.array(img, dtype=np.float32)

    h, w = heightmap.shape

    # Darker = higher  → inversion
    heightmap = 255.0 - heightmap

    # Normalize height to depth
    heightmap = ((heightmap / 255.0) * DEPTH_MM) / 10 # division, sinon le relief est trop exagere

    scale_xy = WIDTH_MM / w

    vertices = []
    faces = []

    for y in range(h):
        for x in range(w):
            vertices.append([
                x * scale_xy,
                y * scale_xy,
                heightmap[y, x]
            ])

    def vid(x, y):
        return y * w + x

    for y in range(h - 1):
        for x in range(w - 1):
            faces.append([vid(x, y), vid(x + 1, y), vid(x, y + 1)])
            faces.append([vid(x + 1, y), vid(x + 1, y + 1), vid(x, y + 1)])

    mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
    mesh.export(STL_PATH)

    # Launch Cura with STL
    cura_exe = r"C:\Program Files\UltiMaker Cura 5.11.0\UltiMaker-Cura.exe"
    subprocess.Popen([cura_exe, STL_PATH])





def envoyer_mail():

    # LISTE TES FICHIERS ICI
    fichiers_a_zipper = [
        "images générées/image.png",
        "images générées/image_contours.png",
    ]
    
    # Crée le ZIP avec TOUS les fichiers
    with zipfile.ZipFile("temp_fichier.zip", 'w') as zipf:
        for fichier in fichiers_a_zipper:
            if os.path.exists(fichier):
                zipf.write(fichier, os.path.basename(fichier))
            else:
                print(f"⚠️  {fichier} non trouvé")
    
    FICHIER = "temp_fichier.zip"
    
    # TES INFOS ICI
    EMAIL = "singergregoire19@gmail.com"
    MDP_APP = "khmr knei cvud vryo"  # SANS ESPACES
    DEST = "g.singer@lecolededesign.com"
    SUJET = "Impression Cura"
    TEXTE = "Bonjour, \n Voici un fichier à imprimer (image_contours.png). \
Ci-joint aussi l'image originale, pour que vous voyiez à quoi elle doit ressembler ! \
Je voudrais du PLA, blanc.\nBonne journée,\nPatricia"
    
    # Message
    msg = MIMEMultipart()
    msg["From"] = EMAIL
    msg["To"] = DEST
    msg["Subject"] = SUJET
    msg.attach(MIMEText(TEXTE, "plain"))
    
    # Fichier joint
    with open(FICHIER, "rb") as f:
        part = MIMEBase("application", "octet-stream")
        part.set_payload(f.read())
    encoders.encode_base64(part)
    part.add_header("Content-Disposition", f"attachment; filename={Path(FICHIER).name}")
    msg.attach(part)
    
    # ENVOI (PORT 587)
    try:
        server = smtplib.SMTP("smtp.gmail.com", 587)
        server.starttls(context=ssl.create_default_context())
        server.login(EMAIL, MDP_APP)
        server.sendmail(EMAIL, DEST, msg.as_string())
        server.quit()
        print("✅ MAIL ENVOYÉ")
    except Exception as e:
        print(f"❌ Erreur: {e}")



from fonctions import *

nom_de_plume = "Noisette"

# Chemin vers le modèle Vosk
MODEL_PATH = os.path.join(os.path.dirname(__file__), "vosk-model-small-fr-0.22")

if not os.path.exists(MODEL_PATH):
    print(f"ERREUR: Modèle non trouvé à {MODEL_PATH}")
    print("Téléchargez depuis https://alphacephei.com/vosk/models")
    input("Appuyez sur Entrée pour quitter...")
    sys.exit(1)

model = Model(MODEL_PATH)
rec = KaldiRecognizer(model, 16000)

q = queue.Queue()

def callback(indata, frames, time, status):
    if status:
        print(status)
    q.put(bytes(indata))


jouer_effet_sonore("bruitages/programme lancé.mp3")
print("Parlez maintenant... (silence pour terminer)")


poeme = ""
generation_image_lancee = False

with sd.RawInputStream(samplerate=16000, blocksize=8000, dtype='int16',
                       channels=1, callback=callback):
    while True:
        data = q.get()
        if rec.AcceptWaveform(data):
            result = json.loads(rec.Result())
            texte = result.get("text", "")
            if texte:
                print("\n✅ Vous avez dit :", texte)
                
                if finir_poeme(texte, nom_de_plume):        # signe "Noisette"
                    sauvegarder_texte(poeme)
                    poeme = ""
                    jouer_effet_sonore("bruitages/enregistré.mp3")
                elif "image" in texte:                      # lancement "Image"
                    generation_image_lancee = True
                    print("Génération : "+texte)
                    jouer_effet_sonore("bruitages/image va se créer.mp3")
                    break
                else:
                    poeme += texte      # sinon, on continue d'écouter le poème !
                    
if generation_image_lancee:
    print("La génération de l'image va se lancer !")
    poemes_unis = reunir_poemes()
    prompt_image = poems_to_image_prompt(poemes_unis)
    print("Le prompt image : "+prompt_image)
    
    print("Création de l'image...")
    generer_image(prompt_image)
    print("Redéfinir les contours...")
    extraire_contours("images générées/image.png", "images générées/image_contours.png", seuil=40, epaisseur=1)

    
    print("Ouverture de l'image sur Cura...")
    lancement_cura()

    envoyer_mail()


supprimer_tous_poemes()

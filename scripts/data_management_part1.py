# -*- coding: utf-8 -*-
"""
Created on 11 Juin 2024

@author: Joris Salmon
site de kaggle : #https://www.kaggle.com/competitions/bike-sharing-demand/data
"""

######################### Correction et description des données #############################

import streamlit as st
import pandas as pd
import os
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns


#path=os.path.dirname(os.path.abspath(__file__))
path="C:/Users/ingemedia/Desktop/ETUDES/DU Sorbonne/Cours Data-Management/Projet"
#Read file
df=pd.read_csv(f"{path}/data/data.csv",sep=",")

#update structure of data
df.rename(columns={
    'datetime': 'date_heure',
    'season': 'saison',
    'holiday': 'jour_ferie',
    'workingday': 'jour_travaille',
    'weather': 'meteo',
    'temp': 'temperature_celsius',
    'atemp': 'temperature_ressentie_celsius',
    'humidity': 'humidite_relative',
    'windspeed': 'vitesse_vent',
    'casual': 'locations_utilisateurs_non_inscrits',
    'registered': 'locations_utilisateurs_inscrits',
    'count': 'total_locations'
}, inplace=True)

print("Affichage des type de données avec le nombre de NA")
df.isna().sum() #pas de valeurs manquantes
df.info() #pas de valeurs manquantes, 10886 valeurs
df.head(5) #visuel
print("Affichage des principales statistiques du DF")
print(df.describe(include='int'))
df["date_heure"]=pd.to_datetime(df["date_heure"])

#Création de variables
##Création de colonnes à partir de la date
df["Annee"]=df["date_heure"].dt.year
df["Trimestre"]=df["date_heure"].dt.quarter
df["Mois"]=df["date_heure"].dt.month
df["Semaine"]=df["date_heure"].dt.isocalendar().week
df["Heure"]=df["date_heure"].dt.hour
df["Jour"]=df["date_heure"].dt.day
df['annee_mois'] = df['date_heure'].dt.strftime('%Y-%m')
df['annee_mois_jour'] = df['date_heure'].dt.strftime('%Y-%m-%d')
##Création de deux nouvelles colonnes
df["temperature_celsius_categ"]=pd.cut(df["temperature_celsius"], bins=([0,5,10,15,20,25,30,35,40,45]))
df["IndiceConfortThermique"]=df["temperature_celsius"]-((0.55-0.0055*df["humidite_relative"])*(df["temperature_celsius"]-14.5))-(df["vitesse_vent"]/10)


# Recodage de certaines variables avec le mapping
saison_mapping = {
    1: "printemps",
    2: "ete",
    3: "automne",
    4: "hiver"
}

meteo_mapping={
    1: "Degage, Quelques nuages, Partiellement nuageux, Partiellement nuageux",
    2: "Brume + Nuageux, Brume + Nuages brises, Brume + Quelques nuages, Brume",
    3: "Legere neige, Legere pluie + Orage + Nuages epars, Legere pluie + Nuages epars",
    4: "Forte pluie + Greleons + Orage + Brume, Neige + Brouillard" 
}
df['saison']=df['saison'].map(saison_mapping)
df['meteo']=df['meteo'].map(meteo_mapping)

# Export en csv des données modifiées
df.to_csv(f"{path}/data/data_modified.csv")

# Visualisation globale des distributions des variables
df.hist(figsize=(20,15))
print("""
      Quelques remarques sur les distributions des variables :
    - il n'y a pas forcément de saison pour les locations
    - on remarque certaines distributions sur la gauche qui peuvent être écrasé par des valeurs aberrantes
    - cela semble suivre une distribution normale si l'on souhaite construire un modèle
      """)



#### Visualisation sur les totaux des locations par mois pour les 2 années d'activités
df_daily=df.groupby('annee_mois')['total_locations'].sum().reset_index()

plt.figure(figsize=(14, 6))
sns.scatterplot(data=df_daily,x='annee_mois',y='total_locations')
plt.xticks(rotation=45)
plt.title('Totaux des locations de vélo par mois depuis le début')
plt.xlabel('Date')
plt.ylabel('Nombre total des locations')
plt.axvline('2012-01', color='red', linestyle='--', lw=1)
y_min, y_max = plt.ylim()
y_max=y_max-15000 
y_min=y_min+15000
plt.text('2011-06', y_max, str(2011), color='red', ha='right', va='center', fontsize=24, fontweight='bold')
plt.text('2012-10', y_min, str(2012), color='red', ha='right', va='center', fontsize=24, fontweight='bold')
plt.show()

#### On va regarder les corrélations des variables numériques entre elles
df_correlation=df.corr(numeric_only=True)
sns.heatmap(data=df_correlation, cmap='viridis')
print("""
      Remarques pour le heatmap des corrélations entre variables numériques : 
      • On remarque des corrélations fortes entre les locations avec les températures, l'ICT et les heures'
      """)
      
#### On veut regarder l'évolution de l'ICT avec le nombre de location de vélo
df_ICT_Nb_Loc=df.groupby('annee_mois')[['IndiceConfortThermique','total_locations']].sum().reset_index()
ICT=df_ICT_Nb_Loc["IndiceConfortThermique"]
Tloc=df_ICT_Nb_Loc["total_locations"]
labels=df_ICT_Nb_Loc["annee_mois"]

fig,ax1=plt.subplots()
color="blue"
ax1.plot(labels, ICT, color=color)
ax1.set_ylabel("ICT", color=color)
ax1.tick_params(axis='y', labelcolor=color)
ax1.tick_params(axis='x', rotation=45)
ax1.spines['left'].set_color(color)
ax1.spines['left'].set_linewidth(3)
ax1.set_title("Indice de Confort Thermique")

ax2 = ax1.twinx()
color="tab:red"
ax2.plot(labels, Tloc, color=color)
ax2.set_ylabel("Tloc", color=color)
ax2.tick_params(axis='y', labelcolor=color)
ax2.spines['right'].set_color(color)
ax2.spines['right'].set_linewidth(3)

#### On peut maintenant regarder le nombre de location pendant les jours fériés et jours travaillées avec leur évolution au cours du temps

###################### Web scraping des données pour le Text Mining #######################################

from selenium import webdriverimport plotly.express as px

df.loc[df['jour_ferie'] == 1, 'ferie'] = df['total_locations']
df.loc[df['jour_travaille'] == 1, 'travaille'] = df['total_locations']
df.loc[(df['jour_travaille'] == 0) & (df['jour_ferie'] == 0), 'autre'] = df['total_locations']

# données pour le graphique sunburst
df_melted = pd.melt(df, id_vars=['Annee', 'Mois'], value_vars=['ferie', 'travaille', 'autre'],
                    var_name='type_jour', value_name='nombre_locations')

# couleurs pour les types de jours
color_mapping = {
    'ferie': '#FF4136',  # Rouge foncé
    'travaille': '#0074D9',  # Bleu clair
    'autre': '#FFDC00'  # Jaune clair
}

mois_mapping = {
    1: 'Janvier',
    2: 'Février',
    3: 'Mars',
    4: 'Avril',
    5: 'Mai',
    6: 'Juin',
    7: 'Juillet',
    8: 'Août',
    9: 'Septembre',
    10: 'Octobre',
    11: 'Novembre',
    12: 'Décembre'
}

# numéros de mois par les noms
if 'Janvier' not in df_melted['Mois']:
    df_melted['Mois'] = df_melted['Mois'].map(mois_mapping)


# On map les couleurs
df_melted['color'] = df_melted['type_jour'].map(color_mapping)

# graphique Sunburst
fig = px.sunburst(df_melted, path=['Annee', 'Mois', 'type_jour'], values='nombre_locations',
                  color='type_jour', color_discrete_map=color_mapping,
                  hover_data=['nombre_locations'],
                  title="Répartition des locations par type de jour et par année/mois")

fig.show()
fig.write_html(f"{path}/graphiques/plotly.html")


from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.options import Options
from selectolax.parser import HTMLParser
from bs4 import BeautifulSoup
import requests
import time
import json
import pandas as pd

# Ouvrir le driver de Selenium pour le Scraping de données
#options = Options()
#options.headless = True 
#options.add_argument('user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36')

#driver = webdriver.Chrome(options=options)


#Initialisation de la liste contenant les dictionnaires, 1 dictionnaire par discussion
#liste_texte=[]
#page_content=None

#Site du Dataset
#url_base="https://kaggle.com/competitions/bike-sharing-demand/discussion?sort=undefined&page="

# Fonction pour extraire les messages de chaque discussion
def extract_discussion_content(url):
    """
    Parameters
    ----------
    url : TYPE : str
        DESCRIPTION : Cette fonctionne récolte les messages de toutes les discussions à partir des pages de discussion du Dataset

    Returns
    -------
    discussions_dict : TYPE
        DESCRIPTION.

    """
    global page_content
    response = driver.get(url)
    time.sleep(10) #On laisse ces timers pour charger la page mais également pour éviter les mesures de scraping
    discussions_dict = {}

    page_content = driver.page_source
    time.sleep(10) 
    
    soup = BeautifulSoup(page_content, 'html.parser')
    
    # Extraire le contenu de la discussion
    discussions = soup.select("div[class='sc-kGLCbq fAhUHg']")
    
    for index,discussion in enumerate(discussions,1):
        discussion_text = discussion.get_text(strip=True)
        discussions_dict[f'Message_{index}'] = discussion_text
    
    return discussions_dict

#On initie un compteur qui va évoluer pour charger les pages suivantes de l'URL
def scrape_pages(max_page):
    i=0 
    while i <= max_page: # On vérifie qu'on est rendu à la page < 8
        i+=1
        url_page=url_base+str(i) # Constitution de l'url avec le numéro de page
        
        print(f"Scraping de la page {i}")
        
        driver.get(url_page) # Le driver s'éxécute sur la page
        time.sleep(10) # temps de chargement de la page
        
        page_main_content = driver.page_source #extaction du code source
        time.sleep(10) #temps de récupération du HTML
        
        soup = BeautifulSoup(page_main_content, 'html.parser') # Initialisation du Parser
        
        # Récupération des liens de toutes les discussions
        liens = soup.select("ul > li[class='MuiListItem-root MuiListItem-gutters MuiListItem-divider sc-drMgrp dllDGS css-iicyhe'] > div > a.sc-fbbrMC.cfgoB")
        
        #On parcourt chaque discussion et on récupère les messages
        for index,lien in enumerate(liens,1):
            print(f"Discussion {index}")
            href = lien.get('href') # lien de la discussion
            if href:
                url = url_base + str(href) # constitution de l'url de chaque message
                donnees=extract_discussion_content(url) #Extraire le dictionnaire de chaque discussion
                liste_texte.append(donnees) #Ajout à la liste des discussions
            
#scrape_pages(1) # On appelle la fonction d'avant avec le paramètre 8 comme max de page

#Nombre_discussions=len(liste_texte)
#Nombre_messages=0

#for i in liste_texte:
#    Nombre_messages+=len(i)
    
#print(f"Il y a {Nombre_discussions} discussions sur ce Dataset")
#print(f"Il y a {Nombre_messages} messages en tout sur ce Dataset")


#corpus=[' '.join(disc.values()) for disc in liste_texte]

# Enregistrer les données par discussions et messages dans un fichier JSON
#with open(f'{path}/data/discussions_data.json', 'w', encoding='utf-8') as f:
#    json.dump(liste_texte, f, ensure_ascii=False, indent=4)

#with open(f'{path}/data/message_data.json', 'w', encoding='utf-8') as f:
#    json.dump(corpus, f, ensure_ascii=False, indent=4)

#print("Extraction et sauvegarde terminées.")

################################## Analyse du text (Workloud) ################################
# Pour le pré processing
from unidecode import unidecode
import re
from nltk.stem import SnowballStemmer
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize


# Les bigrammes
from collections import Counter
from nltk.util import ngrams

# Pour la vectorisation
from sklearn.feature_extraction.text import TfidfVectorizer 
from sklearn.feature_extraction.text import CountVectorizer

# Visualisation Wordcloud
from wordcloud import WordCloud


nltk.download('punkt')
# Définition de la liste de stop words considérés (celle de spacy + lettres)
nltk.download('stopwords')

#Lecture de notre texte
with open(f'{path}/data/message_data.json', 'r',encoding='utf-8') as f:
    text=json.load(f)


# Liste des stop words en anglais
stop_words = set(stopwords.words('english'))

stopWords = [unidecode(sw) for sw in stop_words]

# Création du stemmer
stemmer = SnowballStemmer('english')

# Création d'une fonction pour supprimer les sw
def no_stop_word(string, stopWords):

    """
    Supprime les stop words d'un texte.

    Paramètres
    ----------

    string : chaine de caractère.

    stopWords : liste de mots à exclure. 
    
    ----------
    Sortie : string sans stopWords
    """
    
    string=' '.join([word for word in string.split() if word not in stopWords])
    
    
    return string

# Création d'une fonction pour stemmatiser chaque mot d'un text 
def stemmatise_text(text, stemmer):
    """Stemmatise un texte : Ramène les mots d'un texte à leur racine (peut créer des mots qui n'existe pas).

    Paramètres
    ----------
    text : Chaine de caractères.

    stemmer : Stemmer de NLTK.
    
    ----------
    Sortie : string qui contient la forme stemmatisée des mots
    """
    
    string = " ".join([stemmer.stem(word) for word in text.split()])
    return string

def stem_cleaner(pandasSeries, stemmer, stopWords):
    
    print("#### Nettoyage en cours du corpus ####") # Mettre des print vous permet de comprendre où votre code rencontre des problèmes en cas de bug
    
    # confirmation que chaque article est bien de type str
    pandasSeries = pandasSeries.apply(lambda x : str(x))
    
    # Passage en minuscule
    print("... Passage en minuscule") 
    pandasSeries = pandasSeries.apply(lambda x : x.lower())
    
    # Suppression des accents
    print("... Suppression des accents") 
    pandasSeries = pandasSeries.apply(lambda x : unidecode(x))
    
    # Changement de chaque année numérique en 'annee' en utilisant une regex
    print("... Détection du champs année") 
    pandasSeries = pandasSeries.apply(lambda x : re.sub("[0:9]{4}","annee",x))
    
    # Suppression des caractères spéciaux et numériques
    # Garder uniquement les lettres a-z en utilisant une regex
    print("... Suppression des caractères spéciaux et numériques") 
    pandasSeries = pandasSeries.apply(lambda x : re.sub("[^a-z]+"," ",x))
    
    # Suppression des stop words (appliquer la fonction no_stop_word créée ci-dessus)
    print("... Suppression des stop words") 
    pandasSeries = pandasSeries.apply(lambda x : no_stop_word(x, stopWords))
    
    # Stemmatisation (appliquer la fonction stemmatise_text créée ci-dessus)
    print("... Stemmatisation") 
    pandasSeries = pandasSeries.apply(lambda x : stemmatise_text(x,stemmer))
    
    print("#### Nettoyage OK! ####")

    return pandasSeries

#Exécution
text_stem = stem_cleaner(pd.Series(text), stemmer, stopWords)
text_stem.head()

text_stem_full=' '.join(text_stem)
text_stem_mot = re.findall(r"\w+", text_stem_full) #liste des mots


liste_text_stem=set(text_stem_mot) # collection des mots présents unique dans le texte

#Vectoriser pour modéliser
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(text_stem)
#print(X.toarray())

#Visualisation
wordcloud = WordCloud(
    width=800,
    height=400,
    colormap='plasma',
    background_color='white',
).generate(text_stem_full)

plt.figure(figsize=(10, 5))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.show()



################# Site du Dataset
#https://www.kaggle.com/competitions/bike-sharing-demand/data



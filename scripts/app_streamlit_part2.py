#Import
import streamlit as st
import pandas as pd
import numpy as np
import cufflinks as cf
import seaborn as sns
import plotly.express as px
import plotly as plt
import plotly.graph_objects as go
import matplotlib.pyplot as plt

#Titre
st.markdown("""
    <style>
    .main-title {
        background-color: #76c7c0;  /* Utilisation de la même couleur que dans le tableau */
        padding: 10px;
        border-radius: 10px;
        text-align: center;
        color: #FFFFFF;
        font-size: 36px;
        font-weight: bold;
        width: 100%;
        margin: 0;
    }

    </style>
""", unsafe_allow_html=True)

# Titre principal
st.markdown("<div class='main-title'>Location de vélo de la ville</div>", unsafe_allow_html=True)

#Import csv
path="C:/Users/ingemedia/Desktop/ETUDES/DU Sorbonne/Cours Data-Management/Projet"
#Read file
df=pd.read_csv(f"{path}/data/data_modified.csv",sep=",")

#Page 1 : Description de données
def page1():
    st.subheader("Description des données")
    st.markdown("[Cliquez ici](https://www.kaggle.com/competitions/bike-sharing-demand/data) pour accéder au jeu de données sur Kaggle.")
    st.markdown("Sujet : données sur les locations horaires de vélo d'une ville sur une période de deux ans")

    # Descritption du jeu de données initial
    st.subheader("Jeu de données initial")
    data_initial = {
        "Nom": ["datetime", "season", "holiday", "workingday", "weather", "temp", "atemp", "humidity", "windspeed", "casual", "registered", "count"],
        "Type": ["datetime", "int", "int", "int", "int", "float", "float", "int", "float", "int", "int", "int"],
        "Description": ["Date et heure", "Saison (1:hiver, 2:printemps, 3:été, 4:automne)", "Indicateur de jour férié", "Indicateur de jour travaillé", "Conditions météorologiques (1:clair, 2:nuageux, 3:pluie, 4:orage)", "Température (Celsius)", "Température ressentie (Celsius)", "Humidité (%)", "Vitesse du vent (km/h)", "Nombre de locations non enregistrées", "Nombre de locations enregistrées", "Nombre total de locations"]
    }
    
    df_initial = pd.DataFrame(data_initial)
    st.markdown("Nombre de lignes : 10886")
    st.markdown(f"Nombre de variables : {len(data_initial['Nom'])}")
    st.markdown("Valeurs manquantes : 0")

    st.table(df_initial.style.set_table_styles([{'selector': 'th','props': [('background-color', '#76c7c0'), ('color', 'white')]}, {'selector': 'td','props': [('background-color', 'white'), ('color', '#76c7c0')]}]).set_properties(**{'text-align': 'center'}))

    # Descritption du jeu de données après transformation
    st.subheader("Jeu de données après transformation")
    data_transforme = {
        "Nom": ["date_heure", "saison", "jour_ferie", "jour_travaille", "meteo", "temperature_celsius", "temperature_ressentie_celsius", "humidite_relative", "vitesse_vent", "locations_utilisateurs_non_inscrits", "locations_utilisateurs_inscrits", "total_locations", "Annee", "Trimestre", "Mois", "Semaine", "Heure", "Jour", "annee_mois", "annee_mois_jour", "temperature_celsius_categ", "IndiceConfortThermique"],
        "Type": ["datetime", "str", "int", "int", "str", "float", "float", "int", "float", "int", "int", "int", "int", "int", "int", "int", "int", "int", "str", "str", "category", "float"],
        "Description": ["Date et heure", "Saison (printemps, été, automne, hiver)", "Indicateur de jour férié", "Indicateur de jour travaillé", "Conditions météorologiques (clair, nuageux, luie, orage)", "Température (Celsius)", "Température ressentie (Celsius)", "Humidité (%)", "Vitesse du vent (km/h)", "Nombre de locations non inscrites", "Nombre de locations inscrites", "Nombre total de locations", "Année", "Trimestre de l'année", "Mois de l'année", "Semaine de l'année", "Heure de la journée", "Jour de l'année", "Année et mois", "Année, mois et jour", "Catégorie de température (Celsius)", "Indice de confort thermique"]
    }
    df_transforme = pd.DataFrame(data_transforme)
    st.markdown("Nombre de lignes : 10886")
    st.markdown(f"Nombre de variables : {len(data_transforme['Nom'])}")
    st.markdown("Valeurs manquantes : 0")

    st.table(df_transforme.style.set_table_styles([{'selector': 'th','props': [('background-color', '#76c7c0'), ('color', 'white')]}, {'selector': 'td','props': [('background-color', 'white'), ('color', '#76c7c0')]}]).set_properties(**{'text-align': 'center'}))

    # Visualisation globale des distributions des variables
    st.subheader("Histogramme des distibutions")
    df.hist(figsize=(20,15))
    print("""
       Quelques remarques sur les distributions des variables :
     - il n'y a pas forcément de saison pour les locations
     - on remarque certaines distributions sur la gauche qui peuvent être écrasé par des valeurs aberrantes
     - cela semble suivre une distribution normale si l'on souhaite construire un modèle
      """)
    #Wordcloud
    st.subheader("Wordcloud")
    st.markdown("Voici le wordcloud réalisé a partir des mots les plus cité dans les discussions du jeu de données")
    st.image(f"{path}/graphiques/wordcloud.png")



#Page 2
def page2():
    st.subheader("Statistiques descriptives")
 
    st.subheader('Statistiques descriptives univariées:')
    # Tableau variables quantitatives
    numeric_columns = df.select_dtypes(include=['float64', 'int64']).columns
    numeric_columns = numeric_columns.drop(['Unnamed: 0', 'jour_ferie', 'jour_travaille', 'Annee'])
    desc_stats = df[numeric_columns].describe().transpose()
    desc_stats['median'] = df[numeric_columns].median()
    desc_stats = desc_stats[['mean', 'median', 'std', 'min', 'max']]
    st.table(desc_stats)
    # Calcul du nombre de jours fériés
    num_jours_feries = df[df['jour_ferie'] == 1]['Jour'].nunique()
    st.markdown(f" Nombre de jours fériés : {num_jours_feries}")
    # Nombre d'observations par ans
    observations_par_annee = df['Annee'].value_counts().sort_index()
    st.markdown("##### Nombre d'observations par année")
    st.table(observations_par_annee)
    # Distribution de 'meteo'
    st.markdown("#### #Distribution de la variable 'meteo'")
    meteo_counts = df['meteo'].value_counts()
    st.table(meteo_counts)
    



    st.subheader('Statistiques descriptives bivariées :')
    # Corrélations entre les variables quantitatives
    st.subheader(" Corrélations entre les variables quantitatives")
    numeric_df = df[numeric_columns]
    corr = numeric_df.corr()
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(corr, annot=True, cmap='coolwarm', center=0, ax=ax)
    ax.set_title("Matrice de corrélation")
    st.pyplot(fig)
    

# Page 3
def page3():
    st.subheader("Visualisation des données")


    # 1) Serie temporelle du nombre de location par jour


    st.subheader("Evolution du nombre de location de vélo entre le 2011 et 2012")
    # Convertir en datetime
    df['date_heure'] = pd.to_datetime(df['date_heure'])
    # Créer une nouvelle colonne date sans l'heure
    df['date'] = df['date_heure'].dt.date

    # Widget selectbox pour sélectionner inscrit, non inscrit ou total
    statut = st.selectbox("Sélectionnez le type d'utilisateur", ["Tous", "Inscrit", "Non inscrit"])
    
    # Filtrage des données en fonction widget selectionné + calcul de la somme par jour
    if statut == "Inscrit":
        df_filtrer = df[['date', 'locations_utilisateurs_inscrits']].groupby('date').sum().reset_index()
        y = 'locations_utilisateurs_inscrits'
        title = 'Évolution des locations des utilisateurs inscrits entre 2011 et 2012'
    elif statut == "Non inscrit":
        df_filtrer = df[['date', 'locations_utilisateurs_non_inscrits']].groupby('date').sum().reset_index()
        y = 'locations_utilisateurs_non_inscrits'
        title = 'Évolution des locations des utilisateurs non inscrits entre 2011 et 2012'
    else:
        df_filtrer = df[['date', 'total_locations']].groupby('date').sum().reset_index()
        y = 'total_locations'
        title = 'Évolution des locations totales entre 2011 et 2012'
    
    fig = px.line(df_filtrer, x='date', y=y, title=title, color_discrete_sequence=["#76c7c0"])
    st.plotly_chart(fig)

    st.info("Ce graphique permet de voir une augmentation des locations de vélos entre 2011 et 2012. On observe également une saisonnalité sur ces deux années : les locations de vélos augmentent de janvier pour atteindre un pic en juillet, puis descendent jusqu'à janvier. ")



    # 2) Nombre de location par heure de la journée que choisi l'utilisateur



    st.subheader("Evolution du nombre de location de vélo dans une journée")
    # Widget date avec calendrier
    date_selection = st.date_input("Sélectionnez une date", min_value=pd.to_datetime('2011-01-01'), max_value=pd.to_datetime('2012-12-19'))
    # Filtrer les données en fonction de la date sélectionnée
    df_selection = df[df['date'] == date_selection]
    
    # Calculer le nombre de location par heure de la journée choisi + la moyennes pour comparaison
    if statut == "Inscrit":
        df_selection = df_selection.groupby('Heure')['locations_utilisateurs_inscrits'].sum().reset_index()
        moyenne_heur = df.groupby('Heure')['locations_utilisateurs_inscrits'].mean().reset_index()
        y = 'locations_utilisateurs_inscrits'
        title = f'Nombre de locations par heure de la journée du {date_selection} (Utilisateurs inscrits) et moyenne par heure'
    elif statut == "Non inscrit":
        df_selection = df_selection.groupby('Heure')['locations_utilisateurs_non_inscrits'].sum().reset_index()
        moyenne_heur = df.groupby('Heure')['locations_utilisateurs_non_inscrits'].mean().reset_index()
        y = 'locations_utilisateurs_non_inscrits'
        title = f'Nombre de locations par heure de la journée du {date_selection} (Utilisateurs non inscrits) et moyenne par heure'
    else:
        df_selection = df_selection.groupby('Heure')['total_locations'].sum().reset_index()
        moyenne_heur = df.groupby('Heure')['total_locations'].mean().reset_index()
        y = 'total_locations'
        title = f'Nombre de locations par heure de la journée du {date_selection} (Total utilisateurs) et moyenne par heure'
    
    # Créer le graph + graph comparatif
    fig = px.line(df_selection, x='Heure', y=y, title=title, color_discrete_sequence=["#76c7c0"])
    fig.add_scatter(x=moyenne_heur['Heure'], y=moyenne_heur[y], mode='lines', name='Moyenne par heure')

    st.plotly_chart(fig)


    # 3) Nombre de location par type de météo 


    st.subheader("Impact des conditions météorologiques sur les locations de vélos")
    # Widget selection var météo
    variable_meteo = st.selectbox("Sélectionnez une variable météo", options=["meteo", "temperature_celsius", "humidite_relative", "vitesse_vent", "IndiceConfortThermique"], key="meteo_variable")

    # Filtrage des données en fonction des conditions météorologiques sélectionnées
    if variable_meteo == "meteo":
        condition_value = st.selectbox("Sélectionnez la condition météo", options=df['meteo'].unique())
        df_condition = df[df['meteo'] == condition_value]
        titre = f"Moyenne {condition_value}"
    elif variable_meteo == "temperature_celsius":
        temps_plage = st.slider("Sélectionnez la plage de température", float(df['temperature_celsius'].min()), float(df['temperature_celsius'].max()), (10.0, 20.0))
        df_condition = df[(df['temperature_celsius'] >= temps_plage[0]) & (df['temperature_celsius'] <= temps_plage[1])]
        titre = f"Température entre {temps_plage[0]}°C et {temps_plage[1]}°C"
    elif variable_meteo == "humidite_relative":
        humidite_plage = st.slider("Sélectionnez une plage d'humidité", float(df['humidite_relative'].min()), float(df['humidite_relative'].max()), (30.0, 70.0))
        df_condition = df[(df['humidite_relative'] >= humidite_plage[0]) & (df['humidite_relative'] <= humidite_plage[1])]
        titre = f"Humidité entre {humidite_plage[0]}% et {humidite_plage[1]}%"
    elif variable_meteo == "vitesse_vent":
        vent_plage = st.slider("Sélectionnez une plage pour la vitesse du vent", float(df['vitesse_vent'].min()), float(df['vitesse_vent'].max()), (0.0, 20.0))
        df_condition = df[(df['vitesse_vent'] >= vent_plage[0]) & (df['vitesse_vent'] <= vent_plage[1])]
        titre = f"Vitesse du vent entre {vent_plage[0]} km/h et {vent_plage[1]} km/h"
    elif variable_meteo == "IndiceConfortThermique":
        confort_selection = st.slider("Sélectionnez une plage pour l'indice de confort thermique", float(df['IndiceConfortThermique'].min()), float(df['IndiceConfortThermique'].max()), (5.0, 25.0))
        df_condition = df[(df['IndiceConfortThermique'] >= confort_selection[0]) & (df['IndiceConfortThermique'] <= confort_selection[1])]
        titre = f"Indice de confort thermique entre {confort_selection[0]} et {confort_selection[1]}"
    
    # Calcul des moyennes (moyenne par heure)
    mean_total = df['total_locations'].mean()
    mean_condition = df_condition['total_locations'].mean()

    # graphique en barres
    data = {'Condition': ['Moyenne générale', titre],'Moyenne des locations': [mean_total, mean_condition]}
    fig_bar = px.bar(data, x='Condition', y='Moyenne des locations', title="Nombre moyen de vélos loués par heure et nombre moyen par type de météo", color_discrete_sequence=["#76c7c0"])
    st.plotly_chart(fig_bar)



    # 4) Diagramme circulaire


    # Créer des colonnes pour les graphiques
    col1, col2 = st.columns(2)
    col3, col4 = st.columns(2)

    # locations par type d'utilisateur
    total_locations_inscrits = df['locations_utilisateurs_inscrits'].sum()
    total_locations_non_inscrits = df['locations_utilisateurs_non_inscrits'].sum()
    repartition_utilisateur = pd.DataFrame({
        'type_utilisateur': ['Inscrit', 'Non inscrit'],
        'total_locations': [total_locations_inscrits, total_locations_non_inscrits]
    })
    repartition_utilisateur['percentage'] = (repartition_utilisateur['total_locations'] / repartition_utilisateur['total_locations'].sum()) * 100

    with col1:
        st.subheader("Répartition des locations totale par type d'utilisateur")
        fig1 = px.pie(repartition_utilisateur, names='type_utilisateur', values='percentage', title='Location par type d\'utilisateur', color_discrete_sequence=["#76c7c0", "#a3d39c"])
        fig1.update_layout(width=500, height=300)
        st.plotly_chart(fig1)

    # locations par saison
    repartition_saison = df.groupby('saison')['total_locations'].sum().reset_index()

    with col2:
        st.subheader("Répartition des locations totale par saison")
        fig2 = px.pie(repartition_saison, names='saison', values='total_locations', title='Location par saison', color_discrete_sequence=px.colors.sequential.Teal)
        fig2.update_layout(width=500, height=300)
        st.plotly_chart(fig2)
    
    # locations par condition météorologique
    repartition_meteo = df.groupby('meteo')['total_locations'].mean().reset_index()

    with col3:
        st.subheader("Répartition des locations par type de météo")
        fig3 = px.bar(repartition_meteo, x='meteo', y='total_locations', title='Nombre moyen de locations (pour une heure)', 
                    labels={'total_locations': 'Nombre moyen de locations', 'meteo': 'Condition météorologique'},
                    color_discrete_sequence=["#76c7c0"])
        fig3.update_layout(xaxis_tickangle=70,width=400, height=500)
        st.plotly_chart(fig3)


    # locations par catég de température
    repartition_temperature = df.groupby('temperature_celsius_categ')['total_locations'].mean().reset_index()

    with col4:
        st.subheader("Répartition par catégorie de température")
        fig4 = px.bar(repartition_temperature, x='temperature_celsius_categ', y='total_locations', title='Nombre moyen de locations (pour une heure)',
                    labels={'total_locations': 'Nombre moyen de locations', 'temperature_celsius_categ': 'Catégorie de température'},
                    color_discrete_sequence=["#76c7c0"])
        fig4.update_layout(width=500, height=400)
        st.plotly_chart(fig4)


    # 5) Diagramme nombre de location et ICT

    ##ici nous avons remis le code fait dans le data-management mais en utilisant plotly
    # Calcul de l'ICT et du nombre de locations par mois
    df_ICT_Nb_Loc = df.groupby('annee_mois')[['IndiceConfortThermique', 'total_locations']].sum().reset_index()

    labels = df_ICT_Nb_Loc["annee_mois"]
    ICT = df_ICT_Nb_Loc["IndiceConfortThermique"]
    Tloc = df_ICT_Nb_Loc["total_locations"]

    # Création du graphique avec deux axes y
    fig = go.Figure()

    # Ajout de la trace pour l'ICT
    fig.add_trace(go.Scatter(x=labels, y=ICT, name="ICT", line=dict(color="#76c7c0"), yaxis="y1"))

    # Ajout de la trace pour le nombre total de locations
    fig.add_trace(go.Scatter(x=labels, y=Tloc, name="Total Locations", line=dict(color="#a3d39c"), yaxis="y2"))

    # Configuration des axes
    fig.update_layout(
        title="Évolution de l'ICT avec le nombre de locations de vélos",
        xaxis=dict(title="Mois"),
        yaxis=dict(title="ICT", titlefont=dict(color="#76c7c0"), tickfont=dict(color="#76c7c0")),
        yaxis2=dict(title="Total Locations", titlefont=dict(color="#a3d39c"), tickfont=dict(color="#a3d39c"), overlaying="y", side="right"),
        legend=dict(x=0.1, y=1.1, orientation="h")
    )

    # Affichage du graphique
    st.plotly_chart(fig)






# Sidebar navigation
page = st.sidebar.radio("" ,["Acceuil","Statistiques descriptives", "Visualisations"])

# Display selected page
if page == "Acceuil":
    page1()
elif page == "Statistiques descriptives":
    page2()
elif page == "Visualisations":
    page3()
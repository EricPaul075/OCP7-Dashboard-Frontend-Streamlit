import streamlit as st
import plotly.graph_objects as go
import requests
import shutil
import os

# Variables globales
data_path = './data/'
tmp = data_path + 'tmp/'
if not os.path.exists(tmp):
    os.makedirs(tmp)
url_server = 'https://ocp7-dbbackend.herokuapp.com'  # pour fonctionnement par défaut de fastAPI
timeout = 30  # valeur maximum du timeout de requête

# Configuration de l'affichage sur toute la largeur de la page web
st.set_page_config(page_title="OC P7 Dashboard", layout="wide")


@st.cache(persist=True)
def load_id_list():
    """
    Charge la liste des numéros d'identification client.
    :return: list, liste des numéros d'identification.
    """
    r = requests.get(url_server + '/clients_list', timeout=timeout)
    return r.json()['id_list']

clients_id_list = load_id_list()


# Listes des features établies une seule fois
@st.cache(persist=True)
def get_feature_lists():
    """
    Charge des listes des features.
    :return: list, list, list:
        - features_list: liste de l'ensemble des features.
        - cat_col: liste des features catégorielles.
        - num_col: liste des features numériques.
    """
    r = requests.get(url_server + '/feature_lists', timeout=timeout)
    features_list = r.json()['all']
    cat_col = r.json()['cat']
    num_col = r.json()['num']
    return features_list, cat_col, num_col

features_list, cat_col, num_col = get_feature_lists()


def get_feature_selection_list(client_id, is_wf, filter):
    """
    Etablit la liste des features en vue du menu de sélection.
    :param client_id: int, N° d'identification client
    :param is_wf: bool, si la liste est à ordonner selon
        l'impact (décroissant) des features pour le client.
    :param filter: list, filtre éventuellement la liste:
        - 'all': toutes les features ;
        - 'current': features de la demande de prêt ;
        - 'previous': features des prêts antérieurs.
    :return: list, list de features
    """
    query_param = {'is_wf': is_wf, 'filter': filter}
    r = requests.get(url_server + '/' + str(client_id) + '/feature_selection',
                     params=query_param, timeout=timeout)
    return r.json()['feature_selection']


def set_gauge(client_id):
    """
    Graphique de jauge affichant le score de la prédiction
        du modèle relative à la demande de prêt.
    :param client_id: int , N° d'identification client.
    :return: plotly.graph_objects.Figure
    """
    r = requests.get(url_server + '/' + str(client_id))
    score = r.json()['score']
    result = 'Crédit non accepté' if score >= 0.5 else 'Crédit accepté'
    fig_gauge = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=score,
        domain={'x': [0, 0.75], 'y': [0.25, 1]},  # zone du graphique dans la fenêtre
        title={'text': f"{result}", 'font': {'size': 28}},
        delta={'reference': 0.5, 'increasing': {'color': "rgb(255,0,81)"}, 'decreasing': {'color': "rgb(0,139,251)"}},
        gauge={
            'axis': {'range': [0, 1], 'tickwidth': 1, 'tickcolor': "rgb(49,51,63)"},
            'bar': {'color': "rgb(49,51,63)"},
            'bgcolor': 'rgba(0,0,0,0)',
            'borderwidth': 1,
            'bordercolor': "black",
            'steps': [
                {'range': [0, 0.5], 'color': 'rgb(0,139,251)'},
                {'range': [0.5, 1], 'color': 'rgb(255,0,81)'}]}))
    return fig_gauge


def graph_features_global_impact(max_feat):
    """
    Crée l'image du graphique d'impact globale des features
        selon la moyenne (abs) des valeurs de Shapley de
        l'ensemble de la population.
    :param max_feat: int, nombre de features à afficher sur
        le graphique.
    :return: str, chemin du fichier de l'image.
    """
    filepath = tmp + f"gfgi_{max_feat}.png"
    if not os.path.exists(filepath):
        query_param = {'max_feat': max_feat}
        r = requests.get(url_server + '/global_impact', stream=True,
                         params=query_param, timeout=timeout)
        with open(filepath, 'wb') as f:
            shutil.copyfileobj(r.raw, f)
    return filepath


def graph_features_local_impact(client_id, max_feat):
    """
    Crée l'image du graphique d'impact local des features
        selon les valeurs de Shapley de l'échantillon
        correspondant au client.
    :param client_id: int , N° d'identification client.
    :param max_feat: int, nombre de features à afficher sur
        le graphique.
    :return: str, chemin du fichier de l'image.
    """
    filepath = tmp + f"gfli_{client_id}_{max_feat}.png"
    if not os.path.exists(filepath):
        query_param = {'max_feat': max_feat}
        r = requests.get(url_server + '/' + str(client_id) + '/local_impact',
                         stream=True, params=query_param, timeout=timeout)
        with open(filepath, 'wb') as f:
            shutil.copyfileobj(r.raw, f)
    return filepath


def graph_feature(client_id, feature):
    """
    Crée l'image du graphique du nuage de points des
        valeurs de Shapleys d'une feature en fonction
        de sa valeur, pour l'ensemble de la population.
        Les points sont colorés en fonction de
        l'influence de la feature ayant le plus fort
        effet d'interaction avec la feature considérée.
    :param client_id: int , N° d'identification client.
    :param feature: str, nom de la feature.
    :return: str, chemin du fichier de l'image.
    """
    f_idx = features_list.index(feature)
    filepath = tmp + f"feature_{client_id}_{f_idx}.png"
    if not os.path.exists(filepath):
        query_param = {'feature': feature}
        r = requests.get(url_server + '/' + str(client_id) + '/feature',
                         stream=True, params=query_param, timeout=timeout)
        with open(filepath, 'wb') as f:
            shutil.copyfileobj(r.raw, f)
    return filepath


def bivar(feature_1, feature_2):
    """
    Crée l'image de l'analyse bivariée entre les features:
        - pairplot si les 2 variables sont numériques ;
        - heatmap si les 2 variables sont catégorielles ;
        - anova si l'une est catégorielle et l'autre numérique.
    :param feature_1: str, nom de la feature.
    :param feature_2: str, nom de la feature.
    :return: str, str:
        - filepath: chemin du fichier image;
        - img_size: attribut ('normal', 'large') pour
            l'affichage.
    """
    f1_idx = features_list.index(feature_1)
    f2_idx = features_list.index(feature_2)
    filepath_1 = tmp + f"bivar{f1_idx}_{f2_idx}.png"
    filepath_2 = tmp + f"bivar{f2_idx}_{f1_idx}.png"
    if os.path.exists(filepath_1):
        filepath = filepath_1
    elif os.path.exists(filepath_2):
        filepath = filepath_2
    else:
        filepath = tmp + f"bivar{f1_idx}_{f2_idx}.png"
        query_param = {'feature_1': feature_1, 'feature_2': feature_2}
        r = requests.get(url_server + '/graph_bivar', stream=True,
                         params=query_param, timeout=60)
        with open(filepath, 'wb') as f:
            shutil.copyfileobj(r.raw, f)
    img_size = 'normal'
    if (feature_1 in cat_col and feature_2 in num_col) or (feature_1 in num_col and feature_2 in cat_col):
        img_size = 'large'
    return filepath, img_size


# Découpage de la page web en conteneurs (horizontaux)
header = st.container()
client = st.container()
empty_section = st.container()
features_impact = st.container()
feature_selection = st.container()

with header:
    st.title("Analyse de crédit client")

with client:
    id_col, empty_col, score_col = st.columns([1, 1.5, 2.5], gap='large')

    # Identification du client
    id_col.header('Client')
    client_id = id_col.selectbox(
        label="Entrez un N° client ou sélectionnez-le dans la liste:",
        options=clients_id_list,
        index=0)

    # Colonne vide
    empty_col.empty()

    # Jauge de résultat du crédit
    score_col.header("Situation d'acceptation du crédit")
    score_col.plotly_chart(set_gauge(client_id))

with features_impact:
    gl_col, lc_col = st.columns(2, gap='large')

    # Impact global des features
    gl_col.header('Impact global des features')
    gl_max_feat = gl_col.slider('Nombre de features', min_value=5, max_value=30, value=20)
    gl_col.image(graph_features_global_impact(gl_max_feat))

    # Impact local des features
    lc_col.header('Impact des features sur le score client')
    lc_max_feat = lc_col.slider('Nombre de features', min_value=5, max_value=30, value=16)
    lc_col.image(graph_features_local_impact(client_id, max_feat=lc_max_feat))

    st.image(data_path + 'blank_space.png', use_column_width=True)

with feature_selection:
    st.header('Sélectionnez 2 features pour analyse')
    col_feat_1, col_feat_2 = st.columns(2, gap='large')

    # Feature 1
    # Input utilisateur feature 1
    col_feat_1.subheader('Feature 1:')
    is_wf_1 = col_feat_1.checkbox(
        "Liste par ordre d'importance pour le client",
        value=True,
        key='is_wf_1')
    f_sublist_options_1 = [
        'Features de la demande de prêt',
        'Feature des prêts antérieurs',
        'Toutes les features']
    filter_1 = col_feat_1.radio(
        label='Affiner la liste des features',
        options=f_sublist_options_1,
        index=0,
        key='filter_1')

    # Liste de sélection selon l'input pour feature 1
    if filter_1=='Features de la demande de prêt': filter_1 = 'current'
    elif filter_1=='Feature des prêts antérieurs': filter_1 = 'previous'
    else: filter_1 = 'all'
    features_list_1 = get_feature_selection_list(client_id, is_wf_1, filter_1)

    # Sélection de feature 1 dans la liste déroulante
    feature_1 = col_feat_1.selectbox(
        label="Nom de la feature 1:",
        options=features_list_1,
        index=0,
        key='feature_1')

    # Affichage graphique feature 1
    col_feat_1.image(graph_feature(client_id, feature_1))

    # Feature 2
    # Input utilisateur feature 2
    col_feat_2.subheader('Feature 2:')
    is_wf_2 = col_feat_2.checkbox(
        "Liste par ordre d'importance pour le client",
        value=True,
        key='is_wf_2')
    f_sublist_options_2 = [
        'Features de la demande de prêt',
        'Feature des prêts antérieurs',
        'Toutes les features']
    filter_2 = col_feat_2.radio(
        label='Affiner la liste des features',
        options=f_sublist_options_2,
        index=0,
        key='filter_2')

    # Liste de sélection selon l'input pour feature 2
    if filter_2=='Features de la demande de prêt': filter_2 = 'current'
    elif filter_2=='Feature des prêts antérieurs': filter_2 = 'previous'
    else: filter_2 = 'all'
    features_list_2 = get_feature_selection_list(client_id, is_wf_2, filter_2)

    # Sélection de feature 2 dans la liste déroulante
    feature_2 = col_feat_2.selectbox(
        label="Nom de la feature 2:",
        options=features_list_2,
        index=0,
        key='feature_2')

    # Affichage graphique feature 2
    col_feat_2.image(graph_feature(client_id, feature_2))

    st.image(data_path + 'blank_space.png', use_column_width=True)

    # Graphe bivarié
    if feature_1!=feature_2:
        img, size = bivar(feature_1, feature_2)
        if size == 'large': bivar_col, _ = st.columns([3, 1])
        else: bivar_col, _ = st.columns(2)
        bivar_col.subheader("Analyse bivariée entre les 2 features")
        bivar_col.image(img)

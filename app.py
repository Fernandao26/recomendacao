"""
Versão otimizada para Streamlit Cloud
Carrega apenas as linhas necessárias do dataset
"""
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import os

warnings.filterwarnings('ignore')

# Configuração da página Streamlit
st.set_page_config(
    page_title="🎬 Recomendador de Filmes/Séries",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================================
# CACHE: Carrega dados com otimização
# ============================================================================
@st.cache_data
def load_data():
    """Carrega dados com otimização de memória"""
    try:
        # Ler apenas colunas necessárias
        columns_needed = ['Title', 'Director', 'Cast', 'Country', 'Type']
        df = pd.read_csv('Netflix Dataset.csv', 
                        usecols=columns_needed,
                        dtype={'Type': 'category', 'Country': 'category'})
        
        # Remover linhas com título vazio
        df = df.dropna(subset=['Title'])
        
        st.success(f"✅ Dataset carregado: {len(df)} títulos")
        return df
    except Exception as e:
        st.error(f"❌ Erro ao carregar: {str(e)}")
        return None

@st.cache_data
def prepare_recommendation_model(df):
    """Prepara modelo com otimização"""
    try:
        df_recomm = df.copy()
        
        # Preencher NaN
        for col in df_recomm.columns:
            df_recomm[col] = df_recomm[col].fillna('')
        
        # Limpar dados
        def clean_data(x):
            if isinstance(x, str):
                return str.lower(x.replace(" ", "")).replace(',', ' ')
            return ''
        
        df_recomm['Director'] = df_recomm['Director'].apply(clean_data)
        df_recomm['Cast'] = df_recomm['Cast'].apply(clean_data)
        df_recomm['Type'] = df_recomm['Type'].apply(clean_data)
        
        # Criar tags
        df_recomm['Tags'] = (df_recomm['Director'] + ' ' + 
                            df_recomm['Cast'] + ' ' + 
                            df_recomm['Country'] + ' ' + 
                            df_recomm['Type'])
        
        # TF-IDF
        tfidf = TfidfVectorizer(stop_words='english', max_features=3000)
        tfidf_matrix = tfidf.fit_transform(df_recomm['Tags'])
        cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
        
        # Índices
        indices = pd.Series(df_recomm.index, index=df_recomm['Title']).drop_duplicates()
        
        return cosine_sim, indices, df
    except Exception as e:
        return None, None, None

def get_recommendations(title, cosine_sim, indices, df, num_recommendations=5):
    """Retorna recomendações"""
    try:
        if title not in indices.index:
            return None, None
        
        idx = indices[title]
        sim_scores = list(enumerate(cosine_sim[idx]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:num_recommendations+1]
        
        movie_indices = [i[0] for i in sim_scores]
        scores = [i[1] for i in sim_scores]
        
        return df['Title'].iloc[movie_indices], scores
    except:
        return None, None

# ============================================================================
# INTERFACE
# ============================================================================

st.title("🎬 Recomendador de Filmes e Séries")
st.markdown("*Encontre seus próximos filmes favoritos!*")

df = load_data()

if df is not None:
    cosine_sim, indices, df_model = prepare_recommendation_model(df)
    
    if cosine_sim is not None:
        # Sidebar
        with st.sidebar:
            st.header("⚙️ Configurações")
            num_recommendations = st.slider("Recomendações", 3, 10, 5)
            
            st.markdown("---")
            st.metric("Total", len(df))
            col1, col2 = st.columns(2)
            col1.metric("Filmes", len(df[df['Type'] == 'Movie']))
            col2.metric("Séries", len(df[df['Type'] == 'TV Show']))
        
        # Abas
        tab1, tab2, tab3 = st.tabs(["🔍 Buscar", "📈 Análise", "ℹ️ Sobre"])
        
        with tab1:
            st.subheader("Procure por um Filme ou Série")
            search_term = st.text_input("Digite o nome:", placeholder="Ex: Stranger Things...")
            
            if search_term:
                exact = df[df['Title'].str.lower() == search_term.lower()]
                partial = df[df['Title'].str.lower().str.contains(search_term.lower(), na=False)]
                
                if len(exact) > 0:
                    selected = exact.iloc[0]['Title']
                elif len(partial) > 0:
                    selected = st.selectbox("Resultados:", partial['Title'].unique())
                else:
                    st.warning(f"Nenhum resultado para '{search_term}'")
                    selected = None
                
                if selected:
                    st.markdown("---")
                    movie = df[df['Title'] == selected].iloc[0]
                    
                    col1, col2, col3 = st.columns(3)
                    col1.write(f"**Tipo:** {movie['Type']}")
                    col2.write(f"**País:** {movie['Country']}")
                    col3.write(f"**Diretor:** {movie['Director']}")
                    
                    st.write(f"**Elenco:** {movie['Cast']}")
                    st.markdown("---")
                    
                    st.subheader(f"🎯 {num_recommendations} Recomendações")
                    
                    recs, scores = get_recommendations(selected, cosine_sim, indices, df_model, num_recommendations)
                    
                    if recs is not None:
                        for i, (title, score) in enumerate(zip(recs, scores), 1):
                            col1, col2 = st.columns([3, 1])
                            col1.write(f"{i}. **{title}**")
                            col2.metric("Score", f"{score:.1%}")
                        
                        st.markdown("---")
                        fig, ax = plt.subplots(figsize=(10, 5))
                        colors = plt.cm.viridis(np.linspace(0, 1, len(recs)))
                        ax.barh(range(len(recs)), scores, color=colors)
                        ax.set_yticks(range(len(recs)))
                        ax.set_yticklabels(recs.values)
                        ax.set_xlabel('Similaridade')
                        ax.set_title(f'Recomendações para: {selected}')
                        ax.invert_yaxis()
                        st.pyplot(fig, use_container_width=True)
        
        with tab2:
            st.subheader("📊 Análise de Dados")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("#### Gêneros Populares")
                try:
                    genres = df['Type'].value_counts().head(10)
                    fig, ax = plt.subplots()
                    ax.bar(range(len(genres)), genres.values)
                    ax.set_xticks(range(len(genres)))
                    ax.set_xticklabels(genres.index, rotation=45, ha='right')
                    st.pyplot(fig, use_container_width=True)
                except:
                    st.info("Gráfico indisponível")
            
            with col2:
                st.write("#### Distribuição")
                try:
                    counts = df['Type'].value_counts()
                    fig, ax = plt.subplots()
                    ax.pie(counts.values, labels=counts.index, autopct='%1.1f%%')
                    st.pyplot(fig, use_container_width=True)
                except:
                    st.info("Gráfico indisponível")
        
        with tab3:
            st.markdown(f"""
            ### 🎬 Sobre
            
            Sistema de recomendação baseado em **Content-Based Filtering**
            
            #### Como funciona:
            1. Extrai características (diretor, elenco, país, tipo)
            2. Transforma em vetores numéricos (TF-IDF)
            3. Calcula similaridade de cosseno
            4. Retorna os títulos mais similares
            
            #### Dataset:
            - **Títulos**: {len(df)}
            - **Filmes**: {len(df[df['Type'] == 'Movie'])}
            - **Séries**: {len(df[df['Type'] == 'TV Show'])}
            
            #### Tecnologias:
            - Streamlit, Pandas, Scikit-learn, Matplotlib
            
            *Desenvolvido para encontrar seu próximo favorito! 🎥*
            """)

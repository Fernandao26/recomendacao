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

# CSS personalizado para melhor aparência
st.markdown("""
    <style>
    .main {
        padding-top: 0rem;
    }
    </style>
""", unsafe_allow_html=True)

# ============================================================================
# CACHE: Carrega dados apenas uma vez
# ============================================================================
@st.cache_data
def load_data():
    """Carrega e prepara os dados do Netflix"""
    try:
        # Procurar arquivo em várias localizações
        possible_paths = [
            'Netflix Dataset.csv',
            './Netflix Dataset.csv',
        ]
        
        for path in possible_paths:
            if os.path.exists(path):
                df = pd.read_csv(path)
                return df
        
        # Se não encontrou, tentar mesmo assim
        df = pd.read_csv('Netflix Dataset.csv')
        return df
    except Exception as e:
        return None

@st.cache_data
def prepare_recommendation_model(df):
    """Prepara o modelo de recomendação usando TF-IDF e similaridade de cosseno"""
    try:
        # Selecionar features
        features = ['Director', 'Cast', 'Country', 'Type', 'Title']
        df_recomm = df[features].copy()
        
        # Preencher valores ausentes
        for col in features:
            df_recomm[col] = df_recomm[col].fillna('')
        
        # Função para limpar dados
        def clean_data(x):
            if isinstance(x, str):
                return str.lower(x.replace(" ", "")).replace(',', ' ')
            else:
                return ''
        
        # Aplicar limpeza
        for feature in ['Director', 'Cast', 'Type']:
            df_recomm[feature] = df_recomm[feature].apply(clean_data)
        
        # Criar tags
        def create_tags(x):
            return x['Director'] + ' ' + x['Cast'] + ' ' + x['Country'] + ' ' + x['Type']
        
        df_recomm['Tags'] = df_recomm.apply(create_tags, axis=1)
        
        # TF-IDF e similaridade de cosseno
        tfidf = TfidfVectorizer(stop_words='english', max_features=5000)
        tfidf_matrix = tfidf.fit_transform(df_recomm['Tags'])
        cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
        
        # Criar índices
        indices = pd.Series(df_recomm.index, index=df_recomm['Title']).drop_duplicates()
        
        return cosine_sim, indices, df
    except Exception as e:
        return None, None, None

def get_recommendations(title, cosine_sim, indices, df, num_recommendations=5):
    """Retorna os títulos mais similares"""
    try:
        if title not in indices.index:
            return None, None
        
        idx = indices[title]
        sim_scores = list(enumerate(cosine_sim[idx]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        sim_scores = sim_scores[1:num_recommendations+1]
        
        movie_indices = [i[0] for i in sim_scores]
        scores = [i[1] for i in sim_scores]
        
        return df['Title'].iloc[movie_indices], scores
    except Exception as e:
        return None, None

def get_movie_info(df, title):
    """Retorna informações detalhadas sobre um filme/série"""
    try:
        movie = df[df['Title'] == title].iloc[0]
        return movie
    except:
        return None

# ============================================================================
# INTERFACE STREAMLIT
# ============================================================================

st.title("🎬 Recomendador de Filmes e Séries")
st.markdown("*Encontre seus próximos filmes e séries favoritos!*")

# Carregar dados
df = load_data()

if df is None:
    st.error("❌ Erro ao carregar o arquivo 'Netflix Dataset.csv'")
    st.info("Certifique-se de que o arquivo está no mesmo diretório da aplicação.")
else:
    # Preparar modelo
    cosine_sim, indices, df_model = prepare_recommendation_model(df)
    
    if cosine_sim is None:
        st.error("❌ Erro ao preparar o modelo de recomendação")
    else:
        # Sidebar com configurações
        with st.sidebar:
            st.header("⚙️ Configurações")
            st.markdown("---")
            
            num_recommendations = st.slider(
                "Quantas recomendações deseja?",
                min_value=3,
                max_value=10,
                value=5,
                step=1
            )
            
            st.markdown("---")
            st.subheader("📊 Informações do Dataset")
            st.metric("Total de Títulos", len(df))
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Filmes", len(df[df['Type'] == 'Movie']))
            with col2:
                st.metric("Séries", len(df[df['Type'] == 'TV Show']))
        
        # Abas principais
        tab1, tab2, tab3 = st.tabs(["🔍 Buscar Recomendações", "📈 Análise Exploratória", "ℹ️ Sobre"])
        
        # ============================================================================
        # TAB 1: BUSCAR RECOMENDAÇÕES
        # ============================================================================
        with tab1:
            st.subheader("Procure por um Filme ou Série")
            
            # Input do usuário
            col1, col2 = st.columns([3, 1])
            with col1:
                search_term = st.text_input(
                    "Digite o nome do filme/série:",
                    placeholder="Ex: Stranger Things, Inception, Breaking Bad..."
                )
            with col2:
                search_button = st.button("🔍 Buscar", use_container_width=True)
            
            if search_term:
                # Procurar correspondências exatas ou aproximadas
                exact_matches = df[df['Title'].str.lower() == search_term.lower()]
                partial_matches = df[df['Title'].str.lower().str.contains(search_term.lower(), na=False)]
                
                if len(exact_matches) > 0:
                    selected_title = exact_matches.iloc[0]['Title']
                elif len(partial_matches) > 0:
                    st.write("### Resultados encontrados:")
                    titles_list = partial_matches['Title'].unique()
                    selected_title = st.selectbox(
                        "Selecione um título:",
                        titles_list,
                        label_visibility="collapsed"
                    )
                else:
                    st.warning(f"❌ Nenhum resultado encontrado para '{search_term}'")
                    selected_title = None
                
                if selected_title:
                    st.markdown("---")
                    
                    # Informações do filme/série selecionado
                    st.subheader(f"📌 Informações: {selected_title}")
                    
                    movie_info = get_movie_info(df, selected_title)
                    
                    if movie_info is not None:
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.write(f"**Tipo:** {movie_info['Type']}")
                        with col2:
                            st.write(f"**País:** {movie_info['Country']}")
                        with col3:
                            st.write(f"**Diretor:** {movie_info['Director']}")
                        
                        st.write(f"**Elenco:** {movie_info['Cast']}")
                        
                        st.markdown("---")
                        
                        # Obter recomendações
                        st.subheader(f"🎯 {num_recommendations} Recomendações Similares")
                        
                        recommendations, scores = get_recommendations(
                            selected_title,
                            cosine_sim,
                            indices,
                            df_model,
                            num_recommendations
                        )
                        
                        if recommendations is not None:
                            # Criar DataFrame com recomendações
                            rec_df = pd.DataFrame({
                                'Posição': range(1, len(recommendations) + 1),
                                'Título': recommendations.values,
                                'Similaridade': [f"{score:.1%}" for score in scores]
                            })
                            
                            # Exibir recomendações
                            for idx, (i, row) in enumerate(rec_df.iterrows(), 1):
                                with st.container():
                                    col1, col2, col3 = st.columns([1, 3, 1])
                                    with col1:
                                        st.markdown(f"### {row['Posição']}")
                                    with col2:
                                        st.write(f"**{row['Título']}**")
                                    with col3:
                                        st.metric("Score", row['Similaridade'])
                            
                            # Gráfico de similaridade
                            st.markdown("---")
                            st.subheader("📊 Visualização de Similaridade")
                            
                            fig, ax = plt.subplots(figsize=(10, 5))
                            colors = plt.cm.viridis(np.linspace(0, 1, len(recommendations)))
                            ax.barh(range(len(recommendations)), scores, color=colors)
                            ax.set_yticks(range(len(recommendations)))
                            ax.set_yticklabels(recommendations.values)
                            ax.set_xlabel('Similaridade de Cosseno')
                            ax.set_title(f'Pontuação de Similaridade para: {selected_title}')
                            ax.invert_yaxis()
                            
                            for i, v in enumerate(scores):
                                ax.text(v + 0.01, i, f'{v:.2f}', va='center')
                            
                            st.pyplot(fig, use_container_width=True)
        
        # ============================================================================
        # TAB 2: ANÁLISE EXPLORATÓRIA
        # ============================================================================
        with tab2:
            st.subheader("📊 Análise Exploratória dos Dados")
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Top 10 Gêneros/Tipos
                st.write("#### Top 10 Gêneros Mais Populares")
                
                try:
                    genres_list = df['Type'].str.split(', ', expand=True).stack()
                    genres_count = genres_list.value_counts().nlargest(10)
                    
                    fig, ax = plt.subplots(figsize=(10, 6))
                    colors_genres = plt.cm.viridis(np.linspace(0, 1, len(genres_count)))
                    ax.bar(range(len(genres_count)), genres_count.values, color=colors_genres)
                    ax.set_xticks(range(len(genres_count)))
                    ax.set_xticklabels(genres_count.index, rotation=45, ha='right')
                    ax.set_ylabel('Contagem de Títulos')
                    ax.set_title('Top 10 Gêneros/Tipos')
                    
                    st.pyplot(fig, use_container_width=True)
                except Exception as e:
                    st.error(f"Erro ao gerar gráfico de gêneros")
            
            with col2:
                # Top 5 Diretores
                st.write("#### Top 5 Diretores com Mais Títulos")
                
                try:
                    directors_count = (df['Director']
                                      .fillna('Unknown')
                                      .str.split(', ', expand=True)
                                      .stack()
                                      .value_counts()
                                      .nlargest(5))
                    
                    if 'Unknown' in directors_count.index:
                        directors_count = directors_count.drop('Unknown')
                    
                    fig, ax = plt.subplots(figsize=(10, 6))
                    colors_dir = plt.cm.Spectral(np.linspace(0, 1, len(directors_count)))
                    ax.bar(range(len(directors_count)), directors_count.values, color=colors_dir)
                    ax.set_xticks(range(len(directors_count)))
                    ax.set_xticklabels(directors_count.index, rotation=45, ha='right')
                    ax.set_ylabel('Contagem de Títulos')
                    ax.set_title('Top 5 Diretores')
                    
                    st.pyplot(fig, use_container_width=True)
                except Exception as e:
                    st.error(f"Erro ao gerar gráfico de diretores")
            
            # Distribuição por tipo
            st.write("#### Distribuição: Filmes vs Séries")
            try:
                type_count = df['Type'].value_counts()
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    fig, ax = plt.subplots(figsize=(6, 6))
                    colors_pie = ['#FF6B6B', '#4ECDC4']
                    ax.pie(type_count.values, labels=type_count.index, autopct='%1.1f%%',
                           colors=colors_pie, startangle=90)
                    ax.set_title('Proporção de Títulos')
                    st.pyplot(fig, use_container_width=True)
                
                with col2:
                    st.metric("Filmes", type_count.get('Movie', 0))
                    st.metric("Séries", type_count.get('TV Show', 0))
            except Exception as e:
                st.error(f"Erro ao gerar gráfico de distribuição")
        
        # ============================================================================
        # TAB 3: SOBRE
        # ============================================================================
        with tab3:
            st.markdown(f"""
            ### 🎬 Sobre o Recomendador de Filmes/Séries
            
            Este aplicativo utiliza **Content-Based Filtering** para recomendar filmes e séries.
            
            #### 🔬 Como Funciona:
            
            1. **Extração de Features**: O sistema extrai características de cada título (diretor, elenco, país, tipo)
            2. **TF-IDF Vectorization**: Transforma as características em vetores numéricos
            3. **Similaridade de Cosseno**: Calcula a similaridade entre todos os títulos
            4. **Recomendação**: Retorna os títulos mais similares ao selecionado
            
            #### 📊 Dataset:
            - **Número de Títulos**: {len(df)}
            - **Filmes**: {len(df[df['Type'] == 'Movie'])}
            - **Séries**: {len(df[df['Type'] == 'TV Show'])}
            
            #### 🛠️ Tecnologias Utilizadas:
            - **Streamlit**: Framework para interface web
            - **Pandas**: Manipulação de dados
            - **Scikit-learn**: Machine Learning (TF-IDF, Cosine Similarity)
            - **Matplotlib & Seaborn**: Visualização de dados
            
            #### 💡 Dicas de Uso:
            - Use nomes completos de títulos para melhores resultados
            - Explore diferentes títulos para descobrir padrões nas recomendações
            - Ajuste o número de recomendações na barra lateral
            
            ---
            *Desenvolvido para encontrar seu próximo favorito! 🎥*
            """)


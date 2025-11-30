# 🎬 Recomendador de Filmes e Séries

Uma aplicação interativa para descobrir seus próximos filmes e séries favoritos usando Machine Learning!

## ✨ Funcionalidades

- 🔍 **Buscar Recomendações**: Digite o nome de um filme/série e receba 5-10 recomendações similares
- 📊 **Análise Exploratória**: Veja gráficos dos gêneros e diretores mais populares
- 🎯 **Filtros Personalizados**: Ajuste a quantidade de recomendações que deseja
- 📱 **Acesso Multiplataforma**: Use em PC, celular, tablet, etc.

## 🚀 Como Executar

### Opção 1: Acesso Local (Recomendado)

#### No seu PC/Servidor:
```bash
# Instalar dependências
pip install -r requirements.txt

# Executar a aplicação
python run.py
```

#### Em outro dispositivo na MESMA rede Wi-Fi:
1. Abra um navegador
2. Cole um destes links:
   - **PC local**: `http://localhost:8502`
   - **Outro dispositivo**: `http://10.0.10.103:8502`

> **Nota**: O IP pode variar. Se não funcionar, rode `ipconfig` (Windows) ou `ifconfig` (Linux/Mac) para encontrar seu IP local.

### Opção 2: Streamlit Cloud (Hospedagem Gratuita)

1. Faça login em https://share.streamlit.io
2. Conecte seu repositório GitHub
3. Sua app estará disponível na internet gratuitamente!

## 📊 Dataset

- **Fonte**: Netflix Dataset
- **Total de Títulos**: ~7800 filmes e séries
- **Features**: Diretor, Elenco, País, Tipo

## 🔬 Como Funciona

O sistema usa **Content-Based Filtering**:

1. **Extração de Features**: Director, Cast, Country, Type
2. **TF-IDF Vectorization**: Converte texto em números
3. **Cosine Similarity**: Calcula similaridade entre títulos
4. **Ranking**: Retorna os mais similares ao que você escolheu

## 🛠️ Tecnologias

- **Streamlit**: Interface web interativa
- **Pandas**: Manipulação de dados
- **Scikit-learn**: Machine Learning (TF-IDF, Cosine Similarity)
- **Matplotlib & Seaborn**: Visualizações

## 📋 Requisitos

- Python 3.8+
- Todas as dependências em `requirements.txt`

## 🤝 Contribuindo

Sinta-se livre para fazer fork e enviar pull requests!

## 📄 Licença

MIT License - veja LICENSE para detalhes

---

**Desenvolvido para encontrar seu próximo favorito! 🎥**

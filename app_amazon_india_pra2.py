
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import re

# Configuració inicial
st.set_page_config(page_title="Anàlisi Amazon Índia", layout="wide")

st.title("🛒 Anàlisi Visual de Productes d'Amazon Índia")
st.markdown("""
Aquest **dashboard interactiu** mostra la relació entre preus, descomptes, valoracions i categories dels productes venuts a Amazon Índia.

**Objectius de l'anàlisi**:
- Identificar oportunitats de negoci basades en preus i valoracions.
- Descobrir les categories més rendibles.
- Analitzar l'efectivitat dels descomptes.
- Oferir _insights_ útils per a la gestió del catàleg.
""")

# ===================== CÀRREGA DE DADES =====================
@st.cache_data
def load_data():
    df = pd.read_csv("amazon.csv")

    def clean_price(price_str):
        if isinstance(price_str, str):
            return float(re.sub(r'[₹,]', '', price_str))
        return price_str

    def to_euro(rupee):
        if isinstance(rupee, str):
            return clean_price(rupee) / 95
        return rupee

    df['actual_price'] = df['actual_price'].apply(to_euro)
    df['discounted_price']  = df['discounted_price'].apply(to_euro)
    df['discount_percentage'] = df['discount_percentage'].str.rstrip('%').astype(float)

    category_split = df['category'].str.split('|', expand=True)
    category_split.columns = [f'category_level_{i+1}' for i in range(category_split.shape[1])]
    df = pd.concat([df, category_split], axis=1)

    counts = df['category_level_1'].value_counts()
    categories_to_keep = counts[counts >= 20].index.tolist()
    df = df[df['category_level_1'].isin(categories_to_keep)]

    return df

df = load_data()

# ===================== PREPARACIÓ =====================
def prepare_data(df):
    df = df.copy()
    df['discounted_price'] = pd.to_numeric(df['discounted_price'], errors='coerce')
    df['actual_price'] = pd.to_numeric(df['actual_price'], errors='coerce')
    df['discount_percentage'] = pd.to_numeric(df['discount_percentage'], errors='coerce')
    df['rating'] = pd.to_numeric(df['rating'], errors='coerce')
    df['rating_count'] = pd.to_numeric(df['rating_count'], errors='coerce')

    df['rating_count'] = df['rating_count'].fillna(1)
    df['discount_percentage'] = df['discount_percentage'].fillna(0)
    df = df.dropna(subset=['discounted_price', 'rating'])
    df['rating_count'] = df['rating_count'].clip(lower=1)

    df['price_range'] = pd.cut(df['discounted_price'],
                              bins=[0, 20, 50, 100, 200, np.inf],
                              labels=['Molt Baix (<20)', 'Baix (20-50)', 'Mitjà (50-100)',
                                     'Alt (100-200)', 'Molt Alt (>200)'])

    df['rating_category'] = pd.cut(df['rating'],
                                  bins=[0, 3, 4, 4.5, 5],
                                  labels=['Baix (<3)', 'Mitjà (3-4)', 'Bo (4-4.5)', 'Excel·lent (>4.5)'])

    df['discount_range'] = pd.cut(df['discount_percentage'],
                                 bins=[0, 10, 25, 50, 100],
                                 labels=['Baix (<10%)', 'Mitjà (10-25%)', 'Alt (25-50%)', 'Molt Alt (>50%)'])

    df['main_category'] = df['category_level_1'].fillna(df['category'])

    return df

df_clean = prepare_data(df)

# ===================== ESTADÍSTIQUES =====================
st.subheader("📈 Estadístiques generals")
col1, col2, col3 = st.columns(3)
col1.metric("Preu mitjà", f"{df_clean['discounted_price'].mean():.2f} €")
col2.metric("Valoració mitjana", f"{df_clean['rating'].mean():.2f} / 5")
col3.metric("Descompte mitjà", f"{df_clean['discount_percentage'].mean():.1f} %")


def category_analysis_charts(df):
    """
    Retorna quatre gràfics Plotly independents per Streamlit
    """
    cat_stats = df.groupby('main_category').agg({
        'rating': 'mean',
        'rating_count': 'sum',
        'discounted_price': 'mean',
        'discount_percentage': 'mean',
        'product_id': 'count'
    }).round(2)

    cat_stats.columns = ['Valoració Mitjana', 'Total Valoracions', 'Preu Mitjà', 'Descompte Mitjà %', 'Num Productes']
    cat_stats = cat_stats.sort_values('Total Valoracions', ascending=False).head(15).reset_index()

    fig1 = px.bar(cat_stats, x='main_category', y='Valoració Mitjana',
                  title='📊 Valoració Mitjana per Categoria', labels={'main_category': 'Categoria'})

    fig2 = px.bar(cat_stats, x='main_category', y='Preu Mitjà',
                  title='💶 Preu Mitjà per Categoria', labels={'main_category': 'Categoria'})

    fig3 = px.bar(cat_stats, x='main_category', y='Descompte Mitjà %',
                  title='🏷️ Descompte Mitjà per Categoria', labels={'main_category': 'Categoria'})

    fig4 = px.bar(cat_stats, x='main_category', y='Num Productes',
                  title='📦 Número de Productes per Categoria', labels={'main_category': 'Categoria'})

    return fig1, fig2, fig3, fig4


# ===================== VISUALITZACIONS =====================
st.markdown("### 🔍 Exploració Visual")

with st.expander("1. 💬 Relació entre Preu i Valoració"):
    df_sample = df_clean.sample(min(5000, len(df_clean)))
    df_sample['size_normalized'] = ((df_sample['rating_count'] - df_sample['rating_count'].min()) /
                                    (df_sample['rating_count'].max() - df_sample['rating_count'].min()) * 17 + 3)
    fig1 = px.scatter(df_sample, x='discounted_price', y='rating',
                      color='main_category', size='size_normalized',
                      hover_data=['product_name', 'discount_percentage', 'actual_price', 'rating_count'],
                      title="Relació entre Preu i Valoració per Categoria")
    st.plotly_chart(fig1, use_container_width=True)


with st.expander("📚 Anàlisi Comparativa per Categories"):
    f1, f2, f3, f4 = category_analysis_charts(df_clean)
    st.plotly_chart(f1, use_container_width=True)
    st.plotly_chart(f2, use_container_width=True)
    st.plotly_chart(f3, use_container_width=True)
    st.plotly_chart(f4, use_container_width=True)

    cat_stats = df_clean.groupby('main_category').agg({
        'rating': 'mean',
        'rating_count': 'sum',
        'discounted_price': 'mean',
        'discount_percentage': 'mean',
        'product_id': 'count'
    }).round(2).sort_values('rating_count', ascending=False).head(12).reset_index()
    fig2 = px.bar(cat_stats, x='main_category', y='rating', color='main_category',
                  title="Valoració Mitjana per Categoria", labels={'rating': 'Valoració'})
    st.plotly_chart(fig2, use_container_width=True)

with st.expander("3. 🎯 Efectivitat dels Descomptes"):
    fig3 = px.box(df_clean, x='discount_range', y='rating',
                  title="Valoracions segons Rang de Descompte",
                  labels={'discount_range': 'Rang de Descompte', 'rating': 'Valoració'})
    st.plotly_chart(fig3, use_container_width=True)

with st.expander("4. 💰 Distribució de Preus"):
    fig4 = px.histogram(df_clean, x='discounted_price', nbins=50, title="Distribució del Preu amb Descompte")
    fig4.add_vline(x=df_clean['discounted_price'].median(), line_dash="dash", line_color="red",
                   annotation_text=f"Mediana: {df_clean['discounted_price'].median():.2f}€")
    st.plotly_chart(fig4, use_container_width=True)

with st.expander("5. ⭐ Valoracions per Rang de Preu"):
    fig5 = px.violin(df_clean, x='price_range', y='rating', box=True, points='all',
                     title="Distribució de Valoracions segons el Rang de Preu")
    st.plotly_chart(fig5, use_container_width=True)

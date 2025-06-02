
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import re

st.set_page_config(page_title="Anàlisi Amazon Índia", layout="wide")

# ===================== TÍTOL =====================
st.markdown("""
# 🛒 Anàlisi de Productes d'Amazon Índia

Benvingut al dashboard interactiu de visualització de dades sobre productes venuts a Amazon Índia. Aquesta eina explora la relació entre **preus**, **valoracions**, **descomptes** i **categories** de productes per ajudar a descobrir patrons útils per a la presa de decisions.

---

""")

# ===================== CÀRREGA I NETEJA =====================
@st.cache_data
def load_and_prepare_data():
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
    df['discounted_price'] = df['discounted_price'].apply(to_euro)
    df['discount_percentage'] = df['discount_percentage'].str.rstrip('%').astype(float)

    category_split = df['category'].str.split('|', expand=True)
    category_split.columns = [f'category_level_{i+1}' for i in range(category_split.shape[1])]
    df = pd.concat([df, category_split], axis=1)

    counts = df['category_level_1'].value_counts()
    categories_to_keep = counts[counts >= 20].index.tolist()
    df = df[df['category_level_1'].isin(categories_to_keep)]

    df['rating'] = pd.to_numeric(df['rating'], errors='coerce')
    df['rating_count'] = pd.to_numeric(df['rating_count'], errors='coerce')
    df['rating_count'] = df['rating_count'].fillna(1)
    df['discount_percentage'] = df['discount_percentage'].fillna(0)
    df = df.dropna(subset=['discounted_price', 'rating'])
    df['rating_count'] = df['rating_count'].clip(lower=1)

    df['price_range'] = pd.cut(df['discounted_price'],
        bins=[0, 20, 50, 100, 200, np.inf],
        labels=['Molt Baix (<20€)', 'Baix (20-50€)', 'Mitjà (50-100€)', 'Alt (100-200€)', 'Molt Alt (>200€)'])

    df['discount_range'] = pd.cut(df['discount_percentage'],
        bins=[0, 10, 25, 50, 100],
        labels=['Baix (<10%)', 'Mitjà (10-25%)', 'Alt (25-50%)', 'Molt Alt (>50%)'])

    df['main_category'] = df['category_level_1'].fillna(df['category'])

    return df

df = load_and_prepare_data()

# ===================== KPIs =====================
st.markdown("## 📊 Resum General")

col1, col2, col3 = st.columns(3)
col1.metric("Preu amb descompte mitjà", f"{df['discounted_price'].mean():.2f} €")
col2.metric("Valoració mitjana", f"{df['rating'].mean():.2f} / 5")
col3.metric("Descompte mitjà", f"{df['discount_percentage'].mean():.1f} %")

# ===================== DISTRIBUCIONS =====================
st.markdown("## 📈 Distribucions Generals")

col4, col5 = st.columns(2)
with col4:
    fig_price = px.histogram(df, x='discounted_price', nbins=50,
        title="Distribució del Preu amb Descompte",
        labels={'discounted_price': 'Preu amb Descompte (€)'})
    fig_price.add_vline(x=df['discounted_price'].median(), line_dash="dash", line_color="red",
        annotation_text=f"Mediana: {df['discounted_price'].median():.2f}€")
    st.plotly_chart(fig_price, use_container_width=True)

with col5:
    fig_rating = px.histogram(df, x='rating', nbins=20,
        title="Distribució de les Valoracions",
        labels={'rating': 'Valoració (1-5)'})
    st.plotly_chart(fig_rating, use_container_width=True)

# ===================== PREU vs VALORACIÓ =====================
st.markdown("## 🧮 Relació entre Preu i Valoració")

df_sample = df.sample(min(4000, len(df)))
df_sample['size'] = ((df_sample['rating_count'] - df_sample['rating_count'].min()) /
                    (df_sample['rating_count'].max() - df_sample['rating_count'].min()) * 15 + 5)

fig_scatter = px.scatter(df_sample, x='discounted_price', y='rating',
    color='main_category', size='size',
    hover_data=['product_name', 'actual_price', 'discount_percentage', 'rating_count'],
    labels={'discounted_price': 'Preu amb Descompte (€)', 'rating': 'Valoració'},
    title="Preu amb Descompte vs Valoració per Categoria")

fig_scatter.update_layout(height=600)
st.plotly_chart(fig_scatter, use_container_width=True)

# ===================== CATEGORIES =====================
st.markdown("## 🧾 Anàlisi per Categories")

@st.cache_data
def category_analysis_charts(df):
    cat_stats = df.groupby('main_category').agg({
        'rating': 'mean',
        'rating_count': 'sum',
        'discounted_price': 'mean',
        'discount_percentage': 'mean',
        'product_id': 'count'
    }).round(2).sort_values('rating_count', ascending=False).head(10).reset_index()

    fig1 = px.bar(cat_stats, x='main_category', y='rating',
        title='Valoració Mitjana', labels={'main_category': 'Categoria'})
    fig2 = px.bar(cat_stats, x='main_category', y='discounted_price',
        title='Preu Mitjà amb Descompte', labels={'main_category': 'Categoria'})
    fig3 = px.bar(cat_stats, x='main_category', y='discount_percentage',
        title='Descompte Mitjà (%)', labels={'main_category': 'Categoria'})
    fig4 = px.bar(cat_stats, x='main_category', y='product_id',
        title='Nombre de Productes', labels={'main_category': 'Categoria'})

    return fig1, fig2, fig3, fig4

fig1, fig2, fig3, fig4 = category_analysis_charts(df)

c1, c2 = st.columns(2)
with c1: st.plotly_chart(fig1, use_container_width=True)
with c2: st.plotly_chart(fig2, use_container_width=True)
with c1: st.plotly_chart(fig3, use_container_width=True)
with c2: st.plotly_chart(fig4, use_container_width=True)

# ===================== DESCOMPTES i RANGS =====================
st.markdown("## 🎯 Anàlisi de Descomptes i Rang de Preus")

col6, col7 = st.columns(2)
with col6:
    fig_discount = px.box(df, x='discount_range', y='rating',
        title="Valoracions segons Rang de Descompte",
        labels={'discount_range': 'Descompte', 'rating': 'Valoració'})
    st.plotly_chart(fig_discount, use_container_width=True)

with col7:
    fig_violin = px.violin(df, x='price_range', y='rating', box=True, points='all',
        title="Valoracions segons Rang de Preu",
        labels={'price_range': 'Rang de Preu', 'rating': 'Valoració'})
    st.plotly_chart(fig_violin, use_container_width=True)

# ===================== PEU =====================
st.markdown("---")
st.markdown("📌 Desenvolupat com a pràctica per a l'assignatura de Visualització de Dades - Ciència de Dades Aplicada (UOC).")

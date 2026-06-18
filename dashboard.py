import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import seaborn as sns
from collections import defaultdict
from matplotlib.patches import Patch
import os

st.set_page_config(page_title="หนี้สินครัวเรือนไทย", layout="wide", page_icon="📊")

# ── Thai font setup ───────────────────────────────────────────────────────────
import matplotlib, glob

_FONT_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'THSarabunNew.ttf')
if os.path.exists(_FONT_FILE):
    _FONT_PROP = fm.FontProperties(fname=_FONT_FILE)
    _THAI_FONT = _FONT_PROP.get_name()
    # Register so seaborn/matplotlib can find it by name
    fm.fontManager.addfont(_FONT_FILE)
else:
    _FONT_PROP = fm.FontProperties(family='DejaVu Sans')
    _THAI_FONT = 'DejaVu Sans'

def set_thai_font(ax):
    """Apply Thai font to every text element in the figure — bypasses Streamlit Arial override."""
    fig = ax.get_figure()
    for artist in ([fig.suptitle('')] +
                   [ax.title, ax.xaxis.label, ax.yaxis.label] +
                   ax.get_xticklabels() + ax.get_yticklabels() +
                   [t for t in ax.texts]):
        try:
            artist.set_fontproperties(_FONT_PROP)
        except Exception:
            pass
    # Legend
    leg = ax.get_legend()
    if leg:
        for t in leg.get_texts():
            t.set_fontproperties(_FONT_PROP)
        leg.get_title().set_fontproperties(_FONT_PROP)
    # Also set rcParams as backup for things like ax.set_title called after
    plt.rcParams['font.family'] = _THAI_FONT
    plt.rcParams['axes.unicode_minus'] = False
 
# ── Data loading ──────────────────────────────────────────────────────────────
DEBT_FILE   = '20230521160827_42424.xlsx'
INCOME_FILE = '20230521160113_19514.xlsx'

@st.cache_data
def load_data():
    # Check files exist
    for f in [DEBT_FILE, INCOME_FILE]:
        if not os.path.exists(f):
            return None, None, None, None

    # ── Debt ──
    df_debt = pd.read_excel(DEBT_FILE, skiprows=2)
    df_debt.columns = [
        'Region', 'Province', 'Purpose',
        'Debt_2547','Debt_2549','Debt_2550','Debt_2552','Debt_2554',
        'Debt_2556','Debt_2558','Debt_2560','Debt_2562','Debt_2564','Debt_2566'
    ]
    df_debt[['Province','Region']] = df_debt[['Province','Region']].ffill()
    df_debt = df_debt.drop(df_debt.index[-2:])
    df_debt = df_debt.replace('-', 0)
    vicinity = ['กรุงเทพมหานคร','นนทบุรี','ปทุมธานี','สมุทรปราการ','นครปฐม','สมุทรสาคร']
    df_debt.loc[df_debt['Province'].isin(vicinity), 'Region'] = 'กรุงเทพมหานครและปริมณฑล'

    # ── Income ──
    df_income = pd.read_excel(INCOME_FILE, skiprows=2)
    df_income = df_income.drop(columns=[df_income.columns[0]])
    df_income.columns = [
        'Region','Province',
        'Income_2547','Income_2549','Income_2550','Income_2552','Income_2554',
        'Income_2556','Income_2558','Income_2560','Income_2562','Income_2564','Income_2566'
    ]
    df_income[['Region']] = df_income[['Region']].ffill()
    df_income = df_income.drop(df_income.index[-1:])
    df_income = df_income.fillna(0)
    df_income_sorted = df_income.sort_values(['Region','Province']).reset_index(drop=True)
    df_income_clean = df_income_sorted[df_income_sorted['Region'] != df_income_sorted['Province']].copy()
    df_income_clean.loc[df_income_clean['Province'].isin(vicinity), 'Region'] = 'กรุงเทพมหานครและปริมณฑล'

    # ── Merge for df_long ──
    df_debt_summary = df_debt[df_debt['Purpose'] == 'หนี้สินทั้งสิ้น'].copy()
    df_merged = pd.merge(df_income_clean, df_debt_summary, on=['Province','Region'])
    df_long = pd.wide_to_long(
        df_merged,
        stubnames=['Income','Debt'],
        i=['Region','Province'], j='Year', sep='_'
    ).reset_index()
    for col in ['Income','Debt']:
        df_long[col] = pd.to_numeric(df_long[col], errors='coerce').fillna(0)

    # ── Purpose long (drop education + หนี้สินทั้งสิ้น) ──
    df_debt_purpose = df_debt[df_debt['Purpose'] != 'หนี้สินทั้งสิ้น'].copy()
    df_debt_purpose = df_debt_purpose[df_debt_purpose['Purpose'] != 'เพื่อใช้ในการศึกษา'].copy()
    debt_year_cols = [c for c in df_debt_purpose.columns if c.startswith('Debt_')]
    df_purpose_long = pd.melt(
        df_debt_purpose,
        id_vars=['Region','Province','Purpose'],
        value_vars=debt_year_cols,
        var_name='Year', value_name='Debt'
    )
    df_purpose_long['Year'] = df_purpose_long['Year'].str.replace('Debt_','')
    df_purpose_long['Debt'] = pd.to_numeric(df_purpose_long['Debt'], errors='coerce').fillna(0)

    return df_long, df_purpose_long, df_debt, df_income_clean

df_long, df_purpose_long, df_debt_raw, df_income_raw = load_data()

# ── Header ────────────────────────────────────────────────────────────────────
st.title("📊 วิเคราะห์หนี้สินครัวเรือนไทย")
st.caption("ข้อมูลจากการสำรวจภาวะเศรษฐกิจและสังคมครัวเรือน (NSO) ปี 2547–2566")

if df_long is None:
    st.error(f"ไม่พบไฟล์ข้อมูล กรุณาวาง `{DEBT_FILE}` และ `{INCOME_FILE}` ไว้ในโฟลเดอร์เดียวกับ dashboard.py แล้วรันใหม่")
    st.stop()

YEARS_ALL   = sorted(df_long['Year'].unique())
YEARS_NO47  = [y for y in YEARS_ALL if y != 2547]
PURPOSE_YEARS = sorted(df_purpose_long['Year'].unique())

REGION_PALETTE = {
    'กรุงเทพมหานครและปริมณฑล': '#e74c3c',
    'กลาง':                    '#e67e22',
    'เหนือ':                   '#3498db',
    'ตะวันออกเฉียงเหนือ':       '#9b59b6',
    'ใต้':                     '#1abc9c',
    'ตะวันออก':                '#f39c12',
    'ตะวันตก':                 '#2ecc71',
}

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📈 กราฟ 1: แนวโน้มหนี้รายภาค",
    "📊 กราฟ 2: Growth Index",
    "🏷️ กราฟ 3: วัตถุประสงค์การกู้",
    "⚠️ กราฟ 4: Risk Mapping",
    "🔴 กราฟ 5: Top 10 DTI",
])

# ── GRAPH 1 ───────────────────────────────────────────────────────────────────
with tab1:
    st.subheader("1. แนวโน้มการสะสมหนี้สินครัวเรือนย้อนหลังแยกรายภูมิภาค (2547–2566)")
    df_trend = df_long.groupby(['Year','Region'])['Debt'].mean().reset_index()

    fig, ax = plt.subplots(figsize=(12, 5))
    sns.lineplot(x=df_trend['Year'].astype(str), y='Debt', hue='Region',
                 data=df_trend, marker='o', linewidth=2, ax=ax)
    ax.set_xlabel('ปี พ.ศ.', fontsize=12)
    ax.set_ylabel('หนี้สินสะสมเฉลี่ยต่อครัวเรือน (บาท)', fontsize=12)
    ax.legend(title='ภูมิภาค', bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=9)
    set_thai_font(ax)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

# ── GRAPH 2 ───────────────────────────────────────────────────────────────────
with tab2:
    st.subheader("2. ดัชนีการเติบโตสะสม: รายได้ VS หนี้สิน (ฐานปี 2549 = 100)")
    st.caption("ตัดปี 2547 ออกเนื่องจาก dataset หมวดการศึกษายังไม่มีการสำรวจในปีนั้น — ฐานปี 2549 คือปีแรกที่ข้อมูลครบถ้วน")

    df_nat = df_long[df_long['Year'] != 2547].copy()
    df_nat = df_nat.groupby('Year')[['Income','Debt']].mean().reset_index()
    base_inc  = df_nat.loc[df_nat['Year'] == 2549, 'Income'].values[0]
    base_debt = df_nat.loc[df_nat['Year'] == 2549, 'Debt'].values[0]
    df_nat['Income_Growth'] = (df_nat['Income'] / base_inc)  * 100
    df_nat['Debt_Growth']   = (df_nat['Debt']   / base_debt) * 100

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(df_nat['Year'].astype(str), df_nat['Income_Growth'],
            marker='o', linewidth=2.5, label='ดัชนีรายได้', color='green')
    ax.plot(df_nat['Year'].astype(str), df_nat['Debt_Growth'],
            marker='s', linewidth=2.5, label='ดัชนีหนี้สิน', color='#e74c3c')
    ax.axhline(y=100, color='gray', linestyle='--', alpha=0.5, label='ฐานปี 2549')
    ax.set_xlabel('ปี พ.ศ.', fontsize=12)
    ax.set_ylabel('ดัชนีการเติบโตสะสม (ฐานปี 2549 = 100)', fontsize=12)
    ax.legend(title='ตัวชี้วัด')
    set_thai_font(ax)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    col1, col2 = st.columns(2)
    latest = df_nat.iloc[-1]
    col1.metric("ดัชนีรายได้ ปี 2566", f"{latest['Income_Growth']:.1f}")
    col2.metric("ดัชนีหนี้สิน ปี 2566", f"{latest['Debt_Growth']:.1f}",
                delta=f"+{latest['Debt_Growth'] - latest['Income_Growth']:.1f} เหนือรายได้",
                delta_color="inverse")

# ── GRAPH 3 ───────────────────────────────────────────────────────────────────
with tab3:
    st.subheader("3. โครงสร้างวัตถุประสงค์การกู้ยืมแยกรายภูมิภาค")
    st.caption("หมายเหตุ: ไม่รวมหนี้เพื่อการศึกษา เนื่องจากข้อมูลไม่มีการสำรวจในปี 2547 และขาดหายในบางปี")

    sel_year3 = st.select_slider(
        "เลือกปี", options=PURPOSE_YEARS, value=PURPOSE_YEARS[-1], key='yr3'
    )

    df_y = df_purpose_long[df_purpose_long['Year'] == str(sel_year3)]
    df_avg = df_y.groupby(['Region','Purpose'])['Debt'].mean().reset_index()

    fig, ax = plt.subplots(figsize=(14, 6))
    sns.barplot(data=df_avg, x='Region', y='Debt', hue='Purpose',
                palette='Set2', ax=ax)
    ax.set_title(f'โครงสร้างวัตถุประสงค์การกู้ยืมเงินครัวเรือนแยกรายภูมิภาค ปี {sel_year3}',
                 fontsize=13, fontweight='bold', pad=12)
    ax.set_xlabel('ภูมิภาค', fontsize=11)
    ax.set_ylabel('ยอดหนี้สินเฉลี่ย (บาท)', fontsize=11)
    ax.tick_params(axis='x', rotation=15)
    ax.legend(title='วัตถุประสงค์', bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=8)
    set_thai_font(ax)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

# ── GRAPH 4 ───────────────────────────────────────────────────────────────────
with tab4:
    st.subheader("4. Risk Mapping — การกระจายตัว DTI รายภาค")

    sel_year4 = st.select_slider(
        "เลือกปี", options=YEARS_ALL, value=YEARS_ALL[-1], key='yr4'
    )

    df_dti = df_long.copy()
    df_dti['Annual_Income'] = df_dti['Income'] * 12
    df_dti['DTI'] = np.where(df_dti['Annual_Income'] > 0,
                             df_dti['Debt'] / df_dti['Annual_Income'], np.nan)
    df_dti_y = df_dti[df_dti['Year'] == sel_year4].dropna(subset=['DTI']).copy()

    region_order = (df_dti_y.groupby('Region')['DTI']
                    .median().sort_values(ascending=False).index.tolist())
    region_y_map = {r: i for i, r in enumerate(region_order)}
    df_danger = df_dti_y[df_dti_y['DTI'] > 1.0].copy()

    fig, ax = plt.subplots(figsize=(15, 7))
    sns.set_theme(style='whitegrid')
    sns.boxplot(data=df_dti_y, x='DTI', y='Region', order=region_order,
                palette='coolwarm', width=0.5, linewidth=1.2, fliersize=4, ax=ax)
    sns.stripplot(data=df_dti_y, x='DTI', y='Region', order=region_order,
                  color='#2c3e50', alpha=0.35, size=4, jitter=True, ax=ax)
    ax.axvline(x=1.0, color='#e74c3c', linestyle='--', linewidth=2,
               label='เส้นเตือนภัย DTI = 1.0', zorder=5)

    for _, row in df_danger.iterrows():
        ax.scatter(row['DTI'], region_y_map[row['Region']],
                   color='#e74c3c', s=65, zorder=6, edgecolors='white', linewidths=0.8)

    groups = defaultdict(list)
    for _, row in df_danger.sort_values('DTI').iterrows():
        groups[row['Region']].append((row['DTI'], row['Province']))
    STEP, X_PAD = 0.28, 0.04
    for region, items in groups.items():
        base_y = region_y_map[region]
        n = len(items)
        offsets = [(i - (n - 1) / 2.0) * STEP for i in range(n)]
        for (dti, province), dy in zip(items, offsets):
            label_y = base_y + dy
            ax.annotate("", xy=(dti, base_y), xytext=(dti + X_PAD - 0.01, label_y),
                        arrowprops=dict(arrowstyle='-', color='#e74c3c', lw=0.7, alpha=0.6))
            ax.text(dti + X_PAD, label_y, province,
                    fontsize=8.5, color='#c0392b', fontweight='bold', va='center', ha='left',
                    bbox=dict(boxstyle='round,pad=0.15', fc='white', ec='none', alpha=0.75))

    title_line1 = f'Risk Mapping: การกระจายตัว DTI รายภาค ปี {sel_year4}'
    title_line2 = '(จุดแดง = จังหวัดที่ DTI ทะลุเส้นเตือนภัย 1.0)'
    ax.set_title(title_line1 + '\n' + title_line2, fontsize=13, fontweight='bold', pad=12)
    ax.set_xlabel('Debt-to-Income Ratio (DTI)', fontsize=11)
    ax.set_ylabel('ภูมิภาค', fontsize=11)
    ax.legend(loc='lower right', fontsize=9)
    set_thai_font(ax)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    if len(df_danger) > 0:
        st.warning(f"⚠️ พบ {len(df_danger)} จังหวัด ที่มี DTI เกิน 1.0 ในปี {sel_year4}")
        st.dataframe(
            df_danger[['Region','Province','DTI']].sort_values('DTI', ascending=False)
            .reset_index(drop=True).style.format({'DTI': '{:.3f}'}),
            use_container_width=True
        )
    else:
        st.success(f"✅ ไม่มีจังหวัดที่ DTI เกิน 1.0 ในปี {sel_year4}")

# ── GRAPH 5 ───────────────────────────────────────────────────────────────────
with tab5:
    st.subheader("5. 10 อันดับจังหวัดที่มี DTI วิกฤตที่สุด")

    sel_year5 = st.select_slider(
        "เลือกปี", options=YEARS_ALL, value=YEARS_ALL[-1], key='yr5'
    )

    df_dti2 = df_long.copy()
    df_dti2['Annual_Income'] = df_dti2['Income'] * 12
    df_dti2['DTI'] = np.where(df_dti2['Annual_Income'] > 0,
                              df_dti2['Debt'] / df_dti2['Annual_Income'], np.nan)
    df_dti_y2 = df_dti2[df_dti2['Year'] == sel_year5].dropna(subset=['DTI'])

    df_top10 = (df_dti_y2[['Province','Region','DTI']]
                .sort_values('DTI', ascending=False)
                .head(10).reset_index(drop=True))
    df_top10['Rank'] = df_top10.index + 1
    bar_colors = df_top10['Region'].map(REGION_PALETTE).fillna('#95a5a6')

    fig, ax = plt.subplots(figsize=(13, 6))
    bars = ax.barh(df_top10['Province'][::-1], df_top10['DTI'][::-1],
                   color=bar_colors[::-1].values, edgecolor='white', linewidth=0.6, height=0.65)
    ax.axvline(x=1.0, color='#e74c3c', linestyle='--', linewidth=1.8, alpha=0.8,
               label='เส้นเตือนภัย DTI = 1.0')
    for bar, dti in zip(bars, df_top10['DTI'][::-1]):
        ax.text(bar.get_width() + 0.02, bar.get_y() + bar.get_height() / 2,
                f'{dti:.2f}', va='center', ha='left', fontsize=10,
                fontweight='bold', color='#2c3e50')
    legend_elements = [Patch(facecolor=color, label=region)
                       for region, color in REGION_PALETTE.items()
                       if region in df_top10['Region'].values]
    title_line1 = f'10 อันดับจังหวัดที่มี DTI วิกฤตที่สุด ปี {sel_year5}'
    title_line2 = '(ยิ่งสูงยิ่งเปราะบาง — หนี้ครัวเรือนเกินรายได้ต่อปี)'
    ax.set_title(title_line1 + '\n' + title_line2, fontsize=13, fontweight='bold', pad=12)
    ax.set_xlabel('Debt-to-Income Ratio (DTI)', fontsize=11)
    ax.set_ylabel('จังหวัด', fontsize=11)
    ax.set_xlim(0, df_top10['DTI'].max() * 1.18)
    set_thai_font(ax)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    st.dataframe(
        df_top10[['Rank','Province','Region','DTI']].style.format({'DTI': '{:.3f}'}),
        use_container_width=True
    )
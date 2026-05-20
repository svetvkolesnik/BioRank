# -*- coding: utf-8 -*-
import streamlit as st
import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
import re
import statsmodels.api as sm

st.set_page_config(page_title="GroupRanker Pro", layout="wide", initial_sidebar_state="expanded")
st.title("🎯 GroupRanker Pro")
st.markdown("**Универсальная модель многокритериального ранжирования экспериментальных групп**")
st.caption("Поддерживает: дозозависимые эксперименты, факторный дизайн, комбинированные добавки")

MS_HINTS = [
    "pb","cd","hg","as","al","be","sn","свинец","кадмий","ртуть","мышьяк","олово","бериллий","алюминий",
    "глюкоз","glucos","холестерин","cholesterol","триглицерид","triglycerid",
    "alt","ast","алт","аст","алп","alp","билирубин","bilirubin","креатинин","creatinin",
    "лейкоцит","leukocyt","wbc","мочевин","urea","мочев","cortisol","кортизол","toxic","токсич",
]
ME_HINTS = [
    "weight","масса","прирост","gain","живая","белок","protein","albumin","альбумин",
    "hgb","гемоглобин","hemoglobin","rbc","эритроцит","гематокрит","hematocrit","тромбоцит","platelet",
    "sod","cat","каталаз","глутатион","glutathion","fe_","zn_","cu_","se_","железо_мышц","цинк_мышц",
]

def guess_role(col_name):
    col_lower = col_name.lower()
    for hint in MS_HINTS:
        if hint in col_lower:
            return "M_S (безопасность/риск)"
    for hint in ME_HINTS:
        if hint in col_lower:
            return "M_E (эффективность)"
    return "M_E (эффективность)"

def safe_numeric(col, lod=0.00005):
    s = col.astype(str).str.strip()
    def extract_censored(val):
        if "<" in str(val):
            return lod * np.random.uniform(0.8, 1.2)
        return val
    s = s.apply(extract_censored)
    s = s.str.replace(",", ".", regex=False).str.replace(" ", "", regex=False)
    return pd.to_numeric(s, errors="coerce")

def winsorize_feature(series, limits=(0.05, 0.95)):
    data = series.dropna()
    if len(data) > 5:
        lower, upper = data.quantile(limits)
        return series.clip(lower=lower, upper=upper)
    return series

def visualize_outliers(df, features):
    st.subheader("🔍 Анализ выбросов")
    n_cols = min(3, len(features))
    for i in range(0, len(features), n_cols):
        cols = st.columns(n_cols)
        for j, feature in enumerate(features[i:i+n_cols]):
            if j < len(cols):
                with cols[j]:
                    fig, ax = plt.subplots(figsize=(5, 3))
                    data = df[feature].dropna()
                    if len(data) > 3:
                        sns.boxplot(y=data, ax=ax)
                        ax.set_title(f"{feature}\nn={len(data)}")
                    plt.tight_layout()
                    st.pyplot(fig)
                    plt.close(fig)

def compute_group_sensitivity(df, group_col, features, alpha=0.05, mode="group"):
    results = []
    for feature in features:
        x = df[feature].dropna()
        grp_vals = df.loc[x.index, group_col]
        if len(x) < 4 or grp_vals.nunique() < 2:
            continue
        groups_list = [v.values for _, v in x.groupby(grp_vals)]
        try:
            p_kw = stats.kruskal(*groups_list)[1] if len(groups_list) > 1 else 1.0
        except Exception:
            p_kw = 1.0
        rho, p_spear, p_quad = np.nan, np.nan, np.nan
        if mode == "dose":
            try:
                rho, p_spear = stats.spearmanr(grp_vals.astype(float), x)
            except Exception:
                pass
            if grp_vals.nunique() > 3:
                try:
                    """d = grp_vals.astype(float)
                    X = sm.add_constant(pd.DataFrame({"d": d, "d2": d**2}))
                    model = sm.OLS(x, X).fit()
                    p_quad = model.pvalues.get("d2", 1.0)"""
                    # 1. Приводим к типу float и центрируем дозы для устранения коллинеарности
                    d = grp_vals.astype(float)
                    d_centered = d - d.mean()

                    # 2. Формируем матрицу признаков на основе центрированных доз
                    X = pd.DataFrame({"d_lin": d_centered, "d2_quad": d_centered**2})
                    X = sm.add_constant(X)

                    # 3. Обучаем модель OLS и сразу пересчитываем стандартные ошибки по методу HC3
                    model = sm.OLS(x, X).fit(cov_type="HC3")

                    # 4. Безопасно извлекаем p-value для квадратичного члена
                    p_quad = model.pvalues.get("d2_quad", 1.0)
                except Exception:
                    pass
        if mode == "dose":
            is_sensitive = (p_kw < alpha) or (not np.isnan(p_spear) and p_spear < alpha and abs(rho) > 0.5) or (not np.isnan(p_quad) and p_quad < alpha)
        else:
            is_sensitive = p_kw < alpha
        results.append({
            "Маркер": feature,
            "p (KW)": round(p_kw, 4),
            "ρ Спирмен": round(rho, 3) if not np.isnan(rho) else "—",
            "p (Spear)": round(p_spear, 4) if not np.isnan(p_spear) else "—",
            "p (квадр.)": round(p_quad, 4) if not np.isnan(p_quad) else "—",
            "Значимый": is_sensitive,
        })
    return pd.DataFrame(results)

# ── МАССА ТЕЛА ──────────────────────────────────────────────────────────────
def compute_weight_kw_and_shifts(df_weight, weight_group_col, weight_col, control_group, alpha=0.05):
    """
    KW-тест и нормированные сдвиги медианы массы по группам.
    Возвращает: p_kw, shifts (dict | None), summary (DataFrame).
    
    Обе подвыборки (масса и биохимия) являются независимыми случайными
    выборками из одной группы, поэтому групповые медианы массы — несмещённые
    оценки параметра генеральной совокупности группы.
    """
    df_w = df_weight[[weight_group_col, weight_col]].copy()
    df_w[weight_col] = safe_numeric(df_w[weight_col])
    df_w = df_w.dropna()

    groups = df_w[weight_group_col].unique()
    groups_data = [df_w[df_w[weight_group_col] == g][weight_col].values for g in groups]
    try:
        p_kw = stats.kruskal(*groups_data)[1] if len(groups_data) > 1 else 1.0
    except Exception:
        p_kw = 1.0

    summary_rows = []
    for g in groups:
        vals = df_w[df_w[weight_group_col] == g][weight_col]
        summary_rows.append({"Группа": g, "n": len(vals),
                              "Медиана": round(vals.median(), 2),
                              "IQR": round(vals.quantile(0.75) - vals.quantile(0.25), 2)})
    summary = pd.DataFrame(summary_rows)

    if p_kw >= alpha:
        return p_kw, None, summary

    all_vals = df_w[weight_col]
    iqr_all = all_vals.quantile(0.75) - all_vals.quantile(0.25)
    if iqr_all == 0:
        return p_kw, None, summary

    ctrl_med = df_w[df_w[weight_group_col] == control_group][weight_col].median()
    shifts = {}
    for g in df_w[weight_group_col].unique():
        g_med = df_w[df_w[weight_group_col] == g][weight_col].median()
        shifts[g] = (g_med - ctrl_med) / iqr_all
    return p_kw, shifts, summary

# ── КРИТЕРИИ ────────────────────────────────────────────────────────────────
def compute_efficiency(df, group_col, control_group, me_markers, weight_shifts=None):
    """
    E_g = медиана нормированных сдвигов маркеров M_E.
    weight_shifts: dict {group: delta_weight} | None
        Если передан, сдвиг по массе добавляется к сдвигам биохимических маркеров
        (Вариант А: масса как дополнительный маркер M_E).
        Вычисляется по независимой весовой подвыборке.
    """
    ctrl_data = df[df[group_col] == control_group]
    E_vals = {}
    for group in df[group_col].unique():
        grp_data = df[df[group_col] == group]
        shifts = []
        for m in me_markers:
            g_vals = grp_data[m].dropna()
            c_vals = ctrl_data[m].dropna()
            if len(g_vals) == 0 or len(c_vals) == 0:
                continue
            iqr_m = df[m].quantile(0.75) - df[m].quantile(0.25)
            if iqr_m == 0:
                continue
            shifts.append((g_vals.median() - c_vals.median()) / iqr_m)
        if weight_shifts is not None and group in weight_shifts:
            shifts.append(weight_shifts[group])
        E_vals[group] = float(np.nanmedian(shifts)) if shifts else 0.0
    return E_vals

def compute_safety(df, group_col, control_group, ms_markers):
    ctrl_data = df[df[group_col] == control_group]
    S_vals = {}
    for group in df[group_col].unique():
        grp_data = df[df[group_col] == group]
        risks = []
        for m in ms_markers:
            g_vals = grp_data[m].dropna()
            c_vals = ctrl_data[m].dropna()
            if len(g_vals) == 0 or len(c_vals) == 0:
                continue
            iqr_m = df[m].quantile(0.75) - df[m].quantile(0.25)
            if iqr_m == 0:
                continue
            risks.append(max(0.0, (g_vals.median() - c_vals.median()) / iqr_m))
        S_vals[group] = float(np.nanmedian(risks)) if risks else 0.0
    return S_vals

def compute_balance_criterion(df, group_col, control_group, balance_ratios, features):
    B_vals = {}
    eps = 1e-8
    for group in df[group_col].unique():
        deviations = []
        for ratio_str in balance_ratios:
            ratio_str = ratio_str.strip()
            if "/" not in ratio_str:
                continue
            num_base, den_base = [x.strip() for x in ratio_str.split("/", 1)]
            num_cols = [c for c in features if num_base.lower() in c.lower()]
            for nc in num_cols:
                loc_match = re.search(r"(_[a-z]+)$", nc.lower())
                loc_suffix = loc_match.group(1) if loc_match else ""
                dc_candidates = [c for c in features if den_base.lower() in c.lower() and c.lower().endswith(loc_suffix)]
                if not dc_candidates:
                    continue
                dc = dc_candidates[0]
                grp_data = df[df[group_col] == group]
                ctrl_data = df[df[group_col] == control_group]
                if grp_data.empty or ctrl_data.empty:
                    continue
                d_ratio = (grp_data[nc] / (grp_data[dc] + eps)).median()
                c_ratio = (ctrl_data[nc] / (ctrl_data[dc] + eps)).median()
                iqr_r = (df[nc] / (df[dc] + eps)).quantile(0.75) - (df[nc] / (df[dc] + eps)).quantile(0.25)
                if iqr_r > 0:
                    deviations.append(abs((d_ratio - c_ratio) / iqr_r))
        B_vals[group] = -np.nanmedian(deviations) if deviations else 0.0
    return B_vals

def normalize_dict(d):
    vals = [v for v in d.values() if not pd.isna(v)]
    if not vals:
        return {k: 0.5 for k in d}
    vmin, vmax = min(vals), max(vals)
    if vmax == vmin:
        return {k: 0.5 for k in d}
    return {k: (v - vmin) / (vmax - vmin) for k, v in d.items()}

def safe_rerun():
    try:
        st.rerun()
    except AttributeError:
        st.experimental_rerun()

# ── БУТСТРЭП ────────────────────────────────────────────────────────────────
def bootstrap_indices(df, group_col, control_group, me_markers, ms_markers,
                      balance_ratios, features, w_E, w_S, w_B,
                      n_boot=1000, random_state=42,
                      df_weight=None, weight_group_col=None,
                      weight_col=None, control_group_w=None):
    """
    Бутстрэп по объектам внутри каждой группы.

    Если переданы df_weight и связанные параметры, весовые данные
    ресэмплируются НЕЗАВИСИМО от биохимических на каждой итерации.
    Это корректно: обе подвыборки случайны и независимы, но из одной группы.
    """
    rng = np.random.default_rng(random_state)
    groups = sorted(df[group_col].dropna().unique())
    group_data = {g: df[df[group_col] == g] for g in groups}

    use_weight = (df_weight is not None and weight_group_col is not None
                  and weight_col is not None and control_group_w is not None)
    iqr_weight_global = 0
    weight_group_data = {}
    if use_weight:
        all_w = df_weight[weight_col].dropna()
        iqr_weight_global = all_w.quantile(0.75) - all_w.quantile(0.25)
        for g in df_weight[weight_group_col].unique():
            weight_group_data[g] = df_weight[df_weight[weight_group_col] == g][weight_col].dropna().values

    w_total = w_E + w_S + w_B
    w_E_n, w_S_n, w_B_n = (w_E/w_total, w_S/w_total, w_B/w_total) if w_total > 0 else (1.0, 0.0, 0.0)
    I_boot = {g: [] for g in groups}

    for _ in range(int(n_boot)):
        df_b = pd.concat([
            g_df.iloc[rng.integers(0, len(g_df), size=len(g_df))]
            for g, g_df in group_data.items() if len(g_df) > 0
        ], axis=0)

        weight_shifts_b = None
        if use_weight and iqr_weight_global > 0:
            ctrl_w = weight_group_data.get(control_group_w, np.array([]))
            if len(ctrl_w) > 0:
                ctrl_w_med = np.median(ctrl_w[rng.integers(0, len(ctrl_w), size=len(ctrl_w))])
                weight_shifts_b = {}
                for g in groups:
                    g_w = weight_group_data.get(g, np.array([]))
                    if len(g_w) > 0:
                        g_w_med = np.median(g_w[rng.integers(0, len(g_w), size=len(g_w))])
                        weight_shifts_b[g] = (g_w_med - ctrl_w_med) / iqr_weight_global

        E_n_b = normalize_dict(compute_efficiency(df_b, group_col, control_group, me_markers, weight_shifts_b))
        S_n_b = normalize_dict(compute_safety(df_b, group_col, control_group, ms_markers) if ms_markers else {g: 0.0 for g in groups})
        B_n_b = normalize_dict(compute_balance_criterion(df_b, group_col, control_group, balance_ratios, features) if balance_ratios else {g: 0.0 for g in groups})

        for g in groups:
            I_boot[g].append(w_E_n * E_n_b.get(g, 0.5) - w_S_n * S_n_b.get(g, 0.5) + w_B_n * B_n_b.get(g, 0.5))

    return I_boot

# ══════════════════════════════════════════════════════════════════════════════
# БОКОВАЯ ПАНЕЛЬ
# ══════════════════════════════════════════════════════════════════════════════
st.sidebar.header("📁 Загрузка данных")
uploaded_file = st.sidebar.file_uploader("CSV/Excel — основные данные (биохимия, элементы)", type=["csv","xlsx"], key="main_file")

st.sidebar.markdown("---")
st.sidebar.header("📎 Дополнительная подвыборка (опционально)")
st.sidebar.caption(
    "Файл с признаками из параллельной случайной выборки особей той же группы. "
    "Строки = особи, столбцы = группа + признак(и). "
    "Особи могут не совпадать с основным файлом."
)
weight_file = st.sidebar.file_uploader(
    "CSV/Excel — дополнительная подвыборка",
    type=["csv", "xlsx"],
    key="weight_file",
)

# ══════════════════════════════════════════════════════════════════════════════
# ЗАГРУЗКА ОСНОВНОГО ФАЙЛА
# ══════════════════════════════════════════════════════════════════════════════
if uploaded_file is not None:
    try:
        df_raw = pd.read_csv(uploaded_file, sep=None, decimal=",", engine="python") if uploaded_file.name.endswith(".csv") else pd.read_excel(uploaded_file)
        st.session_state.df_raw = df_raw
        st.success(f"✅ Основной файл: {len(df_raw)} строк × {len(df_raw.columns)} столбцов")
        st.dataframe(df_raw.head(), use_container_width=True)

        st.sidebar.header("🧪 Тип дизайна")
        exp_mode = st.sidebar.radio("Структура опытных групп",
            ["🔢 Числовые дозы (доза-ответ)", "🏷️ Категориальные группы (факторный / комбинации добавок)"])
        is_dose_mode = exp_mode.startswith("🔢")

        st.sidebar.header("⚙️ Столбцы основного файла")
        group_col_name = st.sidebar.selectbox("Столбец с группами" + (" (дозы)" if is_dose_mode else " (метки)"), df_raw.columns)

        if is_dose_mode:
            group_raw = safe_numeric(df_raw[group_col_name])
            if group_raw.isna().all():
                st.error("❌ Столбец доз не содержит числовых значений!"); st.stop()
            df_raw["_group"] = group_raw
            unique_groups = sorted(df_raw["_group"].dropna().unique())
        else:
            df_raw["_group"] = df_raw[group_col_name].astype(str).str.strip()
            # Умная сортировка: числовые строки сортируем как числа
            def smart_sort_main(vals):
                try:
                    return sorted(vals, key=lambda x: float(x))
                except (ValueError, TypeError):
                    return sorted(vals, key=str)
            unique_groups = smart_sort_main(df_raw["_group"].dropna().unique())

        st.sidebar.markdown(f"**Групп ({len(unique_groups)}):** " + ", ".join(str(g) for g in unique_groups))
        control_group = st.sidebar.selectbox("🎯 Контрольная группа", options=unique_groups, index=0)

        exclude_cols = {"_group", group_col_name}
        all_numeric = [c for c in df_raw.select_dtypes(include=[np.number]).columns if c not in exclude_cols]
        features = st.sidebar.multiselect("📊 Признаки для анализа", all_numeric, default=all_numeric[:min(20, len(all_numeric))])

        if st.sidebar.button("🚀 Запустить анализ", type="primary") and features:
            df_processed = df_raw.copy()
            for col in features:
                df_processed[col] = safe_numeric(df_raw[col])
            st.session_state.df_processed = df_processed
            st.session_state.features = features
            st.session_state.group_col = "_group"
            st.session_state.control_group = control_group
            st.session_state.unique_groups = unique_groups
            st.session_state.is_dose_mode = is_dose_mode
            for key in ["df_clean","group_stats","results","I_boot","weight_shifts_final","p_kw_weight","weight_summary"]:
                st.session_state.pop(key, None)
            st.success("✅ Данные подготовлены!")
            safe_rerun()
    except Exception as e:
        import traceback
        st.error(f"❌ Ошибка загрузки: {e}"); st.code(traceback.format_exc()); st.stop()

# ══════════════════════════════════════════════════════════════════════════════
# ЗАГРУЗКА ФАЙЛА МАССЫ
# ══════════════════════════════════════════════════════════════════════════════
if weight_file is not None:
    try:
        df_wraw = (
            pd.read_csv(weight_file, sep=None, decimal=",", engine="python")
            if weight_file.name.endswith(".csv")
            else pd.read_excel(weight_file)
        )

        # ── Предпросмотр ────────────────────────────────────────────────────
        st.sidebar.success(
            f"✅ Доп. файл: {len(df_wraw)} строк × {len(df_wraw.columns)} столбцов"
        )
        st.sidebar.caption("Первые 3 строки:")
        st.sidebar.dataframe(df_wraw.head(3), use_container_width=True)

        # ── Выбор столбца групп ─────────────────────────────────────────────
        # Используем двойной механизм сохранения выбора:
        # 1) key="wg_col_sel"  — стандартный виджетный ключ Streamlit
        # 2) session_state["_ext_wg_choice"] — явное сохранение, которое
        #    переживает st.rerun() во всех версиях Streamlit.
        # index= восстанавливает выбор из явного сохранения,
        # даже если виджетный ключ сбросился.
        wg_options = list(df_wraw.columns)

        saved_wg = st.session_state.get("_ext_wg_choice", None)
        wg_idx = (
            wg_options.index(saved_wg)
            if saved_wg in wg_options
            else 0
        )
        wg_col_name = st.sidebar.selectbox(
            "Столбец с метками групп",
            wg_options,
            index=wg_idx,
            key="wg_col_sel",
        )
        # Явно сохраняем выбор — это надёжнее виджетного ключа
        st.session_state["_ext_wg_choice"] = wg_col_name

        # ── Выбор столбца признака ──────────────────────────────────────────
        wv_options = [c for c in wg_options if c != wg_col_name]
        saved_wv = st.session_state.get("_ext_wv_choice", None)
        wv_idx = (
            wv_options.index(saved_wv)
            if saved_wv in wv_options
            else 0
        )
        wv_col_name = st.sidebar.selectbox(
            "Столбец с измеряемым признаком",
            wv_options,
            index=wv_idx,
            key="wv_col_sel",
        )
        st.session_state["_ext_wv_choice"] = wv_col_name

        # ── Обработка ───────────────────────────────────────────────────────
        df_wraw = df_wraw.copy()
        df_wraw["_wgroup"] = df_wraw[wg_col_name].astype(str).str.strip()
        raw_unique = df_wraw["_wgroup"].dropna().unique()

        def smart_sort_groups(vals):
            try:
                return sorted(vals, key=lambda x: float(x))
            except (ValueError, TypeError):
                return sorted(vals, key=str)

        unique_wgroups = smart_sort_groups(raw_unique)
        n_uniq  = len(unique_wgroups)
        n_rows  = len(df_wraw)

        # Диагностика выбранного столбца — показываем всегда
        preview_vals = ", ".join(str(g) for g in unique_wgroups[:10])
        suffix = f" … ещё {n_uniq - 10}" if n_uniq > 10 else ""
        st.sidebar.caption(
            f"Столбец «{wg_col_name}»: **{n_uniq}** уник. значений: "
            f"{preview_vals}{suffix}"
        )

        _ext_ok = True
        if n_uniq >= n_rows * 0.8:
            st.sidebar.warning(
                f"⚠️ {n_uniq} уникальных значений при {n_rows} строках. "
                f"Проверьте: выбран ли столбец с метками групп?"
            )
            _ext_ok = st.sidebar.checkbox(
                "Продолжить с этим столбцом",
                value=False,
                key="ext_force_ok",
            )
        elif n_uniq > 20:
            st.sidebar.warning(f"⚠️ {n_uniq} групп — больше ожидаемого.")

        # ── Контрольная группа ──────────────────────────────────────────────
        if _ext_ok:
            saved_ctrl = st.session_state.get("_ext_ctrl_choice", None)
            ctrl_idx = (
                unique_wgroups.index(saved_ctrl)
                if saved_ctrl in unique_wgroups
                else 0
            )
            control_group_w = st.sidebar.selectbox(
                "Контрольная группа (доп. файл)",
                options=unique_wgroups,
                index=ctrl_idx,
                key="ctrl_w_sel",
            )
            st.session_state["_ext_ctrl_choice"] = control_group_w

            # Сохраняем данные ТОЛЬКО после успешного прохождения всех проверок
            st.session_state.df_weight_raw      = df_wraw
            st.session_state["wg_col"]          = "_wgroup"
            st.session_state["wv_col"]          = wv_col_name
            st.session_state["control_group_w"] = control_group_w
        else:
            # Очищаем невалидное состояние
            for _k in ["df_weight_raw", "wg_col", "wv_col",
                       "control_group_w", "weight_shifts_final"]:
                st.session_state.pop(_k, None)

    except Exception as e:
        import traceback
        st.sidebar.error(f"❌ Ошибка файла: {e}")
        st.sidebar.code(traceback.format_exc())

# ══════════════════════════════════════════════════════════════════════════════
# ГЛАВНЫЙ АНАЛИЗ
# ══════════════════════════════════════════════════════════════════════════════
if "df_processed" in st.session_state:
    df = st.session_state.df_processed
    features = st.session_state.features
    group_col = st.session_state.group_col
    control_group = st.session_state.control_group
    unique_groups = st.session_state.unique_groups
    is_dose_mode = st.session_state.is_dose_mode
    mode_str = "dose" if is_dose_mode else "group"
    st.markdown("---")

    # ШАГ 1
    st.header("🔍 Шаг 1 — Анализ и обработка выбросов")
    col_a, col_b = st.columns([3, 1])
    with col_a:
        if st.button("📈 Показать боксплоты"):
            visualize_outliers(df, features)
    with col_b:
        outlier_method = st.radio("Метод", ["none (не трогать)", "winsorize (5%–95%)"])
    if st.button("⚙️ Применить и перейти к анализу"):
        df_clean = df.copy()
        if "winsorize" in outlier_method:
            for feat in features:
                df_clean[feat] = winsorize_feature(df_clean[feat])
        st.session_state.df_clean = df_clean
        st.success("✅ Готово!"); safe_rerun()

    # ШАГ 2
    if "df_clean" in st.session_state:
        df_clean = st.session_state.df_clean
        st.markdown("---")
        st.header("🔬 Шаг 2 — " + ("Дозозависимость маркеров" if is_dose_mode else "Значимость различий между группами"))
        if is_dose_mode:
            st.caption("Kruskal-Wallis + ранговая корреляция Спирмена + квадратичная регрессия")
        else:
            st.caption("Критерий Краскела–Уоллиса для каждого маркера.")
        alpha_val = st.slider("Уровень значимости α", 0.01, 0.20, 0.05, 0.01)
        if st.button("📊 Рассчитать значимость", key="sens_btn"):
            with st.spinner("Анализ..."):
                st.session_state.group_stats = compute_group_sensitivity(df_clean, group_col, features, alpha=alpha_val, mode=mode_str)
        if "group_stats" in st.session_state:
            gs = st.session_state.group_stats
            sensitive = gs[gs["Значимый"]]
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("**✅ Значимые маркеры**")
                st.dataframe(sensitive.sort_values("p (KW)"), use_container_width=True)
            with col2:
                st.markdown("**📋 Все маркеры**")
                st.dataframe(gs, use_container_width=True)
            st.metric("Значимых / всего", f"{len(sensitive)} / {len(features)}")

    # ШАГ 3
    if "group_stats" in st.session_state:
        st.markdown("---")
        st.header("🎯 Шаг 3 — Многокритериальное ранжирование групп")
        df_clean = st.session_state.df_clean
        gs = st.session_state.group_stats
        sensitive_markers = gs[gs["Значимый"]]["Маркер"].tolist()
        if not sensitive_markers:
            st.warning("⚠️ Значимых маркеров нет — используем все признаки")
            sensitive_markers = features

        # Корреляционный фильтр
        with st.expander("🔗 Фильтр мультиколлинеарности", expanded=True):
            corr_threshold = st.slider("Порог корреляции Спирмена", 0.70, 1.00, 0.90, 0.05)
            corr_cols = [m for m in sensitive_markers if m in df_clean.columns]
            if len(corr_cols) > 1:
                corr_matrix = df_clean[corr_cols].corr(method="spearman")
                to_drop = set()
                for i in range(len(corr_cols)):
                    for j in range(i+1, len(corr_cols)):
                        if abs(corr_matrix.iloc[i, j]) > corr_threshold:
                            to_drop.add(corr_cols[j])
                sensitive_markers = [c for c in corr_cols if c not in to_drop]
                if to_drop:
                    st.info(f"Удалено {len(to_drop)} маркеров (|r|>{corr_threshold}). Осталось **{len(sensitive_markers)}**.")
                    with st.expander("Тепловая карта корреляций"):
                        fig_corr, ax_corr = plt.subplots(figsize=(max(6, len(corr_cols)*0.6), max(5, len(corr_cols)*0.5)))
                        sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="RdBu_r", center=0, ax=ax_corr, linewidths=0.5)
                        plt.tight_layout(); st.pyplot(fig_corr); plt.close(fig_corr)
                else:
                    st.info("Сильно коррелирующих маркеров не обнаружено.")

        # ── БЛОК ДОПОЛНИТЕЛЬНОЙ ПОДВЫБОРКИ ──────────────────────────────────
        st.markdown("---")
        st.markdown("### 📎 Дополнительная подвыборка")
        has_weight = "df_weight_raw" in st.session_state

        if not has_weight:
            st.info(
                "📂 Файл дополнительной подвыборки не загружен. "
                "Загрузите его в боковой панели, чтобы включить признаки из параллельной выборки "
                "в критерий эффективности или безопасности."
            )
            weight_shifts_final = None
        else:
            df_w = st.session_state.df_weight_raw
            wg_col = st.session_state["wg_col"]
            wv_col = st.session_state["wv_col"]
            control_group_w = st.session_state["control_group_w"]

            p_kw_w, shifts_w, summary_w = compute_weight_kw_and_shifts(
                df_w, wg_col, wv_col, control_group_w, alpha=alpha_val
            )

            col_w1, col_w2 = st.columns(2)
            with col_w1:
                st.markdown(f"**Описательная статистика: «{wv_col}»**")
                st.dataframe(summary_w, use_container_width=True)
            with col_w2:
                st.markdown(f"**KW-тест для «{wv_col}»:** p = {p_kw_w:.4f}")
                if p_kw_w < alpha_val:
                    st.success(
                        f"✅ Различия значимы (p = {p_kw_w:.4f} < {alpha_val}). "
                        f"Признак «{wv_col}» будет включён в M_E."
                    )
                else:
                    st.warning(
                        f"⚠️ Различия не значимы (p = {p_kw_w:.4f} ≥ {alpha_val}). "
                        f"Признак «{wv_col}» исключён по критерию значимости."
                    )

            force_include = st.checkbox(
                f"Включить «{wv_col}» принудительно (игнорировать p-значение)",
                value=False,
                help="Используйте, если различия биологически значимы, "
                     "но не достигли порога из-за малой выборки в дополнительной подвыборке.",
            )

            if p_kw_w < alpha_val or force_include:
                weight_shifts_final = shifts_w
                if shifts_w:
                    st.caption(
                        f"Нормированные сдвиги медианы «{wv_col}» по группам: "
                        + ", ".join(
                            f"{g}: {v:+.3f}"
                            for g, v in sorted(shifts_w.items(), key=lambda x: str(x[0]))
                        )
                    )
                else:
                    st.warning(
                        "Сдвиги не вычислены: IQR = 0 или контрольная группа не найдена в файле."
                    )
                    weight_shifts_final = None
            else:
                weight_shifts_final = None

            st.session_state.weight_shifts_final = weight_shifts_final

            if st.button(f"📊 Боксплоты «{wv_col}» по группам"):
                groups_w_uniq = sorted(df_w[wg_col].unique(), key=str)
                fig_w, ax_w = plt.subplots(figsize=(max(8, len(groups_w_uniq) * 1.2), 4))
                data_bx = [df_w[df_w[wg_col] == g][wv_col].dropna().values for g in groups_w_uniq]
                ax_w.boxplot(data_bx, labels=[str(g) for g in groups_w_uniq], showmeans=True)
                ax_w.set_xlabel("Группа")
                ax_w.set_ylabel(wv_col)
                ax_w.set_title(f"«{wv_col}» по группам (дополнительная подвыборка)")
                plt.tight_layout()
                st.pyplot(fig_w)
                plt.close(fig_w)

        if has_weight:
            weight_shifts_final = st.session_state.get("weight_shifts_final", None)
        else:
            weight_shifts_final = None

        # Назначение ролей
        st.markdown("---")
        st.markdown("### 📋 Роли маркеров")
        st.caption("**M_E** — рост желателен. **M_S** — рост нежелателен. Один маркер — одна роль.")
        if weight_shifts_final is not None:
            st.info(
                f"📎 Признак «{st.session_state.get('wv_col', '')}» из дополнительной подвыборки "
                f"включён в M_E (обрабатывается по независимой выборке, не отображается в таблице ниже)."
            )

        role_df = pd.DataFrame({"Маркер": sensitive_markers, "Роль": [guess_role(m) for m in sensitive_markers]})
        edited_roles = st.data_editor(role_df,
            column_config={
                "Маркер": st.column_config.TextColumn(disabled=True),
                "Роль": st.column_config.SelectboxColumn(options=["M_E (эффективность)", "M_S (безопасность/риск)", "Игнорировать"]),
            }, use_container_width=True, hide_index=True, key="role_editor")

        me_selected = edited_roles[edited_roles["Роль"]=="M_E (эффективность)"]["Маркер"].tolist()
        ms_selected = edited_roles[edited_roles["Роль"]=="M_S (безопасность/риск)"]["Маркер"].tolist()

        col_m1, col_m2, col_m3 = st.columns(3)
        col_m1.metric("M_E (осн. выборка)", len(me_selected))
        col_m2.metric("M_S", len(ms_selected))
        col_m3.metric(
            "Доп. подвыборка",
            f"«{st.session_state.get('wv_col','')}» ✅" if weight_shifts_final else "не подключена",
        )

        # Критерий баланса
        st.markdown("### ⚖️ Критерий баланса (необязательно)")
        use_balance = st.checkbox("Включить критерий баланса физиологических соотношений")
        balance_ratios = []
        if use_balance:
            balance_input = st.text_area("Соотношения — одно на строку:", placeholder="Ca/P\nNa/K\nZn/Cu", height=110)
            balance_ratios = [r.strip() for r in balance_input.strip().split("\n") if "/" in r]
            if balance_ratios:
                st.info(f"Соотношения: {', '.join(balance_ratios)}")

        # Веса
        st.markdown("### 🎚️ Веса критериев")
        n_w = 3 if (use_balance and balance_ratios) else 2
        w_cols = st.columns(n_w)
        w_E = w_cols[0].slider("w₁ Эффективность", 0.0, 1.0, 0.5, 0.05, key="w_e")
        w_S = w_cols[1].slider("w₂ Безопасность", 0.0, 1.0, 0.3 if n_w==3 else 0.5, 0.05, key="w_s")
        w_B = w_cols[2].slider("w₃ Баланс", 0.0, 1.0, 0.2, 0.05, key="w_b") if n_w==3 else 0.0
        w_total = w_E + w_S + w_B
        if w_total > 0:
            w_E_n, w_S_n, w_B_n = w_E/w_total, w_S/w_total, w_B/w_total
            st.caption(f"Нормализованные веса: E={w_E_n:.2f} | S={w_S_n:.2f} | B={w_B_n:.2f}  →  I_g = w_E·E_norm − w_S·S_norm + w_B·B_norm")
        else:
            w_E_n, w_S_n, w_B_n = 1.0, 0.0, 0.0

        # Бутстрэп параметры
        st.markdown("### 🔁 Бутстрэп устойчивости I_g")
        col_bs1, col_bs2 = st.columns(2)
        use_bootstrap = col_bs1.checkbox("Включить бутстрэп по животным", value=False)
        n_boot = col_bs2.number_input("Число итераций", min_value=100, max_value=5000, value=1000, step=100)
        if use_bootstrap and has_weight and weight_shifts_final is not None:
            st.caption(
                f"ℹ️ Бутстрэп проводит независимый ресэмплинг основной и дополнительной "
                f"подвыборок на каждой итерации (признак «{st.session_state.get('wv_col','')}»)."
            )

        # РАСЧЁТ
        if st.button("🚀 Рассчитать I_g и ранжировать группы", type="primary", key="calc_btn"):
            if not me_selected and weight_shifts_final is None:
                st.error("❌ Добавьте хотя бы один маркер M_E или подключите данные о массе!")
            else:
                with st.spinner("Расчёт..."):
                    E_vals = compute_efficiency(df_clean, group_col, control_group, me_selected, weight_shifts_final)
                    S_vals = compute_safety(df_clean, group_col, control_group, ms_selected) if ms_selected else {g: 0.0 for g in df_clean[group_col].unique()}
                    B_vals = compute_balance_criterion(df_clean, group_col, control_group, balance_ratios, features) if (use_balance and balance_ratios) else {g: 0.0 for g in df_clean[group_col].unique()}

                    all_groups = list(df_clean[group_col].unique())
                    results = pd.DataFrame({
                        "Группа": all_groups,
                        "E_g (raw)": [E_vals.get(g,0) for g in all_groups],
                        "S_g (raw)": [S_vals.get(g,0) for g in all_groups],
                        "B_g (raw)": [B_vals.get(g,0) for g in all_groups],
                    })
                    E_n = normalize_dict(dict(zip(results["Группа"], results["E_g (raw)"])))
                    S_n = normalize_dict(dict(zip(results["Группа"], results["S_g (raw)"])))
                    B_n = normalize_dict(dict(zip(results["Группа"], results["B_g (raw)"])))
                    results["E_norm"] = [E_n[g] for g in all_groups]
                    results["S_norm"] = [S_n[g] for g in all_groups]
                    results["B_norm"] = [B_n[g] for g in all_groups]
                    st.session_state.results = results

                    if use_bootstrap:
                        with st.spinner(f"Бутстрэп {int(n_boot)} итераций..."):
                            bstrap_kwargs = {}
                            if has_weight and weight_shifts_final is not None:
                                bstrap_kwargs = dict(
                                    df_weight=st.session_state.df_weight_raw,
                                    weight_group_col=st.session_state["wg_col"],
                                    weight_col=st.session_state["wv_col"],
                                    control_group_w=st.session_state["control_group_w"],
                                )
                            I_boot = bootstrap_indices(
                                df_clean, group_col, control_group, me_selected, ms_selected,
                                balance_ratios if (use_balance and balance_ratios) else [],
                                features, w_E, w_S, w_B, int(n_boot), 42, **bstrap_kwargs)
                            st.session_state.I_boot = I_boot
                    else:
                        st.session_state.pop("I_boot", None)

        # ОТОБРАЖЕНИЕ РЕЗУЛЬТАТОВ
        if "results" in st.session_state:
            results = st.session_state.results.copy()
            results["I_g"] = w_E_n*results["E_norm"] - w_S_n*results["S_norm"] + w_B_n*results["B_norm"]

            st.markdown("### 🏆 Ранжирование групп")
            display_cols = ["Группа","E_norm","S_norm","B_norm","I_g"]
            st.dataframe(results[display_cols].round(3).sort_values("I_g", ascending=False), use_container_width=True)

            x_labels = [str(g) for g in results["Группа"]]
            x_pos = list(range(len(x_labels)))
            n_plots = 2 + (1 if use_balance and balance_ratios else 0)
            fig, axes = plt.subplots(1, n_plots, figsize=(6*n_plots, 5))
            if n_plots == 1: axes = [axes]
            axes[0].bar(x_pos, results["E_norm"], color="#2ecc71", alpha=0.85)
            _ext_label = st.session_state.get("wv_col", "")
            axes[0].set_title(
                "E_norm — Эффективность" + (f" (+ «{_ext_label}»)" if weight_shifts_final and _ext_label else ""),
                fontweight="bold",
            )
            axes[0].set_xticks(x_pos); axes[0].set_xticklabels(x_labels, rotation=40, ha="right")
            axes[1].bar(x_pos, results["S_norm"], color="#e74c3c", alpha=0.85)
            axes[1].set_title("S_norm — Риск", fontweight="bold")
            axes[1].set_xticks(x_pos); axes[1].set_xticklabels(x_labels, rotation=40, ha="right")
            if n_plots == 3:
                axes[2].bar(x_pos, results["B_norm"], color="#3498db", alpha=0.85)
                axes[2].set_title("B_norm — Баланс", fontweight="bold")
                axes[2].set_xticks(x_pos); axes[2].set_xticklabels(x_labels, rotation=40, ha="right")
            plt.tight_layout(); st.pyplot(fig); plt.close(fig)

            best_val = results["I_g"].max()
            fig2, ax2 = plt.subplots(figsize=(max(8, len(x_labels)*1.2), 5))
            bar_colors = ["#f1c40f" if v==best_val else "#2980b9" for v in results["I_g"]]
            bars = ax2.bar(x_pos, results["I_g"], color=bar_colors, alpha=0.9, edgecolor="white")
            ax2.axhline(y=0, color="gray", linestyle="--", alpha=0.5)
            ax2.set_title("I_g — Интегральный индекс", fontweight="bold", fontsize=13)
            ax2.set_xticks(x_pos); ax2.set_xticklabels(x_labels, rotation=40, ha="right", fontsize=10)
            ax2.set_ylabel("I_g")
            for bar, val in zip(bars, results["I_g"]):
                ax2.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.01, f"{val:.3f}", ha="center", va="bottom", fontsize=9)
            plt.tight_layout(); st.pyplot(fig2); plt.close(fig2)

            # Победитель
            best_idx = results["I_g"].idxmax()
            best_group = results.loc[best_idx, "Группа"]
            best_ig = results.loc[best_idx, "I_g"]
            st.success(f"🎉 **Оптимальная группа (все): {best_group}** (I_g = {best_ig:.4f})")

            results_test = results[results["Группа"].astype(str) != str(control_group)]
            if len(results_test) > 0:
                best_test_idx = results_test["I_g"].idxmax()
                best_test_group = results_test.loc[best_test_idx, "Группа"]
                best_test_ig = results_test.loc[best_test_idx, "I_g"]
                if str(best_group) == str(control_group):
                    st.info(f"📌 **Лучшая среди опытных: {best_test_group}** (I_g = {best_test_ig:.4f})  — контрольная группа исключена из выбора дозы.")

            m_cols = st.columns(4)
            show_idx = best_test_idx if str(best_group)==str(control_group) else best_idx
            m_cols[0].metric("Группа", str(results_test.loc[best_test_idx,"Группа"] if str(best_group)==str(control_group) else best_group))
            m_cols[1].metric("E (эффективность)", f"{results.loc[show_idx,'E_norm']:.3f}")
            m_cols[2].metric("S (риск) ↓", f"{results.loc[show_idx,'S_norm']:.3f}")
            m_cols[3].metric("B (баланс)", f"{results.loc[show_idx,'B_norm']:.3f}")

            # Бутстрэп результаты
            if "I_boot" in st.session_state:
                st.markdown("### 📊 Бутстрэп устойчивости I_g")
                I_boot = st.session_state.I_boot
                groups_bs = sorted(I_boot.keys())
                n_boot_eff = len(next(iter(I_boot.values()))) if I_boot else 0
                control_str = str(control_group)
                test_groups = [g for g in groups_bs if str(g) != control_str]

                best_counts_all = {g: 0 for g in groups_bs}
                best_counts_dose = {g: 0 for g in test_groups}
                if n_boot_eff > 0:
                    I_mat = np.vstack([I_boot[g] for g in groups_bs]).T
                    for row in I_mat:
                        best_counts_all[groups_bs[int(np.argmax(row))]] += 1
                        vals_test = [row[groups_bs.index(g)] for g in test_groups]
                        if vals_test:
                            best_counts_dose[test_groups[int(np.argmax(vals_test))]] += 1

                boot_rows = []
                for g in groups_bs:
                    vals = np.array(I_boot[g])
                    is_ctrl = str(g) == control_str
                    boot_rows.append({
                        "Группа": g,
                        "I_med": round(np.median(vals), 3),
                        "I_2.5%": round(np.percentile(vals, 2.5), 3),
                        "I_97.5%": round(np.percentile(vals, 97.5), 3),
                        "P(лучшая, все)": round(best_counts_all[g]/n_boot_eff, 3) if n_boot_eff else np.nan,
                        "P(лучшая, опытные)": round(best_counts_dose[g]/n_boot_eff, 3) if (not is_ctrl and n_boot_eff) else "—",
                    })
                boot_df = pd.DataFrame(boot_rows).sort_values("Группа", key=lambda s: s.astype(str))
                st.dataframe(boot_df, use_container_width=True)

                if n_boot_eff > 0 and test_groups:
                    probs_dose = {g: best_counts_dose[g]/n_boot_eff for g in test_groups}
                    best_dose = max(probs_dose, key=probs_dose.get)
                    random_level = 1/len(test_groups)
                    st.success(f"📌 **Лучшая опытная группа по бутстрэпу: {best_dose}**  (P={probs_dose[best_dose]:.2f} > случайный уровень {random_level:.2f})")
                    if weight_shifts_final is not None:
                        st.caption(
                            f"Бутстрэп проводился с независимым ресэмплингом основной "
                            f"и дополнительной подвыборок «{st.session_state.get('wv_col','')}» на каждой итерации."
                        )

                fig_bs, ax_bs = plt.subplots(figsize=(max(8, len(groups_bs)*1.2), 5))
                ax_bs.boxplot([I_boot[g] for g in groups_bs], labels=[str(g) for g in groups_bs], showmeans=True)
                ax_bs.axhline(y=0, color="gray", linestyle="--", alpha=0.5)
                ax_bs.set_xlabel("Группа"); ax_bs.set_ylabel("I_g (бутстрэп)")
                ax_bs.set_title("Распределение I_g по бутстрэпу")
                plt.tight_layout(); st.pyplot(fig_bs); plt.close(fig_bs)

            st.download_button("⬇️ Скачать результаты CSV",
                data=results.round(4).sort_values("I_g", ascending=False).to_csv(index=False, sep=";"),
                file_name="groupranker_results.csv", mime="text/csv")

else:
    st.info("📁 Загрузите CSV или Excel в боковой панели для начала работы.")
    st.markdown("""
### 🚀 Быстрый старт

| Шаг | Действие |
|-----|----------|
| 1 | Загрузите **основной файл** (биохимия, элементный анализ) |
| 2 | Опционально загрузите **файл массы тела** (строки = особи, столбцы = группа + масса) |
| 3 | Выберите тип дизайна, контрольную группу, признаки |
| 4 | Последовательно пройдите шаги 1–3 анализа |
| 5 | Назначьте роли маркерам, настройте веса, получите I_g |

**Об учёте массы тела:** файл массы может содержать данных других особей, нежели основной файл — это корректно, если обе подвыборки случайны из одной группы. KW-тест проверяет значимость; при p < α масса автоматически добавляется в критерий эффективности. При бутстрэпе оба набора данных ресэмплируются независимо.
""")

st.markdown("---")
st.markdown("*© 2026 GroupRanker Pro | Универсальная модель многокритериального ранжирования*")

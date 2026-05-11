# -*- coding: utf-8 -*-


import streamlit as st
import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
import re
import statsmodels.api as sm

# ======================
# КОНФИГУРАЦИЯ
# ======================
st.set_page_config(
    page_title="GroupRanker Pro",
    layout="wide",
    initial_sidebar_state="expanded"
)
st.title("🎯 GroupRanker Pro")
st.markdown("**Универсальная модель многокритериального ранжирования экспериментальных групп**")
st.caption("Поддерживает: дозозависимые эксперименты, факторный дизайн, комбинированные добавки")

# ======================
# АВТОПОДСКАЗКИ РОЛИ МАРКЕРОВ
# ======================
MS_HINTS = [
    # Токсичные элементы
    "pb", "cd", "hg", "as", "al", "be", "sn", "свинец", "кадмий", "ртуть",
    "мышьяк", "олово", "бериллий", "алюминий",
    # Маркеры стресса и патологии
    "глюкоз", "glucos", "холестерин", "cholesterol", "триглицерид", "triglycerid",
    "alt", "ast", "алт", "аст", "алп", "alp",
    "билирубин", "bilirubin", "креатинин", "creatinin",
    "лейкоцит", "leukocyt", "wbc",
    "мочевин", "urea", "мочев",
    # Прочие риски
    "cortisol", "кортизол", "toxic", "токсич",
]

ME_HINTS = [
    # Продуктивность
    "weight", "масса", "прирост", "gain", "живая",
    # Белковый статус
    "белок", "protein", "albumin", "альбумин",
    # Кровь
    "hgb", "гемоглобин", "hemoglobin", "rbc", "эритроцит",
    "гематокрит", "hematocrit", "тромбоцит", "platelet",
    # Антиоксиданты и ферменты
    "sod", "cat", "каталаз", "глутатион", "glutathion",
    # Эссенциальные элементы (мышцы)
    "fe_", "zn_", "cu_", "se_", "железо_мышц", "цинк_мышц",
]


def guess_role(col_name: str) -> str:
    col_lower = col_name.lower()
    for hint in MS_HINTS:
        if hint in col_lower:
            return "M_S (безопасность/риск)"
    for hint in ME_HINTS:
        if hint in col_lower:
            return "M_E (эффективность)"
    return "M_E (эффективность)"  # нейтральный дефолт


# ======================
# ПРЕДОБРАБОТКА
# ======================
def safe_numeric(col, lod: float = 0.00005) -> pd.Series:
    s = col.astype(str).str.strip()

    def extract_censored(val):
        if "<" in str(val):
            return lod * np.random.uniform(0.8, 1.2)
        return val

    s = s.apply(extract_censored)
    s = s.str.replace(",", ".", regex=False).str.replace(" ", "", regex=False)
    return pd.to_numeric(s, errors="coerce")


def winsorize_feature(series: pd.Series, limits=(0.05, 0.95)) -> pd.Series:
    data = series.dropna()
    if len(data) > 5:
        lower, upper = data.quantile(limits)
        return series.clip(lower=lower, upper=upper)
    return series


def visualize_outliers(df: pd.DataFrame, features: list):
    st.subheader("🔍 Анализ выбросов")
    n_cols = min(3, len(features))
    for i in range(0, len(features), n_cols):
        cols = st.columns(n_cols)
        for j, feature in enumerate(features[i : i + n_cols]):
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


# ======================
# ШАГ 2: АНАЛИЗ ЗНАЧИМОСТИ
# ======================
def compute_group_sensitivity(
    df: pd.DataFrame,
    group_col: str,
    features: list,
    alpha: float = 0.05,
    mode: str = "group",
) -> pd.DataFrame:
    results = []
    for feature in features:
        x = df[feature].dropna()
        grp_vals = df.loc[x.index, group_col]

        if len(x) < 4 or grp_vals.nunique() < 2:
            continue

        # 1. Kruskal-Wallis
        groups_list = [v.values for _, v in x.groupby(grp_vals)]
        try:
            p_kw = stats.kruskal(*groups_list)[1] if len(groups_list) > 1 else 1.0
        except Exception:
            p_kw = 1.0

        rho, p_spear, p_quad = np.nan, np.nan, np.nan

        if mode == "dose":
            # 2. Спирмен
            try:
                rho, p_spear = stats.spearmanr(grp_vals.astype(float), x)
            except Exception:
                pass

            # 3. Квадратичная регрессия
            if grp_vals.nunique() > 3:
                try:
                    d = grp_vals.astype(float)
                    X = sm.add_constant(pd.DataFrame({"d": d, "d2": d ** 2}))
                    model = sm.OLS(x, X).fit()
                    p_quad = model.pvalues.get("d2", 1.0)
                except Exception:
                    pass

        if mode == "dose":
            is_sensitive = (p_kw < alpha) or (
                not np.isnan(p_spear) and p_spear < alpha and abs(rho) > 0.5
            ) or (not np.isnan(p_quad) and p_quad < alpha)
        else:
            is_sensitive = p_kw < alpha

        results.append(
            {
                "Маркер": feature,
                "p (KW)": round(p_kw, 4),
                "ρ Спирмен": round(rho, 3) if not np.isnan(rho) else "—",
                "p (Spear)": round(p_spear, 4) if not np.isnan(p_spear) else "—",
                "p (квадр.)": round(p_quad, 4) if not np.isnan(p_quad) else "—",
                "Значимый": is_sensitive,
            }
        )
    return pd.DataFrame(results)


# ======================
# ШАГ 3А: КРИТЕРИЙ ЭФФЕКТИВНОСТИ
# ======================
def compute_efficiency(
    df: pd.DataFrame,
    group_col: str,
    control_group,
    me_markers: list,
) -> dict:
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

            shift = (g_vals.median() - c_vals.median()) / iqr_m
            shifts.append(shift)

        E_vals[group] = float(np.nanmedian(shifts)) if shifts else 0.0

    return E_vals


# ======================
# ШАГ 3Б: КРИТЕРИЙ БЕЗОПАСНОСТИ (РИСКА)
# ======================
def compute_safety(
    df: pd.DataFrame,
    group_col: str,
    control_group,
    ms_markers: list,
) -> dict:
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

            shift = (g_vals.median() - c_vals.median()) / iqr_m
            risk = max(0.0, shift)
            risks.append(risk)

        S_vals[group] = float(np.nanmedian(risks)) if risks else 0.0

    return S_vals


# ======================
# ШАГ 3В: КРИТЕРИЙ БАЛАНСА (УНИВЕРСАЛЬНЫЙ)
# ======================
def compute_balance_criterion(
    df: pd.DataFrame,
    group_col: str,
    control_group,
    balance_ratios: list,
    features: list,
) -> dict:
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

                dc_candidates = [
                    c for c in features
                    if den_base.lower() in c.lower() and c.lower().endswith(loc_suffix)
                ]
                if not dc_candidates:
                    continue
                dc = dc_candidates[0]

                grp_data = df[df[group_col] == group]
                ctrl_data = df[df[group_col] == control_group]

                if grp_data.empty or ctrl_data.empty:
                    continue

                d_ratio = (grp_data[nc] / (grp_data[dc] + eps)).median()
                c_ratio = (ctrl_data[nc] / (ctrl_data[dc] + eps)).median()
                all_r = df[nc] / (df[dc] + eps)
                iqr_r = all_r.quantile(0.75) - all_r.quantile(0.25)

                if iqr_r > 0:
                    deviations.append(abs((d_ratio - c_ratio) / iqr_r))

        B_vals[group] = -np.nanmedian(deviations) if deviations else 0.0

    return B_vals


# ======================
# УТИЛИТЫ
# ======================
def normalize_dict(d: dict) -> dict:
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


# ======================
# БУТСТРЭП ДЛЯ I_g
# ======================
def bootstrap_indices(
    df: pd.DataFrame,
    group_col: str,
    control_group,
    me_markers: list,
    ms_markers: list,
    balance_ratios: list,
    features: list,
    w_E: float,
    w_S: float,
    w_B: float,
    n_boot: int = 1000,
    random_state: int = 42,
) -> dict:
    """
    Бутстрэп по объектам внутри каждой группы.
    На каждом шаге:
      - ресэмплинг строк с возвращением в каждой группе,
      - пересчёт E_g, S_g, B_g и I_g.
    Возвращает словарь: group -> список значений I_g (длина n_boot).
    """
    rng = np.random.default_rng(random_state)
    groups = sorted(df[group_col].dropna().unique())
    group_data = {g: df[df[group_col] == g] for g in groups}

    w_total = w_E + w_S + w_B
    if w_total == 0:
        w_E_n = w_S_n = w_B_n = 0.0
    else:
        w_E_n, w_S_n, w_B_n = w_E / w_total, w_S / w_total, w_B / w_total

    I_boot = {g: [] for g in groups}

    for _ in range(int(n_boot)):
        # Бутстрэп-выборка
        df_b = []
        for g in groups:
            g_df = group_data[g]
            if len(g_df) == 0:
                continue
            idx = rng.integers(0, len(g_df), size=len(g_df))
            df_b.append(g_df.iloc[idx])
        df_b = pd.concat(df_b, axis=0)

        # Пересчёт критериев на бутстрэп-выборке
        E_vals_b = compute_efficiency(df_b, group_col, control_group, me_markers)
        if ms_markers:
            S_vals_b = compute_safety(df_b, group_col, control_group, ms_markers)
        else:
            S_vals_b = {g: 0.0 for g in groups}
        if balance_ratios:
            B_vals_b = compute_balance_criterion(df_b, group_col, control_group, balance_ratios, features)
        else:
            B_vals_b = {g: 0.0 for g in groups}

        E_n_b = normalize_dict(E_vals_b)
        S_n_b = normalize_dict(S_vals_b)
        B_n_b = normalize_dict(B_vals_b)

        for g in groups:
            I_g = (
                w_E_n * E_n_b.get(g, 0.5)
                - w_S_n * S_n_b.get(g, 0.5)
                + w_B_n * B_n_b.get(g, 0.5)
            )
            I_boot[g].append(I_g)

    return I_boot


# ======================
# БОКОВАЯ ПАНЕЛЬ: ЗАГРУЗКА И КОНФИГУРАЦИЯ
# ======================
st.sidebar.header("📁 Загрузка данных")
uploaded_file = st.sidebar.file_uploader("CSV или Excel", type=["csv", "xlsx"])

if uploaded_file is not None:
    try:
        if uploaded_file.name.endswith(".csv"):
            df_raw = pd.read_csv(uploaded_file, sep=None, decimal=",", engine="python")
        else:
            df_raw = pd.read_excel(uploaded_file)

        st.session_state.df_raw = df_raw
        st.success(f"✅ Загружено: {len(df_raw)} строк × {len(df_raw.columns)} столбцов")
        st.dataframe(df_raw.head(), use_container_width=True)

        # ── Режим эксперимента ──────────────────────────────────────────────
        st.sidebar.header("🧪 Тип дизайна эксперимента")
        exp_mode = st.sidebar.radio(
            "Структура опытных групп",
            [
                "🔢 Числовые дозы (доза-ответ)",
                "🏷️ Категориальные группы (факторный / комбинации добавок)",
            ],
            help=(
                "Числовые дозы: группы различаются количеством одного вещества (0, 0.2, 0.4 мг/кг). "
                "Категориальные: группы — разные комбинации добавок без единого числового параметра."
            ),
        )
        is_dose_mode = exp_mode.startswith("🔢")

        # ── Столбец групп ────────────────────────────────────────────────────
        st.sidebar.header("⚙️ Столбцы")
        group_col_name = st.sidebar.selectbox(
            "Столбец с группами" + (" (дозы)" if is_dose_mode else " (метки)"),
            df_raw.columns,
        )

        # Создаём служебный столбец _group
        if is_dose_mode:
            group_raw = safe_numeric(df_raw[group_col_name])
            if group_raw.isna().all():
                st.error("❌ Столбец доз не содержит числовых значений!")
                st.stop()
            df_raw["_group"] = group_raw
            unique_groups = sorted(df_raw["_group"].dropna().unique())
        else:
            df_raw["_group"] = df_raw[group_col_name].astype(str).str.strip()
            unique_groups = sorted(df_raw["_group"].dropna().unique(), key=str)

        st.sidebar.markdown(
            f"**Найдено групп ({len(unique_groups)}):** " + ", ".join(str(g) for g in unique_groups)
        )

        # Контрольная группа
        control_group = st.sidebar.selectbox(
            "🎯 Контрольная группа (эталон)",
            options=unique_groups,
            index=0,
        )

        # Числовые признаки для анализа
        exclude_cols = {"_group", group_col_name}
        all_numeric = [
            c for c in df_raw.select_dtypes(include=[np.number]).columns
            if c not in exclude_cols
        ]
        features = st.sidebar.multiselect(
            "📊 Признаки для анализа",
            all_numeric,
            default=all_numeric[: min(20, len(all_numeric))],
        )

        if st.sidebar.button("🚀 Запустить анализ", type="primary") and features:
            df_processed = df_raw.copy()
            for col in features:
                df_processed[col] = safe_numeric(df_raw[col])

            # Сохраняем состояние
            st.session_state.df_processed = df_processed
            st.session_state.features = features
            st.session_state.group_col = "_group"
            st.session_state.control_group = control_group
            st.session_state.unique_groups = unique_groups
            st.session_state.is_dose_mode = is_dose_mode

            # Сбрасываем результаты предыдущего запуска
            for key in ["df_clean", "group_stats", "results", "I_boot"]:
                st.session_state.pop(key, None)

            st.success("✅ Данные подготовлены!")
            safe_rerun()

    except Exception as e:
        import traceback
        st.error(f"❌ Ошибка загрузки: {e}")
        st.code(traceback.format_exc())
        st.stop()


# ======================
# ГЛАВНЫЙ АНАЛИЗ — запускается только после загрузки
# ======================
if "df_processed" in st.session_state:
    df = st.session_state.df_processed
    features = st.session_state.features
    group_col = st.session_state.group_col
    control_group = st.session_state.control_group
    unique_groups = st.session_state.unique_groups
    is_dose_mode = st.session_state.is_dose_mode
    mode_str = "dose" if is_dose_mode else "group"

    st.markdown("---")

    # ── ШАГ 1: Выбросы ──────────────────────────────────────────────────────
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
        st.success("✅ Готово!")
        safe_rerun()

    # ── ШАГ 2: Значимость признаков ─────────────────────────────────────────
    if "df_clean" in st.session_state:
        df_clean = st.session_state.df_clean
        st.markdown("---")

        if is_dose_mode:
            st.header("🔬 Шаг 2 — Дозозависимость маркеров")
            st.caption("Kruskal-Wallis + ранговая корреляция Спирмена + квадратичная регрессия")
        else:
            st.header("🔬 Шаг 2 — Значимость различий между группами")
            st.caption(
                "Критерий Краскела-Уоллиса для каждого маркера. "
                "Тренд и квадратичная регрессия не применяются: группы категориальные."
            )

        alpha_val = st.slider("Уровень значимости α", 0.01, 0.20, 0.05, 0.01)

        if st.button("📊 Рассчитать значимость", key="sens_btn"):
            with st.spinner("Анализ..."):
                group_stats = compute_group_sensitivity(
                    df_clean, group_col, features, alpha=alpha_val, mode=mode_str
                )
                st.session_state.group_stats = group_stats

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

    # ── ШАГ 3: Многокритериальный анализ ────────────────────────────────────
    if "group_stats" in st.session_state:
        st.markdown("---")
        st.header("🎯 Шаг 3 — Многокритериальное ранжирование групп")

        df_clean = st.session_state.df_clean
        gs = st.session_state.group_stats

        # 3.1 Начальный список: значимые маркеры
        sensitive_markers = gs[gs["Значимый"]]["Маркер"].tolist()
        if not sensitive_markers:
            st.warning("⚠️ Значимых маркеров нет — используем все признаки")
            sensitive_markers = features

        # 3.2 Корреляционный фильтр
        with st.expander("🔗 Фильтр мультиколлинеарности", expanded=True):
            corr_threshold = st.slider(
                "Порог корреляции Спирмена для удаления избыточных маркеров",
                0.70, 1.00, 0.90, 0.05,
                help="Из каждой пары с |r| выше порога удаляется второй маркер. "
                     "Спирмен предпочтителен для малых выборок."
            )
            corr_cols = [m for m in sensitive_markers if m in df_clean.columns]
            if len(corr_cols) > 1:
                corr_matrix = df_clean[corr_cols].corr(method="spearman")
                to_drop = set()
                for i in range(len(corr_cols)):
                    for j in range(i + 1, len(corr_cols)):
                        if abs(corr_matrix.iloc[i, j]) > corr_threshold:
                            to_drop.add(corr_cols[j])
                sensitive_markers = [c for c in corr_cols if c not in to_drop]
                if to_drop:
                    st.info(
                        f"Удалено {len(to_drop)} маркеров (|r| > {corr_threshold}). "
                        f"Осталось **{len(sensitive_markers)}** маркеров."
                    )
                    with st.expander("Тепловая карта корреляций"):
                        fig_corr, ax_corr = plt.subplots(
                            figsize=(max(6, len(corr_cols)*0.6), max(5, len(corr_cols)*0.5))
                        )
                        sns.heatmap(
                            corr_matrix, annot=True, fmt=".2f", cmap="RdBu_r",
                            center=0, ax=ax_corr, linewidths=0.5
                        )
                        plt.tight_layout()
                        st.pyplot(fig_corr)
                        plt.close(fig_corr)
                else:
                    st.info("Сильно коррелирующих маркеров не обнаружено.")

        # 3.3 Назначение ролей маркеров
        st.markdown("### 📋 Роли маркеров")
        st.caption(
            "**M_E (эффективность)** — маркеры, рост которых относительно контроля желателен: "
            "цинкзависимые ферменты, антиоксиданты, эссенциальные элементы, продуктивность. "
            "**M_S (безопасность/риск)** — маркеры, рост которых нежелателен: "
            "токсичные элементы, маркеры стресса и патологии. "
            "Один маркер — одна роль."
        )

        role_df = pd.DataFrame({
            "Маркер": sensitive_markers,
            "Роль": [guess_role(m) for m in sensitive_markers],
        })

        edited_roles = st.data_editor(
            role_df,
            column_config={
                "Маркер": st.column_config.TextColumn(disabled=True),
                "Роль": st.column_config.SelectboxColumn(
                    options=["M_E (эффективность)", "M_S (безопасность/риск)", "Игнорировать"]
                ),
            },
            use_container_width=True,
            hide_index=True,
            key="role_editor",
        )

        me_selected = edited_roles[edited_roles["Роль"] == "M_E (эффективность)"]["Маркер"].tolist()
        ms_selected = edited_roles[edited_roles["Роль"] == "M_S (безопасность/риск)"]["Маркер"].tolist()

        col_m1, col_m2 = st.columns(2)
        col_m1.metric("M_E маркеров", len(me_selected))
        col_m2.metric("M_S маркеров", len(ms_selected))

        # 3.4 Критерий баланса (опционально)
        st.markdown("### ⚖️ Критерий баланса (необязательно)")
        use_balance = st.checkbox(
            "Включить критерий баланса физиологических соотношений",
            help="Оценивает, насколько соотношения элементов (Ca/P, Na/K, Zn/Cu) "
                 "отклоняются от контроля. Требует, чтобы оба элемента соотношения "
                 "присутствовали в данных."
        )
        balance_ratios = []
        if use_balance:
            balance_input = st.text_area(
                "Введите соотношения — одно на строку (числитель/знаменатель):",
                placeholder="Ca/P\nNa/K\nZn/Cu\nCa/Fe",
                height=110,
            )
            balance_ratios = [r.strip() for r in balance_input.strip().split("\n") if "/" in r]
            if balance_ratios:
                st.info(f"Соотношения: {', '.join(balance_ratios)}")
            else:
                st.warning("Введите хотя бы одно соотношение в формате 'A/B'")

        # 3.5 Веса критериев
        st.markdown("### 🎚️ Веса критериев")
        n_w = 3 if (use_balance and balance_ratios) else 2
        w_cols = st.columns(n_w)
        w_E = w_cols[0].slider("w₁ Эффективность", 0.0, 1.0, 0.5, 0.05, key="w_e")
        w_S = w_cols[1].slider("w₂ Безопасность", 0.0, 1.0, 0.3 if n_w == 3 else 0.5, 0.05, key="w_s")
        w_B = w_cols[2].slider("w₃ Баланс", 0.0, 1.0, 0.2, 0.05, key="w_b") if n_w == 3 else 0.0

        w_total = w_E + w_S + w_B
        if w_total > 0:
            w_E_n, w_S_n, w_B_n = w_E / w_total, w_S / w_total, w_B / w_total
            st.caption(
                f"Нормализованные веса: E = {w_E_n:.2f} | S = {w_S_n:.2f} | B = {w_B_n:.2f}  "
                f"→ I_g = w_E·E_norm − w_S·S_norm + w_B·B_norm"
            )
        else:
            w_E_n, w_S_n, w_B_n = 1.0, 0.0, 0.0

        # 3.5a Бутстрэп-параметры
        st.markdown("### 🔁 Бутстрэп устойчивости I_g")
        col_bs1, col_bs2 = st.columns(2)
        use_bootstrap = col_bs1.checkbox("Включить бутстрэп по животным", value=False)
        n_boot = col_bs2.number_input(
            "Число итераций бутстрэпа",
            min_value=100, max_value=5000, value=1000, step=100
        )

        # 3.6 Расчёт
        if st.button("🚀 Рассчитать I_g и ранжировать группы", type="primary", key="calc_btn"):
            if not me_selected:
                st.error("❌ Добавьте хотя бы один маркер эффективности (роль M_E)!")
            else:
                with st.spinner("Расчёт..."):
                    E_vals = compute_efficiency(
                        df_clean, group_col, control_group, me_selected
                    )
                    S_vals = (
                        compute_safety(
                            df_clean, group_col, control_group, ms_selected
                        )
                        if ms_selected
                        else {g: 0.0 for g in df_clean[group_col].unique()}
                    )
                    B_vals = (
                        compute_balance_criterion(
                            df_clean, group_col, control_group, balance_ratios, features
                        )
                        if (use_balance and balance_ratios)
                        else {g: 0.0 for g in df_clean[group_col].unique()}
                    )

                    all_groups = list(df_clean[group_col].unique())
                    results = pd.DataFrame(
                        {
                            "Группа": all_groups,
                            "E_g (raw)": [E_vals.get(g, 0) for g in all_groups],
                            "S_g (raw)": [S_vals.get(g, 0) for g in all_groups],
                            "B_g (raw)": [B_vals.get(g, 0) for g in all_groups],
                        }
                    )
                    E_n = normalize_dict(dict(zip(results["Группа"], results["E_g (raw)"])))
                    S_n = normalize_dict(dict(zip(results["Группа"], results["S_g (raw)"])))
                    B_n = normalize_dict(dict(zip(results["Группа"], results["B_g (raw)"])))

                    results["E_norm"] = [E_n[g] for g in all_groups]
                    results["S_norm"] = [S_n[g] for g in all_groups]
                    results["B_norm"] = [B_n[g] for g in all_groups]
                    st.session_state.results = results

                    # Бутстрэп (по желанию)
                    if use_bootstrap:
                        with st.spinner(f"Бутстрэп {int(n_boot)} итераций..."):
                            I_boot = bootstrap_indices(
                                df_clean,
                                group_col=group_col,
                                control_group=control_group,
                                me_markers=me_selected,
                                ms_markers=ms_selected,
                                balance_ratios=balance_ratios if (use_balance and balance_ratios) else [],
                                features=features,
                                w_E=w_E,
                                w_S=w_S,
                                w_B=w_B,
                                n_boot=int(n_boot),
                                random_state=42,
                            )
                            st.session_state.I_boot = I_boot
                    else:
                        st.session_state.pop("I_boot", None)

        # 3.7 Отображение результатов (пересчёт I_g при изменении весов — без повторного расчёта)
        if "results" in st.session_state:
            results = st.session_state.results.copy()
            results["I_g"] = (
                w_E_n * results["E_norm"]
                - w_S_n * results["S_norm"]
                + w_B_n * results["B_norm"]
            )

            st.markdown("### 🏆 Ранжирование групп")
            display_cols = ["Группа", "E_norm", "S_norm", "B_norm", "I_g"]
            st.dataframe(
                results[display_cols].round(3).sort_values("I_g", ascending=False),
                use_container_width=True,
            )

            # Графики критериев
            x_labels = [str(g) for g in results["Группа"]]
            x_pos = list(range(len(x_labels)))
            n_plots = 2 + (1 if use_balance and balance_ratios else 0)
            fig, axes = plt.subplots(1, n_plots, figsize=(6 * n_plots, 5))
            if n_plots == 1:
                axes = [axes]

            axes[0].bar(x_pos, results["E_norm"], color="#2ecc71", alpha=0.85)
            axes[0].set_title("E_norm — Эффективность", fontweight="bold")
            axes[0].set_xticks(x_pos)
            axes[0].set_xticklabels(x_labels, rotation=40, ha="right")

            axes[1].bar(x_pos, results["S_norm"], color="#e74c3c", alpha=0.85)
            axes[1].set_title("S_norm — Риск (безопасность)", fontweight="bold")
            axes[1].set_xticks(x_pos)
            axes[1].set_xticklabels(x_labels, rotation=40, ha="right")

            if n_plots == 3:
                axes[2].bar(x_pos, results["B_norm"], color="#3498db", alpha=0.85)
                axes[2].set_title("B_norm — Баланс соотношений", fontweight="bold")
                axes[2].set_xticks(x_pos)
                axes[2].set_xticklabels(x_labels, rotation=40, ha="right")

            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

            # Интегральный индекс
            best_val = results["I_g"].max()
            fig2, ax2 = plt.subplots(figsize=(max(8, len(x_labels) * 1.2), 5))
            bar_colors = ["#f1c40f" if v == best_val else "#2980b9" for v in results["I_g"]]
            bars = ax2.bar(x_pos, results["I_g"], color=bar_colors, alpha=0.9, edgecolor="white")
            ax2.axhline(y=0, color="gray", linestyle="--", alpha=0.5)
            ax2.set_title(
                "I_g — Интегральный индекс (чем выше, тем лучше)",
                fontweight="bold", fontsize=13
            )
            ax2.set_xticks(x_pos)
            ax2.set_xticklabels(x_labels, rotation=40, ha="right", fontsize=10)
            ax2.set_ylabel("I_g")
            for bar, val in zip(bars, results["I_g"]):
                ax2.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.01,
                    f"{val:.3f}",
                    ha="center", va="bottom", fontsize=9
                )
            plt.tight_layout()
            st.pyplot(fig2)
            plt.close(fig2)

            # Победитель
            best_idx = results["I_g"].idxmax()
            best_group = results.loc[best_idx, "Группа"]
            best_ig = results.loc[best_idx, "I_g"]
            st.success(f"🎉 **Оптимальная группа: {best_group}** (I_g = {best_ig:.4f})")

            m_cols = st.columns(4)
            m_cols[0].metric("Группа", str(best_group))
            m_cols[1].metric("E (эффективность)", f"{results.loc[best_idx, 'E_norm']:.3f}")
            m_cols[2].metric("S (риск) ↓ меньше лучше", f"{results.loc[best_idx, 'S_norm']:.3f}")
            m_cols[3].metric("B (баланс)", f"{results.loc[best_idx, 'B_norm']:.3f}")

                       # Если есть бутстрэп — показываем его статистику
            if "I_boot" in st.session_state:
                st.markdown("### 📊 Бутстрэп устойчивости I_g")
                I_boot = st.session_state.I_boot
                groups_bs = sorted(I_boot.keys())
                n_boot_eff = len(next(iter(I_boot.values()))) if I_boot else 0

                boot_rows = []
                best_counts_all = {g: 0 for g in groups_bs}

                if n_boot_eff > 0:
                    I_mat = np.vstack([I_boot[g] for g in groups_bs]).T  # shape: (n_boot, n_groups)
                    for row in I_mat:
                        idx_best = int(np.argmax(row))
                        g_best = groups_bs[idx_best]
                        best_counts_all[g_best] += 1

                for g in groups_bs:
                    vals = np.array(I_boot[g])
                    med = np.median(vals)
                    low = np.percentile(vals, 2.5)
                    high = np.percentile(vals, 97.5)
                    prob_best_all = best_counts_all[g] / n_boot_eff if n_boot_eff > 0 else np.nan
                    boot_rows.append(
                        {
                            "Группа": g,
                            "I_med": med,
                            "I_2.5%": low,
                            "I_97.5%": high,
                            "P(группа лучшая, с контролем)": prob_best_all,
                        }
                    )

                boot_df = pd.DataFrame(boot_rows).sort_values(
                    "Группа", key=lambda s: s.astype(str)
                )
                st.dataframe(boot_df.round(3), use_container_width=True)

                # Дополнительно: лучшая опытная группа без контроля
                control_str = str(control_group)
                test_groups = [g for g in groups_bs if str(g) != control_str]

                if n_boot_eff > 0 and test_groups:
                    best_counts_test = {g: 0 for g in test_groups}

                    for row in I_mat:
                        vals_test = [row[groups_bs.index(g)] for g in test_groups]
                        idx_best_t = int(np.argmax(vals_test))
                        g_best_t = test_groups[idx_best_t]
                        best_counts_test[g_best_t] += 1

                    test_rows = []
                    for g in test_groups:
                        vals = np.array(I_boot[g])
                        test_rows.append(
                            {
                                "Группа": g,
                                "I_med_без_контроля": np.median(vals),
                                "I_2.5%_без_контроля": np.percentile(vals, 2.5),
                                "I_97.5%_без_контроля": np.percentile(vals, 97.5),
                                "P(лучшая среди опытных)": best_counts_test[g] / n_boot_eff,
                            }
                        )

                    test_df = pd.DataFrame(test_rows).sort_values(
                        "P(лучшая среди опытных)", ascending=False
                    )

                    st.markdown("### 📌 Лучшие опытные группы без контроля")
                    st.dataframe(test_df.round(3), use_container_width=True)

                    best_test_idx = test_df["P(лучшая среди опытных)"].idxmax()
                    best_test_group = test_df.loc[best_test_idx, "Группа"]
                    best_test_prob = test_df.loc[best_test_idx, "P(лучшая среди опытных)"]

                    st.info(
                        f"Лучшая опытная группа без контроля: **{best_test_group}** "
                        f"(P = {best_test_prob:.2f})"
                    )

                fig_bs, ax_bs = plt.subplots(figsize=(max(8, len(groups_bs) * 1.2), 5))
                data_bs = [I_boot[g] for g in groups_bs]
                labels_bs = [str(g) for g in groups_bs]
                ax_bs.boxplot(data_bs, labels=labels_bs, showmeans=True)
                ax_bs.set_xlabel("Группа")
                ax_bs.set_ylabel("I_g (бутстрэп)")
                ax_bs.set_title("Распределение I_g по бутстрэпу")
                plt.tight_layout()
                st.pyplot(fig_bs)
                plt.close(fig_bs)

            # Экспорт
            st.download_button(
                "⬇️ Скачать результаты CSV",
                data=results.round(4).sort_values("I_g", ascending=False).to_csv(index=False, sep=";"),
                file_name="groupranker_results.csv",
                mime="text/csv",
            )

else:
    st.info("📁 Загрузите CSV или Excel в боковой панели для начала работы.")
    st.markdown(
        """
### 🚀 Быстрый старт

| Шаг | Действие |
|-----|----------|
| 1 | Загрузите файл (CSV/Excel) — нужен столбец групп и числовые признаки |
| 2 | Выберите **тип дизайна**: числовые дозы ИЛИ категориальные группы |
| 3 | Укажите **контрольную группу** |
| 4 | Запустите предобработку и анализ значимости |
| 5 | В таблице назначьте каждому маркеру **роль**: M_E (эффективность) или M_S (риск) |
| 6 | Настройте веса и получите **интегральный индекс I_g** |

**Поддерживаемые дизайны:**
- Дозо-ответные эксперименты
- Факторные и комбинаторные
- Любые эксперименты с группами
"""
    )

st.markdown("---")
st.markdown("*© 2026 GroupRanker Pro | Универсальная модель многокритериального ранжирования*")

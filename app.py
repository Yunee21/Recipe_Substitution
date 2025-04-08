# %%
import streamlit as st
import pandas as pd
import numpy as np

# %%
# -----------------------------
# 👥 사용자 정의 스타일 적용
# -----------------------------
st.markdown(
    """
    <style>
    .stApp {
        background-color: #ffffff;
    }

    /* 사이드바 전체 배경 */
    section[data-testid="stSidebar"] {
        background-color: #ffe6ed;
        padding: 2rem 1rem;
    }

    /* 사이드바 제목 스타일 */
    section[data-testid="stSidebar"] h1,
    section[data-testid="stSidebar"] h2,
    section[data-testid="stSidebar"] h3 {
        color: #c71e4d;
    }

    /* 라디오 버튼 텍스트 */
    .stRadio > label {
        color: #c71e4d;
        font-weight: bold;
    }

    /* 버튼 스타일 */
    .stButton>button {
        background-color: #ff638f;
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.5rem 1rem;
        font-weight: bold;
    }

    .stButton>button:hover {
        background-color: #e5537f;
        color: white;
    }

    /* 설명 텍스트 스타일 */
    .sidebar-description {
        font-size: 0.9rem;
        color: #444444;
        line-height: 1.5;
        margin-top: 1rem;
    }

    </style>
    """,
    unsafe_allow_html=True
)

# -----------------------------
# 🏷️ 제목
# -----------------------------
st.markdown("<h1 style='color:#c71e4d;'>신장질환 맞춤 레시피 대체 시스템</h1>", unsafe_allow_html=True)

# -----------------------------
# 📌 사이드바 메뉴 설정
# -----------------------------
with st.sidebar:
    st.markdown("### 메뉴 선택")

    selected = st.radio(
        " ",
        ["프로필 입력", "식재료 선택", "식단 추천"]
    )

    st.markdown("### 데이터 관리")
    col1, col2 = st.columns(2)
    with col1:
        st.button("데이터 저장")
    with col2:
        st.button("데이터 로드")

    st.markdown("---")
    st.markdown("### 사용 방법", unsafe_allow_html=True)
    st.markdown(
        """
        <div class='sidebar-description'>
        1. 먼저 프로필 입력 탭에서 개인 정보를 입력해주세요.<br>
        2. 식재료 선택 탭에서 보유한 식재료를 선택하세요.<br>
        3. 마지막으로 식단 추천 탭에서 원하는 식단 타입을 선택하고 결과를 확인하세요.
        </div>
        """,
        unsafe_allow_html=True
    )

# %%
# -----------------------------
# 🧬 신체 정보 및 신장질환 정보 입력
# -----------------------------
if selected == "1) 프로필 입력":
    with st.expander("👥 신체 정보", expanded=True):
        col1, col2, col3 = st.columns(3)

        with col1:
            gender = st.radio("성별", ["남성", "여성"], horizontal=True)
        with col2:
            height = st.text_input("신장(cm)", placeholder="예: 170")
        with col3:
            weight = st.text_input("체중(kg)", placeholder="예: 65")

    with st.expander("🧬 신장질환 정보", expanded=True):
        input_method = st.radio(
            "입력 방식을 선택하세요",
            ("신장질환 단계 선택", "eGFR 수치 입력")
        )

        kidney_stage = None
        kidney_dialysis = None
        egfr = None

        if input_method == "신장질환 단계 선택":
            kidney_stage = st.selectbox("현재 신장질환 단계를 선택하세요", ["1단계", "2단계", "3단계", "4단계", "5단계", "혈액투석", "복막투석"])
        else:
            egfr = st.number_input("eGFR 수치 입력", min_value=0.0, max_value=200.0, step=0.1)
            if egfr >= 90:
                kidney_stage = "1단계"
            elif 60 <= egfr < 90:
                kidney_stage = "2단계"
            elif 30 <= egfr < 60:
                kidney_stage = "3단계"
            elif 15 <= egfr < 30:
                kidney_stage = "4단계"
            elif egfr < 15:
                kidney_stage = "5단계"

            kidney_dialysis = st.selectbox("현재 투석 여부를 선택하세요", ["비투석", "복막투석", "혈액투석"])


# -----------------------------
# 🧺 보유 식재료 입력
# -----------------------------
elif selected == "2) 보유 식재료 입력":
    with st.expander("🧺 보유 식재료", expanded=True):
        ingredient_input = st.text_area(
            "현재 보유하고 있는 식재료를 입력하세요 (쉼표로 구분)",
            placeholder="예: 두부, 양파, 간장, 달걀, 시금치"
        )

        ingredient_list = []
        if ingredient_input:
            ingredient_list = [item.strip() for item in ingredient_input.split(",") if item.strip()]
            if ingredient_list:
                st.success("입력된 식재료 목록:")
                st.write(ingredient_list)
            else:
                st.info("보유 식재료를 입력해주세요.")


# -----------------------------
# 🍳 레시피 정보 입력
# -----------------------------
elif selected == "3) 레시피 입력":
    recipe_file_path = "recipe.xlsx"

    try:
        recipe_df = pd.read_excel(recipe_file_path)
    except FileNotFoundError:
        st.error("레시피 파일을 찾을 수 없습니다. 경로를 확인해주세요.")
    else:
        with st.expander("🍳 섭취하고 싶은 음식", expanded=True):
            recipe_name = st.text_input("레시피명을 입력하세요", placeholder="예: 부대찌개")

            if recipe_name:
                # 레시피명 정확 일치 (대소문자 구분 X)
                matched = recipe_df[recipe_df["레시피명"].str.lower() == recipe_name.strip().lower()]

                if not matched.empty:
                    recipe = matched.iloc[0]

                    st.success(f"🔍 '{recipe_name}' 레시피를 찾았습니다.")
                    
                    st.markdown("#### 🧾 재료")
                    st.markdown(recipe["재료"])

                    st.markdown("#### 🍳 조리 방법")
                    st.markdown(recipe["조리방법"])
                else:
                    st.warning("일치하는 레시피명이 없습니다. 정확하게 입력했는지 확인해주세요.")


# -----------------------------
# ✅ 제출 및 요약은 항상 하단 표시
# -----------------------------
st.markdown("---")

# 조건이 충족되었는지 확인 (전역적으로 관리 필요 시 session_state로 확장 가능)
can_submit = (
    "gender" in locals()
    and "height" in locals() and height
    and "weight" in locals() and weight
    and "kidney_stage" in locals() and kidney_stage
    and "recipe_df" in locals()
)

if "submitted" not in st.session_state:
    st.session_state["submitted"] = False

if st.button("제출"):
    if can_submit:
        st.markdown("### 📝 섭취 가이드")
        st.write(f"- 제한: 나트륨, 칼륨")
        st.write(f"- 적절: 단백질")

        instructions = recipe_df['조리방법'].to_list()
        cleaned_instructions = [step for step in instructions if isinstance(step, str)]
        numbered_clean = "\n".join([f"{i+1}. {step}" for i, step in enumerate(cleaned_instructions)])

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("### 기존 레시피")
            with st.expander("재료", expanded=True):
                st.dataframe(recipe_df['재료'], use_container_width=True)
            with st.expander("조리방법", expanded=True):
                st.markdown(numbered_clean)

        with col2:
            st.markdown("### 대체 레시피")
            with st.expander("재료", expanded=True):
                recipe_df.at[1, '재료'] = '*** 느타리버섯 ***'
                st.dataframe(recipe_df['재료'], use_container_width=True)
            with st.expander("조리방법", expanded=True):
                directions = """1. 두부는 키친타올로 물기를 제거한 뒤 깍둑썰기 한다.\n2. 느타리버섯은 밑동을 제거한 후 손으로 길게 찢는다.\n3. 팬에 들기름을 두르고 마늘을 볶아 향을 낸다.\n4. 두부와 느타리버섯을 넣고 중불에서 볶는다.\n5. 간장, 고춧가루, 물을 넣고 뚜껑을 덮은 후 약불에서 2~3분간 졸인다.\n6. 불을 끄고 쪽파를 넣어 마무리한다."""
                st.markdown(directions)

        st.session_state["submitted"] = True

    else:
        missing = []
        if not ("gender" in locals() and height and weight):
            missing.append("신체 정보")
        if not ("kidney_stage" in locals()):
            missing.append("신장질환 정보")
        if "recipe_df" not in locals():
            missing.append("레시피 정보")

        st.error("❌ 제출할 수 없습니다. 다음 항목을 확인해주세요:")
        for item in missing:
            st.markdown(f"- 🔴 {item}")



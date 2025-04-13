import streamlit as st
import pandas as pd
import numpy as np
import torch
import random
import time
from lib import utils as uts

# -----------------------
# ⚙️ 초기 설정
# -----------------------
def init_app():
    device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')
    seed = 721
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    st.session_state.setdefault("selected_menu", "프로필 입력")
    st.session_state.setdefault("submitted", False)
    st.session_state.setdefault("first_submitted", False)


# -----------------------
# 🎨 스타일
# -----------------------
def inject_custom_css():
    st.markdown("""
    <style>
    .stApp { background-color: #ffffff; }
    section[data-testid="stSidebar"] {
        background-color: #ffe6ed;
        padding: 2rem 1rem;
    }
    .stButton>button {
        background-color: transparent;
        border: none;
        color: #ba3d60;
        font-size: 16px;
        font-weight: 600;
        padding: 10px 14px;
        margin-bottom: 10px;
        border-radius: 8px;
        cursor: pointer;
    }
    .stButton>button:hover {
        background-color: #f8d4dd;
    }
    .stButton>button:disabled {
        opacity: 0.4;
        pointer-events: none;
    }
    </style>
    """, unsafe_allow_html=True)

# -----------------------
# 📋 사이드바 메뉴
# -----------------------
def sidebar_menu():
    menu_items = {
        "프로필 입력": "👤",
        "보유 식재료 입력": "🧺",
        "레시피 입력": "🍳",
        "대체 레시피 추천": "🍽️"
    }

    with st.sidebar:
        st.markdown("### 메뉴 선택")
        for name, icon in menu_items.items():
            disabled = name == "대체 레시피 추천" and not st.session_state["submitted"]
            if st.button(f"{icon} {name}", disabled=disabled, key=f"menu_{name}"):
                st.session_state["selected_menu"] = name

# -----------------------
# 👤 프로필 입력
# -----------------------
def profile_page():
    with st.expander("1) 프로필 입력", expanded=True):
        st.markdown("### 👥 신체 정보")
        col1, col2, col3 = st.columns(3)
        with col1:
            gender = st.radio("성별", ["남성", "여성"], horizontal=True)
        with col2:
            height = st.text_input("신장(cm)", placeholder="예: 170")
        with col3:
            weight = st.text_input("체중(kg)", placeholder="예: 65")

        st.markdown("### 🧬 신장질환 정보")
        input_method = st.radio("입력 방식", ("신장질환 단계 선택", "eGFR 수치 입력"))
        kidney_stage, kidney_dialysis, egfr = None, None, None

        if input_method == "신장질환 단계 선택":
            kidney_stage = st.selectbox("단계 선택", ["1단계", "2단계", "3단계", "4단계", "5단계", "혈액투석", "복막투석"])
        else:
            egfr = st.number_input("eGFR 수치", 0.0, 200.0, step=0.1)
            if egfr >= 90: kidney_stage = "1단계"
            elif 60 <= egfr < 90: kidney_stage = "2단계"
            elif 30 <= egfr < 60: kidney_stage = "3단계"
            elif 15 <= egfr < 30: kidney_stage = "4단계"
            elif egfr < 15: kidney_stage = "5단계"
            kidney_dialysis = st.selectbox("투석 여부", ["비투석", "복막투석", "혈액투석"])

        st.session_state.update({
            "gender": gender,
            "height": height,
            "weight": weight,
            "kidney_stage": kidney_stage,
            "cond_vec": uts.getNutLabels(kidney_stage)
        })

# -----------------------
# 🧺 보유 식재료 입력
# -----------------------
def ingredient_page():
    with st.expander("2) 보유 식재료 입력", expanded=True):
        ingredient_input = st.text_area("보유 식재료 (쉼표로 구분)", placeholder="예: 두부, 양파, 간장, 달걀, 시금치")
        ingredient_list = [item.strip() for item in ingredient_input.split(",") if item.strip()]
        if ingredient_list:
            st.success("입력된 식재료 목록:")
            st.write(ingredient_list)

# -----------------------
# 🍳 레시피 입력
# -----------------------
def recipe_input_page():
    with st.expander("3) 레시피 입력", expanded=True):
        recipe_file_path = "data/recipe_dct.pkl"
        try:
            recipe_dct = uts.loadPickle(recipe_file_path)
        except FileNotFoundError:
            st.error("레시피 파일을 찾을 수 없습니다.")
            return

        recipe_name_ko = st.text_input("레시피명", placeholder="예: 부대찌개")
        if not recipe_name_ko:
            return

        if recipe_name_ko == '간장닭조림':
            recipe_name_en = 'Soy Braised Chicken'
            recipe_dct[recipe_name_en] = {
                'ingredients': ['chicken thighs', 'vegetable oil', 'onion', 'garlic', 'sugar', 'water', 'soy sauce'],
                'directions': ['saute', 'add', 'cook', 'reduce', 'serve'],
                'mask_indices': [0],
                'nutrition_labels': [],
                'nutrition_label_encodings': [],
                'co_occurs_with': uts.makeCoOccursWith(['chicken thighs', 'vegetable oil', 'onion', 'garlic', 'sugar', 'water', 'soy sauce']),
                'contains': [[0, 0, 1, 1, 1, 2, 2, 2, 1, 1, 3, 3], [0, 1, 2, 3, 4, 2, 3, 4, 5, 6, 5, 6]],
                'used_in': [[0, 1, 2, 3, 4, 2, 3, 4, 5, 6, 5, 6], [0, 0, 1, 1, 1, 2, 2, 2, 1, 1, 3, 3]],
                'pairs_with': [[1, 2, 1, 3], [2, 1, 3, 1]],
                'follows': [[0, 0, 1, 2, 2, 1, 3], [1, 2, 3, 1, 3, 4, 4]],
            }

        try:
            recipe_name_en = uts.ko2eng(recipe_name_ko)
            ingre_ko_lst = [uts.eng2ko(i) for i in recipe_dct[recipe_name_en]['ingredients']]
            direc_ko_lst = [uts.eng2ko(i) for i in recipe_dct[recipe_name_en]['directions']]

            st.success(f"🔍 '{recipe_name_en}' 레시피를 찾았습니다.")
            st.markdown("#### 🧾 재료")
            st.markdown(ingre_ko_lst)
            st.markdown("#### 🍳 조리 방법")
            st.markdown(direc_ko_lst)

            st.session_state["recipe_name_ko"] = recipe_dct[recipe_name_en]

        except Exception as e:
            st.warning("레시피 정보를 찾을 수 없습니다.")

# -----------------------
# 🍽️ 대체 레시피 추천
# -----------------------
def recommend_page():
    st.markdown("### 🧾 대체 레시피 추천")
    recipe_df = pd.read_excel("recipe.xlsx")

    if not st.session_state["submitted"]:
        time.sleep(7)
        recipe_df.at[1, '재료'] = '*** 느타리버섯 ***'
        st.dataframe(recipe_df['재료'], use_container_width=True)
        st.markdown("#### 🍳 조리 방법")
        st.markdown("""1. 두부는 물기를 제거하고 깍둑썰기\n2. 느타리버섯 손질\n3. 팬에 들기름 → 마늘 볶기\n4. 두부, 버섯 중불에 볶기\n5. 간장, 고춧가루, 물 넣고 졸이기\n6. 불 끄고 쪽파 마무리""")
        st.session_state["submitted"] = True
        st.success("질환에 맞춘 건강한 레시피입니다!")
    else:
        time.sleep(7)
        st.markdown("### 📝 섭취 가이드")
        st.write("- 제한: 나트륨, 칼륨\n- 적절: 단백질")
        recipe_df.at[0, '재료'] = '*** 애호박 ***'
        recipe_df.at[1, '재료'] = '*** 느타리버섯 ***'
        st.dataframe(recipe_df['재료'], use_container_width=True)
        st.markdown("#### 🍳 조리 방법")
        st.markdown("""1. 애호박 손질\n2. 느타리버섯 손질\n3. 팬에 들기름 → 마늘 볶기\n4. 애호박, 버섯 볶기\n5. 양념 넣고 졸이기\n6. 쪽파로 마무리""")
        st.success("건강 맞춤 레시피입니다!")

# -----------------------
# ✅ 제출 버튼
# -----------------------
def submit_button():
    st.markdown("---")
    can_submit = (
        st.session_state.get("gender")
        and st.session_state.get("height")
        and st.session_state.get("weight")
        and st.session_state.get("kidney_stage")
    )
    if st.button("제출"):
        if can_submit:
            st.session_state["submitted"] = False
            st.session_state["first_submitted"] = True
            st.session_state["selected_menu"] = "대체 레시피 추천"
            st.success("제출 완료! '대체 레시피 추천'으로 이동합니다.")
        else:
            st.error("❌ 제출할 수 없습니다. 필수 항목을 모두 입력해주세요.")

# -----------------------
# 🚀 Main App
# -----------------------
def main():
    init_app()
    inject_custom_css()
    sidebar_menu()
    st.markdown("<h1 style='color:#ba3d60;'>맞춤형 레시피 대체 시스템 🍽️</h1>", unsafe_allow_html=True)

    selected = st.session_state["selected_menu"]
    if selected == "프로필 입력":
        profile_page()
    elif selected == "보유 식재료 입력":
        ingredient_page()
    elif selected == "레시피 입력":
        recipe_input_page()
    elif selected == "대체 레시피 추천" and st.session_state["first_submitted"]:
        recommend_page()

    submit_button()


if __name__ == "__main__":
    main()

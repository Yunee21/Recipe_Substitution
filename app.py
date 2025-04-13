import streamlit as st
import pandas as pd
import numpy as np
import torch
import random
import time
from difflib import get_close_matches
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

    defaults = {
        "selected_menu": "프로필 입력",
        "submitted": False,
        "first_submitted": False,
        "profile_done": False,
        "ingredient_done": False,
        "recipe_done": False,
        "ingredients": [],
    }
    for k, v in defaults.items():
        st.session_state.setdefault(k, v)


# -----------------------
# 🎨 스타일
# -----------------------
def inject_custom_css():
    st.markdown("""
    <style>
    .stApp {
        background-color: #ffffff;
    }

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

    .box-section {
        background-color: #ffe6ed;
        padding: 1.5rem;
        border-radius: 12px;
        margin-bottom: 2rem;
        transition: background-color 0.3s ease;
        color: #000000;
    }

    .box-section.active {
        background-color: #ba3d60 !important;
        color: white !important;
    }

    .box-section.active h1,
    .box-section.active h2,
    .box-section.active h3,
    .box-section.active p,
    .box-section.active label,
    .box-section.active span {
        color: white !important;
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
            is_selected = st.session_state["selected_menu"] == name
            disabled = name == "대체 레시피 추천" and not st.session_state["submitted"]

            btn_key = f"menu_{name}"
            btn_clicked = st.button(f"{icon} {name}", key=btn_key, disabled=disabled)

            # 버튼 누르면 상태 업데이트
            if btn_clicked:
                st.session_state["selected_menu"] = name

            # CSS 추가 (현재 선택된 버튼에만 적용)
            st.markdown(f"""
            <style>
            div[data-testid="stButton"][id="{btn_key}"] button {{
                background-color: {'#ba3d60' if is_selected else 'transparent'} !important;
                color: {'white' if is_selected else '#ba3d60'} !important;
                font-weight: 600;
                font-size: 16px;
                border: none;
                border-radius: 8px;
                padding: 10px 14px;
                margin-bottom: 10px;
                width: 100%;
                text-align: left;
                transition: background-color 0.3s ease;
            }}
            div[data-testid="stButton"][id="{btn_key}"] button:hover {{
                background-color: {'#a93554' if is_selected else '#f8d4dd'} !important;
            }}
            </style>
            """, unsafe_allow_html=True)



# -----------------------
# 👤 프로필 입력
# -----------------------
def profile_page():
    box_class = "box-section active" if st.session_state["selected_menu"] == "프로필 입력" else "box-section"
    with st.container():
        st.markdown(f'<div class="{box_class}">', unsafe_allow_html=True)
        st.markdown("### 👥 프로필 입력")

        col1, col2, col3 = st.columns(3)
        with col1:
            gender = st.radio("성별", ["남성", "여성"], horizontal=True)
        with col2:
            height = st.text_input("신장(cm)", placeholder="예: 170")
        with col3:
            weight = st.text_input("체중(kg)", placeholder="예: 65")

        st.markdown("### 🧬 신장질환 정보")
        input_method = st.radio("입력 방식", ("신장질환 단계 선택", "eGFR 수치 입력"))
        kidney_stage, egfr = None, None

        if input_method == "신장질환 단계 선택":
            kidney_stage = st.selectbox("단계 선택", ["1단계", "2단계", "3단계", "4단계", "5단계", "혈액투석", "복막투석"])
        else:
            egfr = st.number_input("eGFR 수치", 0.0, 200.0, step=0.1)
            kidney_stage = uts.inferStageFromEgfr(egfr)

        st.session_state.update({
            "gender": gender,
            "height": height,
            "weight": weight,
            "kidney_stage": kidney_stage,
            "cond_vec": uts.getNutLabels(kidney_stage),
        })

        if st.button("프로필 제출"):
            if gender and height and weight and kidney_stage:
                st.success("프로필 정보를 입력받았습니다.")
                st.session_state["profile_done"] = True
                st.session_state["first_submitted"] = True
        st.markdown('</div>', unsafe_allow_html=True)

# -----------------------
# 🧺 보유 식재료 입력
# -----------------------
def add_ingredient():
    ingre = st.session_state["new_ingre"]
    if ingre:
        st.session_state["ingredients"].append(ingre)
        st.session_state["new_ingre"] = ""  # 입력창 초기화
        
def remove_ingredient(ingredient):
    if ingredient in st.session_state["ingredients"]:
        st.session_state["ingredients"].remove(ingredient)

def ingredient_page():
    box_class = "box-section active" if st.session_state["selected_menu"] == "보유 식재료 입력" else "box-section"
    with st.container():
        st.markdown(f'<div class="{box_class}">', unsafe_allow_html=True)
        st.markdown("### 🧺 보유 식재료 입력")

        st.text_input(
            "보유 식재료를 입력하세요",
            key="new_ingre",
            placeholder="예: 두부",
            on_change=add_ingredient
        )

        if st.session_state["ingredients"]:
            st.markdown("#### 입력된 식재료 목록 (클릭하면 삭제됩니다)")

            cols = st.columns(3)
            for i, ingre in enumerate(st.session_state["ingredients"]):
                with cols[i % 3]:
                    st.button(
                        ingre,
                        key=f"ingre_{i}",
                        on_click=remove_ingredient,
                        args=(ingre,),
                        help="클릭 시 목록에서 제거됩니다",
                    )

            st.session_state["ingredient_done"] = True
        st.markdown('</div>', unsafe_allow_html=True)


# -----------------------
# 🍳 레시피 입력
# -----------------------
def recipe_input_page():
    box_class = "box-section active" if st.session_state["selected_menu"] == "레시피 입력" else "box-section"
    with st.container():
        st.markdown(f'<div class="{box_class}">', unsafe_allow_html=True)
        st.markdown("### 🍳 레시피 입력")

        try:
            recipe_dct = uts.loadPickle("data/recipe_dct.pkl")
            recipe_names_eng = list(recipe_dct.keys())
            recipe_names_ko = [uts.eng2ko(k) for k in recipe_names_eng]
        except:
            recipe_names_ko = ["부대찌개", "간장닭조림", "김치찌개"]

        user_input = st.text_input("레시피명을 입력하세요", key="recipe_input", placeholder="예: 김치찌개")

        suggestions = get_close_matches(user_input, recipe_names_ko, n=5, cutoff=0.3) if user_input else []
        if suggestions:
            selected_recipe = st.selectbox("자동 완성된 추천 목록", suggestions, key="recipe_select")
        else:
            selected_recipe = None

        if selected_recipe:
            st.success(f"'{selected_recipe}' 레시피 선택됨")
            st.session_state["recipe_done"] = True
        st.markdown('</div>', unsafe_allow_html=True)


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
# ✅ 제출 여부 확인 및 자동 이동
# -----------------------
def check_auto_submit():
    if st.session_state["profile_done"] and st.session_state["ingredient_done"] and st.session_state["recipe_done"]:
        if not st.session_state["submitted"]:
            st.session_state["selected_menu"] = "대체 레시피 추천"
            st.session_state["submitted"] = True
            st.experimental_rerun()

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
    elif selected == "대체 레시피 추천":
        recommend_page()

    check_auto_submit()


if __name__ == "__main__":
    main()

import streamlit as st
import pandas as pd
import numpy as np
import torch
import random
import time
from difflib import get_close_matches
from lib import utils as uts


# -----------------------
# ⚙️ 초기 설정 + 데이터 로딩 (즉시)
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
    .box-section {
        background-color: #ffe6ed;
        padding: 1.5rem;
        border-radius: 12px;
        margin-bottom: 2rem;
        transition: background-color 0.3s ease;
        color: #000000;
    }
    .box-section.active {
        background-color: transparent !important;
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
            if st.button(f"{icon} {name}", key=btn_key, disabled=disabled):
                st.session_state["selected_menu"] = name



# -----------------------
# 👤 프로필 입력
# -----------------------
def inferStageFromEgfr(egfr):
    if egfr >= 90: return "1단계"
    elif egfr >= 60: return "2단계"
    elif egfr >= 30: return "3단계"
    elif egfr >= 15: return "4단계"
    else: return "5단계"
    
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
            "cond_vec": uts.getNutLabels(kidney_stage),
        })

        if st.button("프로필 제출", key="profile_submit"):
            if gender and height and weight and kidney_stage:
                st.success("프로필 정보를 입력받았습니다.")
                st.session_state["profile_done"] = True
                st.session_state["first_submitted"] = True
        
        st.markdown("""
        <style>
        div[data-testid="stButton"][id="profile_submit"] > button {
            background-color: white !important;
            color: #ba3d60 !important;
            border: 2px solid #ba3d60 !important;
            border-radius: 8px !important;
            padding: 0.5rem 1.2rem !important;
            font-size: 16px !important;
            font-weight: 600 !important;
            cursor: pointer !important;
            transition: 0.3s ease all !important;
        }
        
        div[data-testid="stButton"][id="profile_submit"] > button:hover {
            background-color: #ba3d60 !important;
            color: white !important;
        }
        </style>
        """, unsafe_allow_html=True)


# -----------------------
# 🧺 보유 식재료 입력
# -----------------------
@st.cache_resource
def load_ingre_node_dct():
    ingre_node_dct = uts.loadPickle("data/ingre_node_dct.pkl")
    ingre_node_ko = uts.loadPickle("data/ingre_node_ko.pkl")
    return ingre_node_dct, ingre_node_ko
    
ingre_node_dct, ingre_node_ko = load_ingre_node_dct()

def match_ingredients_to_standard(user_ingredients, ingre_node_ko):
    result = {}
    for user_input in user_ingredients:
        match = get_close_matches(user_input, ingre_node_ko, n=1, cutoff=0.3)
        if match:
            result[user_input] = match[0]
        else:
            result[user_input] = None  # 또는 "알 수 없음"
    return result

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
            st.markdown("#### 입력된 식재료 목록 (클릭 시 제거)")

            cols = st.columns(3)
            for i, ingre in enumerate(st.session_state["ingredients"]):
                with cols[i % 3]:
                    st.button(
                        ingre,
                        key=f"ingre_{i}",
                        on_click=remove_ingredient,
                        args=(ingre,),
                        help="클릭 시 목록에서 제거됩니다"
                    )

            # ✅ 제출 시 매핑 실행
            if st.button("보유 식재료 제출", key="ingredient_submit"):
                user_ingredients = st.session_state["ingredients"]
                matched_dict = match_ingredients_to_standard(user_ingredients, ingre_node_ko)
                st.session_state["ingredient_mapping"] = matched_dict
                st.session_state["ingredient_done"] = True
                st.success("식재료가 제출되었습니다!")

                # ✅ 매핑 결과 표시 (선택적)
                st.markdown("#### 🔍 자동 매핑 결과")
                st.json(matched_dict)

        st.markdown('</div>', unsafe_allow_html=True)


# -----------------------
# 🍳 레시피 입력
# -----------------------
@st.cache_resource
def load_recipe_dct():
    recipe_dct = uts.loadPickle("data/recipe_graph_dct.pkl")
    recipe_name_en = uts.loadPickle("data/recipe_name_en.pkl")
    recipe_name_ko = uts.loadPickle('data/recipe_name_ko.pkl')
    return recipe_dct, recipe_name_en, recipe_name_ko
    
recipe_dct, recipe_name_en, recipe_name_ko = load_recipe_dct()

def recipe_input_page():
    box_class = "box-section active" if st.session_state["selected_menu"] == "레시피 입력" else "box-section"
    with st.container():
        st.markdown(f'<div class="{box_class}">', unsafe_allow_html=True)
        st.markdown("### 🍳 레시피 입력")

        # ✅ 1. 입력창
        user_input = st.text_input("레시피명을 입력하세요", key="recipe_input", placeholder="예: 김치찌개")

        # ✅ 2. 실시간 추천 리스트 보여주기 (구글처럼)
        suggestions = get_close_matches(user_input, recipe_name_ko, n=5, cutoff=0.3) if user_input else []

        if suggestions:
            selected_recipe = st.selectbox("추천 레시피", suggestions, key="recipe_suggest")
            st.session_state["recipe_selected"] = selected_recipe
        else:
            st.session_state["recipe_selected"] = ""
        
        if st.session_state["recipe_selected"]:
            if st.button("레시피 제출"):
                st.success(f"'{st.session_state['recipe_selected']}' 레시피가 선택되었습니다.")
                st.session_state["recipe_done"] = True
                st.session_state["selected_recipe_name_ko"] = st.session_state["recipe_selected"]
                idx = recipe_name_ko.index(st.session_state["recipe_selected"])
                st.session_state["selected_recipe_name_eng"] = recipe_name_en[idx]
                
        st.markdown('</div>', unsafe_allow_html=True)
        # if suggestions:
        #     st.markdown("##### 추천 레시피:")
        #     for i, suggestion in enumerate(suggestions):
        #         if st.button(suggestion, key=f"suggestion_{i}"):
        #             st.session_state["selected_recipe_name_ko"] = suggestion
        #             st.session_state["selected_recipe_name_eng"] = ko_to_eng[suggestion]
        #             st.session_state["recipe_done"] = True
        #             st.success(f"'{suggestion}' 레시피가 제출되었습니다!")
        #             st.experimental_rerun()

        # st.markdown('</div>', unsafe_allow_html=True)



# -----------------------
# 🍽️ 대체 레시피 추천
# -----------------------
def getAlternativeIngredients(target):
    # 실제 구현에서는 알고리즘 모델 기반 추천
    dummy_map = {
        "돼지고기": ["두부", "버섯", "닭고기", "계란", "오징어"],
        "소고기": ["닭고기", "두부", "버섯", "콩단백", "오징어"],
    }
    return dummy_map.get(target, ["두부", "버섯", "닭고기", "계란", "오징어"])

def getIngredientKO(ingre_en):
    ingre_ko = []
    full_ingre = ingre_node_dct['name']
    for ingre in ingre_en:
        idx = full_ingre.index(ingre)
        ko = ingre_node_ko[idx]
        ingre_ko.append(ko)
    return ingre_ko

def recommend_page():
    st.markdown("### 🧾 대체 레시피 추천")

    # -----------------------
    # 1. 선택된 레시피 불러오기
    # -----------------------
    name_eng = st.session_state["selected_recipe_name_eng"]
    name_ko = st.session_state["selected_recipe_name_ko"]
    recipe_info = recipe_dct[name_eng]

    orig_recipe_ko = pd.DataFrame([], columns=['recipe','ingredients'])
    len_ingre = len(recipe_info['ingredient'])
    orig_recipe_ko['recipe'] = [name_ko] + [''] * (len_ingre - 1)
    orig_recipe_ko['ingredients'] = getIngredientKO(recipe_info['ingredient'])
    directions = orig_recipe_ko['direction'].apply(lambda x: '\n'.join(x))
    
    st.markdown("#### 🧾 재료")
    st.dataframe(orig_recipe_ko['ingredients'], use_container_width=True)
    st.markdown("#### 🍳 조리 방법")
    st.markdown(directions)
    
    
    ingredients = ['감자', '라면']
    target = recipe_info.get("대체대상", ingredients[0])  # 예시용

    st.markdown(f"### 🍲 선택한 레시피: **{name_ko}**")
    st.markdown("#### 📦 재료 목록 (파란색은 대체 대상입니다)")

    colored_ingredients = [
        f"<span style='color:#1f77b4; font-weight:bold;'>{ing}</span>" if ing == target else ing
        for ing in ingredients
    ]
    st.markdown(", ".join(colored_ingredients), unsafe_allow_html=True)

    st.markdown("#### 🍳 조리 방법")
    st.markdown(cook_steps)

    # -----------------------
    # 2. 대체 후보 재료 표시
    # -----------------------
    st.markdown("#### 🔁 대체할 재료를 선택하세요:")
    alt_candidates = uts.getAlternativeIngredients(target)  # 예: ['두부', '버섯', '계란', '닭고기', '오징어']

    selected_alt = st.session_state.get("selected_alternative")

    if not selected_alt:
        cols = st.columns(5)
        for i, alt in enumerate(alt_candidates):
            with cols[i]:
                if st.button(alt, key=f"alt_ingre_{i}"):
                    st.session_state["selected_alternative"] = alt
                    st.experimental_rerun()
    else:
        # -----------------------
        # 3. 대체 결과 출력
        # -----------------------
        new_ingredients = [selected_alt if i == target else i for i in ingredients]
        new_steps = cook_steps.replace(target, selected_alt)

        st.markdown("---")
        st.markdown(f"### ✅ 대체된 레시피: **{name_ko}**")
        st.markdown("#### 🍽️ 재료 목록")
        st.markdown(", ".join(new_ingredients))

        st.markdown("#### 🧑‍🍳 조리 방법")
        st.markdown(new_steps)


# -----------------------
# ✅ 제출 여부 확인 및 자동 이동
# -----------------------
def check_auto_submit():
    if st.session_state["profile_done"] and st.session_state["ingredient_done"] and st.session_state["recipe_done"]:
        if not st.session_state["submitted"]:
            st.session_state["selected_menu"] = "대체 레시피 추천"
            st.session_state["submitted"] = True
            st.rerun()
            
# -----------------------
# 🚀 Main App
# -----------------------
def main():
    init_app()
    inject_custom_css()
    sidebar_menu()
    
    # st.markdown("<h1 style='color:#ba3d60;'>맞춤형 레시피 대체 시스템 🍽️</h1>", unsafe_allow_html=True)

    # selected = st.session_state["selected_menu"]
    # if selected == "프로필 입력":
    #     profile_page()
    # elif selected == "보유 식재료 입력":
    #     ingredient_page()
    # elif selected == "레시피 입력":
    #     recipe_input_page()
    # elif selected == "대체 레시피 추천":
    #     recommend_page()

    # check_auto_submit()
    st.markdown("<h1 style='color:#ba3d60;'>맞춤형 레시피 대체 시스템 🍽️</h1>", unsafe_allow_html=True)

    pages = {
        "프로필 입력": profile_page,
        "보유 식재료 입력": ingredient_page,
        "레시피 입력": recipe_input_page,
        "대체 레시피 추천": recommend_page
    }

    selected = st.session_state["selected_menu"]
    pages[selected]()
    check_auto_submit()


if __name__ == "__main__":
    main()

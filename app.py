# %%
import streamlit as st
import pandas as pd
import numpy as np
import torch
import random
from lib import utils as uts


device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')
seed = 721
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)

# %%
# -----------------------------
# 📌 상태 초기화
# -----------------------------
if "selected_menu" not in st.session_state:
    st.session_state["selected_menu"] = "프로필 입력"
if "submitted" not in st.session_state:
    st.session_state["submitted"] = False

def set_menu(menu_name):
    st.session_state["selected_menu"] = menu_name

selected = st.session_state["selected_menu"]

# -----------------------------
# 🎨 사용자 정의 스타일
# -----------------------------
st.markdown(
    """
    <style>
    .stApp { background-color: #ffffff; }

    section[data-testid="stSidebar"] {
        background-color: #ffe6ed;
        padding: 2rem 1rem;
    }

    .menu-button {
        display: flex;
        align-items: center;
        gap: 0.5rem;
        padding: 10px 14px;
        margin-bottom: 10px;
        font-size: 16px;
        font-weight: 600;
        border-radius: 8px;
        cursor: pointer;
        background-color: transparent;
        color: #ba3d60;
        transition: all 0.2s ease;
    }

    .menu-button:hover {
        background-color: #f8d4dd;
    }

    .menu-button.selected {
        background-color: #ba3d60 !important;
        color: white !important;
    }

    .menu-button.disabled {
        opacity: 0.4;
        pointer-events: none;
    }

    .stButton>button {
        /* display: none; */
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
    .stButton>button.selected {
        background-color: #ba3d60 !important;
        color: white !important;
    }
    .stButton>button.disabled {
        opacity: 0.4;
        pointer-events: none;
    }
    
    .sidebar-description {
        font-size: 0.9rem;
        color: #444;
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
st.markdown("<h1 style='color:#ba3d60;'>맞춤형 레시피 대체 시스템 🍽️</h1>", unsafe_allow_html=True)

# -----------------------------
# 📌 메뉴 구성
# -----------------------------
menu_items = {
    "프로필 입력": "👤",
    "보유 식재료 입력": "🧺",
    "레시피 입력": "🍳",
    "대체 레시피 추천": "🍽️"
}

# -----------------------------
# 📌 사이드바 메뉴 출력 (중복 없이)
# -----------------------------
with st.sidebar:
    st.markdown("### 메뉴 선택")
    for name, icon in menu_items.items():
        is_selected = (selected == name)
        is_disabled = name == "대체 레시피 추천" and not st.session_state["submitted"]

        class_names = ["menu-button"]
        if is_selected:
            class_names.append("selected")
        if is_disabled:
            class_names.append("disabled")

        # 버튼 클릭 처리만 실제로 감지 (표시는 안 보임)
        if st.button(f"{icon} {name}", key=f"click_{name}"):
            if not is_disabled:
                set_menu(name)
        


    st.markdown("---")
    st.markdown("""
        <div class='sidebar-description'>
        1. 프로필, 보유 식재료, 레시피 정보를 입력하고 제출해주세요.<br>
        2. 제출 후 '대체 레시피 추천' 메뉴에서 추천 결과를 확인할 수 있습니다.
        </div>
    """, unsafe_allow_html=True)

# -----------------------------
# 👤 프로필 입력
# -----------------------------

if selected == "프로필 입력":
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
        
        st.session_state["gender"] = gender
        st.session_state["height"] = height
        st.session_state["weight"] = weight
        st.session_state["kidney_stage"] = kidney_stage
        st.session_state["cond_vec"] = uts.getNutLabels(kidney_stage)

# -----------------------------
# 🧺 보유 식재료 입력
# -----------------------------
elif selected == "보유 식재료 입력":
    with st.expander("2) 보유 식재료 입력", expanded=True):
        ingredient_input = st.text_area(
            "보유 식재료 (쉼표로 구분)", placeholder="예: 두부, 양파, 간장, 달걀, 시금치"
        )
        ingredient_list = [item.strip() for item in ingredient_input.split(",") if item.strip()]
        if ingredient_list:
            st.success("입력된 식재료 목록:")
            st.write(ingredient_list)

# -----------------------------
# 🍳 레시피 입력
# -----------------------------

elif selected == "레시피 입력":
    with st.expander("3) 레시피 입력", expanded=True):
        #recipe_file_path = "recipe.xlsx"
        recipe_file_path = "data/recipe_dct.pkl"
        try:
            #recipe_df = pd.read_excel(recipe_file_path)
            recipe_dct = uts.loadPickle(recipe_file_path)
        except FileNotFoundError:
            st.error("레시피 파일을 찾을 수 없습니다.")
        else:
            recipe_name_ko = st.text_input("레시피명", placeholder="예: 부대찌개")
            if recipe_name_ko:
        
                if (recipe_name_ko == '간장닭조림'):
                    recipe_name_en = 'Soy Braised Chicken'
                    recipe_dct[recipe_name_en] = {
                        'ingredients': ['chicken thighs', 'vegetable oil', 'onion', 'garlic', 'sugar', 'water', 'soy sauce'],
                        'directions': ['saute', 'add', 'cook', 'reduce', 'serve'],
                        'mask_indices': [0],
                        'nutrition_labels': [],
                        'nutrition_label_encodings': [],
                        'co_occurs_with': uts.makeCoOccursWith(['chicken thighs', 'vegetable oil', 'onion', 'garlic', 'sugar', 'water', 'soy sauce']),
                        'contains': [[0, 0, 1, 1, 1, 2, 2, 2, 1, 1, 3, 3],
                                     [0, 1, 2, 3, 4, 2, 3, 4, 5, 6, 5, 6]],
                        'used_in': [[0, 1, 2, 3, 4, 2, 3, 4, 5, 6, 5, 6],
                                    [0, 0, 1, 1, 1, 2, 2, 2, 1, 1, 3, 3]],
                        'pairs_with': [[1,2,1,3],
                                       [2,1,3,1]],
                        'follows': [[0,0,1,2,2,1,3],
                                    [1,2,3,1,3,4,4]],
                    }

                elif (recipe_name_ko == '매콤 두부 가지볶음'):
                    recipe_df = pd.read_excel("recipe.xlsx")
                    instructions = recipe_df['조리방법'].to_list()
                    cleaned_instructions = [step for step in instructions if isinstance(step, str)]
                    numbered_clean = "\n".join([f"{i+1}. {step}" for i, step in enumerate(cleaned_instructions)])
                    st.markdown("#### 🧾 재료")
                    st.dataframe(recipe_df['재료'], use_container_width=True)
                    st.markdown("#### 🍳 조리 방법")
                    st.markdown(numbered_clean)

                    
                
                else:
                    recipe_name_en = uts.ko2eng(recipe_name_ko)

                    ingre_ko_lst = [uts.eng2ko(ingre_en) for ingre_en in recipe_dct[recipe_name_en]['ingredients']]
                    direc_ko_lst = [uts.eng2ko(direc_en) for direc_en in recipe_dct[recipe_name_en]['directions']]
                    
                    st.success(f"🔍 '{recipe_name_en}' 레시피를 찾았습니다.")
                    st.markdown("#### 🧾 재료")
                    st.markdown(ingre_ko_lst)
                    st.markdown("#### 🍳 조리 방법")
                    st.markdown(direc_ko_lst)
    
                    st.session_state["recipe_name_ko"] = recipe_dct[recipe_name_en]

                         

# -----------------------------
# ✅ 제출
# -----------------------------
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
        set_menu("대체 레시피 추천")  # 제출 직후 자동으로 이동
        st.success("제출이 완료되었습니다. '대체 레시피 추천' 메뉴를 확인해보세요.")
    else:
        st.error("❌ 제출할 수 없습니다. 필수 항목을 모두 입력해주세요.")

# -----------------------------
# 🍽️ 대체 레시피 추천
# -----------------------------
if selected == "대체 레시피 추천" and st.session_state["first_submitted"]:

    recipe_df = pd.read_excel("recipe.xlsx")

    if (not st.session_state["submitted"]):
        
        st.markdown("#### 🧾 재료")
        
        recipe_df.at[1, '재료'] = '*** 느타리버섯 ***'
        st.dataframe(recipe_df['재료'], use_container_width=True)
    
        st.markdown("#### 🍳 조리 방법")
        directions = """1. 두부는 키친타올로 물기를 제거한 뒤 깍둑썰기 한다. \n2. 느타리버섯은 밑동을 제거한 후 손으로 길게 찢는다. \n3. 팬에 들기름을 두르고 마늘을 볶아 향을 낸다. \n4. 두부와 느타리버섯을 넣고 중불에서 볶는다. \n5. 간장, 고춧가루, 물을 넣고 뚜껑을 덮은 후 약불에서 2~3분간 졸인다. \n 6.불을 끄고 쪽파를 넣어 마무리한다."""
        st.markdown(directions)
    
        st.session_state["submitted"] = True
        st.success("질환에 맞춘 건강한 레시피입니다!")

    else:
        st.markdown("### 📝 섭취 가이드")
        st.write(f"- 제한: 나트륨, 칼륨")
        st.write(f"- 적절: 단백질")

        instructions = recipe_df['조리방법'].to_list()
        cleaned_instructions = [step for step in instructions if isinstance(step, str)]
        numbered_clean = "\n".join([f"{i+1}. {step}" for i, step in enumerate(cleaned_instructions)])

        recipe_df.at[0, '재료'] = '*** 애호박 ***'
        recipe_df.at[1, '재료'] = '*** 느타리버섯 ***'
        st.dataframe(recipe_df['재료'], use_container_width=True)
    
        st.markdown("#### 🍳 조리 방법")
        directions = """1. 애호박은 반으로 갈라 어슷하게 썬다. \n2. 느타리버섯은 밑동을 제거한 후 손으로 길게 찢는다. \n3. 팬에 들기름을 두르고 마늘을 볶아 향을 낸다. \n4. 애호박과 느타리버섯을 넣고 중불에서 볶는다. \n5. 간장, 고춧가루, 물을 넣고 뚜껑을 덮은 후 약불에서 2~3분간 졸인다. \n6. 불을 끄고 쪽파를 넣어 마무리한다."""
        st.markdown(directions)
            
    
        st.success("질환에 맞춘 건강한 레시피입니다!")


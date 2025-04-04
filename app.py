# %%
import streamlit as st
import pandas as pd
import numpy as np

# %%
st.title("신장질환 맞춤 레시피 대체 시스템")

# -----------------------------
# 👥 신체 정보 입력 (3열 구성)
# -----------------------------
with st.expander("👥 신체 정보", expanded=True):
    col1, col2, col3 = st.columns(3)

    with col1:
        gender = st.radio("성별", ["남성", "여성"], horizontal=True)
    with col2:
        height = st.text_input("신장(cm)", placeholder="예: 170")
    with col3:
        weight = st.text_input("체중(kg)", placeholder="예: 65")

# -----------------------------
# 🧬 신장질환 정보 입력
# -----------------------------
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

# %%
# -----------------------------
# 🧺 보유 식재료 입력
# -----------------------------
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

# %%
# -----------------------------
# 🍳 레시피 정보 입력
# -----------------------------
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
            else:
                st.warning("일치하는 레시피명이 없습니다. 정확하게 입력했는지 확인해주세요.")


# %%
# -----------------------------
# ✅ 제출 버튼 및 요약
# -----------------------------
st.markdown("---")

can_submit = (
    gender and height and weight
    and kidney_stage
    and 'recipe_df' in locals()
)

# 제출 상태 관리
if "submitted" not in st.session_state:
    st.session_state["submitted"] = False

if st.button("제출"):
    if can_submit:
        if not st.session_state["submitted"]:
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
                    directions = """1. 두부는 키친타올로 물기를 제거한 뒤 깍둑썰기 한다. \n2. 느타리버섯은 밑동을 제거한 후 손으로 길게 찢는다. \n3. 팬에 들기름을 두르고 마늘을 볶아 향을 낸다. \n4. 두부와 느타리버섯을 넣고 중불에서 볶는다. \n5. 간장, 고춧가루, 물을 넣고 뚜껑을 덮은 후 약불에서 2~3분간 졸인다. \n 6.불을 끄고 쪽파를 넣어 마무리한다."""
                    st.markdown(directions)

            st.session_state["submitted"] = True


        else:
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
                    recipe_df.at[0, '재료'] = '*** 애호박 ***'
                    recipe_df.at[1, '재료'] = '*** 느타리버섯 ***'
                    st.dataframe(recipe_df['재료'], use_container_width=True)
                with st.expander("조리방법", expanded=True):
                    directions = """1. 애호박은 반으로 갈라 어슷하게 썬다. \n2. 느타리버섯은 밑동을 제거한 후 손으로 길게 찢는다. \n3. 팬에 들기름을 두르고 마늘을 볶아 향을 낸다. \n4. 애호박과 느타리버섯을 넣고 중불에서 볶는다. \n5. 간장, 고춧가루, 물을 넣고 뚜껑을 덮은 후 약불에서 2~3분간 졸인다. \n6. 불을 끄고 쪽파를 넣어 마무리한다."""
                    st.markdown(directions)

    else:
        # 누락 항목 파악
        missing = []
        if not gender or not height or not weight:
            missing.append("신체 정보")
        if not kidney_stage or not kidney_dialysis:
            missing.append("신장질환 정보")
        if 'recipe_df' not in locals():
            missing.append("레시피 정보")

        st.error("❌ 제출할 수 없습니다. 다음 항목을 확인해주세요:")
        for item in missing:
            st.markdown(f"- 🔴 {item}")


